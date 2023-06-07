// Check if node
if (typeof module === 'object' && module.exports) {
  var tf = require('@tensorflow/tfjs');
}

function log() {
  console.log('[PPO]', ...arguments);
}

class PPO {
  constructor(env, config) {
    const configDefault = {
      nSteps: 512,
      nEpochs: 10,
      policyLearningRate: 1e-3,
      valueLearningRate: 1e-3,
      clipRatio: 0.2,
      targetKL: 0.01,
      useSDE: false, // TODO: State Dependent Exploration (gSDE)
      netArch: {
        pi: [32, 32],
        vf: [32, 32],
      },
      activation: 'relu',
      verbose: 0,
    };
    this.config = Object.assign({}, configDefault, config);

    // Prepare network architecture
    if (Array.isArray(this.config.netArch)) {
      this.config.netArch = {
        pi: this.config.netArch,
        vf: this.config.netArch,
      };
    }

    // Initialize logger
    this.log = (...args) => {
      if (this.config.verbose > 0) {
        console.log('[PPO]', ...args);
      }
    };

    // Initialize environment
    this.env = env;
    if (
      this.env.actionSpace.class == 'Discrete' &&
      !this.env.actionSpace.dtype
    ) {
      this.env.actionSpace.dtype = 'int32';
    } else if (
      this.env.actionSpace.class == 'Box' &&
      !this.env.actionSpace.dtype
    ) {
      this.env.actionSpace.dtype = 'float32';
    }

    // Initialize counters
    this.numTimesteps = 0;
    this.lastObservation = null;

    // Initialize buffer
    this.buffer = new StateBuffer(config);

    // Initialize models for actor and critic
    this.actor = this.createActor();
    this.critic = this.createCritic();

    // Initialize logStd (for continuous action space)
    if (this.env.actionSpace.class == 'Box') {
      this.logStd = tf.variable(
        tf.zeros([this.env.actionSpace.shape[0]]),
        true,
        'logStd'
      );
    }

    // Initialize optimizers
    this.optPolicy = tf.train.adam(this.config.policyLearningRate);
    this.optValue = tf.train.adam(this.config.valueLearningRate);
  }

  createActor() {
    const input = tf.layers.input({ shape: this.env.observationSpace.shape });
    let l = input;
    this.config.netArch.pi.forEach((units, i) => {
      l = tf.layers
        .dense({
          units,
          activation: this.config.activation,
        })
        .apply(l);
    });
    if (this.env.actionSpace.class == 'Discrete') {
      l = tf.layers
        .dense({
          units: this.env.actionSpace.n,
          // kernelInitializer: 'glorotNormal'
        })
        .apply(l);
    } else if (this.env.actionSpace.class == 'Box') {
      l = tf.layers
        .dense({
          units: this.env.actionSpace.shape[0],
          // kernelInitializer: 'glorotNormal'
        })
        .apply(l);
    } else {
      throw new Error(
        'Unknown action space class: ' + this.env.actionSpace.class
      );
    }
    return tf.model({ inputs: input, outputs: l });
  }

  createCritic() {
    // Initialize critic
    const input = tf.layers.input({ shape: this.env.observationSpace.shape });
    let l = input;
    this.config.netArch.vf.forEach((units, i) => {
      l = tf.layers
        .dense({
          units: units,
          activation: this.config.activation,
        })
        .apply(l);
    });
    l = tf.layers
      .dense({
        units: 1,
        activation: 'linear',
      })
      .apply(l);
    return tf.model({ inputs: input, outputs: l });
  }

  sampleAction(observationT) {
    return tf.tidy(() => {
      const preds = tf.squeeze(this.actor.predict(observationT), 0);
      let action;
      if (this.env.actionSpace.class == 'Discrete') {
        action = tf.squeeze(tf.multinomial(preds, 1), 0); // > []
      } else if (this.env.actionSpace.class == 'Box') {
        action = tf.add(
          tf.mul(
            tf.randomStandardNormal([this.env.actionSpace.shape[0]]),
            tf.exp(this.logStd)
          ),
          preds
        ); // > [actionSpace.shape[0]]
      }
      return [preds, action];
    });
  }

  logProbCategorical(logits, x) {
    return tf.tidy(() => {
      const numActions = logits.shape[logits.shape.length - 1];
      const logprobabilitiesAll = tf.logSoftmax(logits);
      return tf.sum(
        tf.mul(tf.oneHot(x, numActions), logprobabilitiesAll),
        logprobabilitiesAll.shape.length - 1
      );
    });
  }

  logProbNormal(loc, scale, x) {
    return tf.tidy(() => {
      const logUnnormalized = tf.mul(
        -0.5,
        tf.square(tf.sub(tf.div(x, scale), tf.div(loc, scale)))
      );
      const logNormalization = tf.add(
        tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
        tf.log(scale)
      );
      return tf.sum(
        tf.sub(logUnnormalized, logNormalization),
        logUnnormalized.shape.length - 1
      );
    });
  }

  logProb(preds, actions) {
    // Preds can be logits or means
    if (this.env.actionSpace.class == 'Discrete') {
      return this.logProbCategorical(preds, actions);
    } else if (this.env.actionSpace.class == 'Box') {
      return this.logProbNormal(preds, tf.exp(this.logStd), actions);
    }
  }

  predict(observation, deterministic = false) {
    return this.actor.predict(observation);
  }

  trainPolicy(
    observationBufferT,
    actionBufferT,
    logprobabilityBufferT,
    advantageBufferT
  ) {
    const optFunc = () => {
      const predsT = this.actor.predict(observationBufferT); // -> Logits or means
      const diffT = tf.sub(
        this.logProb(predsT, actionBufferT),
        logprobabilityBufferT
      );
      const ratioT = tf.exp(diffT);
      const minAdvantageT = tf.where(
        tf.greater(advantageBufferT, 0),
        tf.mul(tf.add(1, this.config.clipRatio), advantageBufferT),
        tf.mul(tf.sub(1, this.config.clipRatio), advantageBufferT)
      );
      const policyLoss = tf.neg(
        tf.mean(tf.minimum(tf.mul(ratioT, advantageBufferT), minAdvantageT))
      );
      return policyLoss;
    };

    return tf.tidy(() => {
      const { values, grads } = this.optPolicy.computeGradients(optFunc);
      this.optPolicy.applyGradients(grads);
      const kl = tf.mean(
        tf.sub(
          logprobabilityBufferT,
          this.logProb(this.actor.predict(observationBufferT), actionBufferT)
        )
      );
      return kl.arraySync();
    });
  }

  trainValue(observationBufferT, returnBufferT) {
    const optFunc = () => {
      const valuesPredT = this.critic.predict(observationBufferT);
      return tf.losses.meanSquaredError(returnBufferT, valuesPredT);
    };

    tf.tidy(() => {
      const { values, grads } = this.optValue.computeGradients(optFunc);
      this.optValue.applyGradients(grads);
    });
  }

  _initCallback(callback) {
    // Function, not class
    if (typeof callback === 'function') {
      if (callback.prototype.constructor === undefined) {
        return new FunctionalCallback(callback);
      }
      return callback;
    }
    if (typeof callback === 'object') {
      return new DictCallback(callback);
    }
    return new BaseCallback();
  }

  async collectRollouts(callback) {
    if (this.lastObservation === null) {
      this.lastObservation = this.env.reset();
    }

    this.buffer.reset();
    callback.onRolloutStart(this);

    let sumReturn = 0;
    let sumLength = 0;
    let numEpisodes = 0;

    const allPreds = [];
    const allActions = [];
    const allClippedActions = [];

    for (let step = 0; step < this.config.nSteps; step++) {
      // Predict action, value and logprob from last observation
      const [preds, action, value, logprobability] = tf.tidy(() => {
        const lastObservationT = tf.tensor([this.lastObservation]);
        const [predsT, actionT] = this.sampleAction(lastObservationT);
        const valueT = this.critic.predict(lastObservationT);
        const logprobabilityT = this.logProb(predsT, actionT);
        return [
          predsT.arraySync(), // -> Discrete: [actionSpace.n] or Box: [actionSpace.shape[0]]
          actionT.arraySync(), // -> Discrete: [] or Box: [actionSpace.shape[0]]
          valueT.arraySync()[0][0],
          logprobabilityT.arraySync(),
        ];
      });
      allPreds.push(preds);
      allActions.push(action);

      // Rescale for continuous action space
      let clippedAction = action;
      if (this.env.actionSpace.class == 'Box') {
        let h = this.env.actionSpace.high;
        let l = this.env.actionSpace.low;
        if (typeof h === 'number' && typeof l === 'number') {
          clippedAction = action.map((a) => {
            return Math.min(Math.max(a, l), h);
          });
        }
      }
      allClippedActions.push(clippedAction);

      // Take action in environment
      const [newObservation, reward, done] = await this.env.step(clippedAction);
      sumReturn += reward;
      sumLength += 1;

      // Update global timestep counter
      this.numTimesteps += 1;

      callback.onStep(this);

      this.buffer.add(
        this.lastObservation,
        action,
        reward,
        value,
        logprobability
      );

      this.lastObservation = newObservation;

      if (done || step === this.config.nSteps - 1) {
        const lastValue = done
          ? 0
          : tf.tidy(() =>
              this.critic.predict(tf.tensor([newObservation])).arraySync()
            )[0][0];
        this.buffer.finishTrajectory(lastValue);
        numEpisodes += 1;
        this.lastObservation = this.env.reset();
      }
    }

    callback.onRolloutEnd(this);
  }

  async train(config) {
    // Get values from the buffer
    const [
      observationBuffer,
      actionBuffer,
      advantageBuffer,
      returnBuffer,
      logprobabilityBuffer,
    ] = this.buffer.get();

    const [
      observationBufferT,
      actionBufferT,
      advantageBufferT,
      returnBufferT,
      logprobabilityBufferT,
    ] = tf.tidy(() => [
      tf.tensor(observationBuffer),
      tf.tensor(actionBuffer, null, this.env.actionSpace.dtype),
      tf.tensor(advantageBuffer),
      tf.tensor(returnBuffer).reshape([-1, 1]),
      tf.tensor(logprobabilityBuffer),
    ]);

    for (let i = 0; i < this.config.nEpochs; i++) {
      const kl = this.trainPolicy(
        observationBufferT,
        actionBufferT,
        logprobabilityBufferT,
        advantageBufferT
      );
      if (kl > 1.5 * this.config.targetKL) {
        break;
      }
    }

    for (let i = 0; i < this.config.nEpochs; i++) {
      this.trainValue(observationBufferT, returnBufferT);
    }

    tf.dispose([
      observationBufferT,
      actionBufferT,
      advantageBufferT,
      returnBufferT,
      logprobabilityBufferT,
    ]);
  }

  async learn(learnConfig) {
    const learnConfigDefault = {
      totalTimesteps: 1000,
      logInterval: 1,
      callback: null,
    };
    let { totalTimesteps, logInterval, callback } = Object.assign(
      {},
      learnConfigDefault,
      learnConfig
    );

    callback = this._initCallback(callback);

    let iteration = 0;

    callback.onTrainingStart(this);

    while (this.numTimesteps < totalTimesteps) {
      await this.collectRollouts(callback);
      iteration += 1;
      if (logInterval && iteration % logInterval === 0) {
        log(`Timesteps: ${this.numTimesteps}`);
      }
      this.train();
    }

    callback.onTrainingEnd(this);
  }
}

if (typeof module === 'object' && module.exports) {
  module.exports = PPO;
}
