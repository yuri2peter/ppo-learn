import * as tfDefault from '@tensorflow/tfjs';
import { StateBuffer } from './Buffer';
import { Env } from './Env';
import {
  BoxActionSpace,
  DiscreteActionSpace,
  ModelsData,
  ObservationSpace,
  PPOParams,
  StepObservation,
  TrainingConfig,
  trainingConfigDefault,
} from './defines';
import { LayersModel } from '@tensorflow/tfjs';
import { exportModel, importModel } from './utils';
import { PPOEvent } from './event';

export class PPO {
  config: TrainingConfig;
  eventEmitter = new PPOEvent();
  env: Env;
  numTimesteps: number;
  lastObservation: null | StepObservation;
  buffer: StateBuffer;
  actor: LayersModel;
  critic: LayersModel;
  logStd: any;
  tf: any;
  optPolicy: any;
  optValue: any;
  envActionSpace: BoxActionSpace | DiscreteActionSpace;
  envObservationSpace: ObservationSpace;

  constructor({ env, config = {}, tf = tfDefault }: PPOParams) {
    this.config = Object.assign({}, trainingConfigDefault, config);
    this.tf = tf;

    // Initialize environment
    this.env = env;
    this.envActionSpace = env.getActionSpace();
    this.envObservationSpace = env.getObservationSpace();

    // Initialize counters
    this.numTimesteps = 0;
    this.lastObservation = null;

    // Initialize buffer
    this.buffer = new StateBuffer({ ...config, tf });

    // Initialize models for actor and critic
    this.actor = this.createActor();
    this.critic = this.createCritic();

    // Initialize logStd (for continuous action space)
    if (this.envActionSpace.class == 'Box') {
      this.logStd = this.tf.variable(
        this.tf.zeros([this.envActionSpace.shape[0]]),
        true,
        'logStd'
      );
    }

    // Initialize optimizers
    this.optPolicy = this.tf.train.adam(this.config.policyLearningRate);
    this.optValue = this.tf.train.adam(this.config.valueLearningRate);
  }

  log(...args: any) {
    if (this.config.verbose > 0) {
      console.log('[PPO]', ...args);
    }
  }

  createActor() {
    const input = this.tf.layers.input({
      shape: this.envObservationSpace.shape,
    });
    let l = input;
    this.config.netArch.pi.forEach((units: any, i: any) => {
      l = this.tf.layers
        .dense({
          units,
          activation: this.config.activation,
        })
        .apply(l);
    });
    if (this.envActionSpace.class == 'Discrete') {
      l = this.tf.layers
        .dense({
          units: this.envActionSpace.n,
          // kernelInitializer: 'glorotNormal'
        })
        .apply(l);
    } else if (this.envActionSpace.class == 'Box') {
      l = this.tf.layers
        .dense({
          units: this.envActionSpace.shape[0],
          // kernelInitializer: 'glorotNormal'
        })
        .apply(l);
    } else {
      throw new Error('Unknown action space class');
    }
    return this.tf.model({ inputs: input, outputs: l });
  }

  createCritic() {
    // Initialize critic
    const input = this.tf.layers.input({
      shape: this.envObservationSpace.shape,
    });
    let l = input;
    this.config.netArch.vf.forEach((units: any, i: any) => {
      l = this.tf.layers
        .dense({
          units: units,
          activation: this.config.activation,
        })
        .apply(l);
    });
    l = this.tf.layers
      .dense({
        units: 1,
        activation: 'linear',
      })
      .apply(l);
    return this.tf.model({ inputs: input, outputs: l });
  }

  sampleAction(observationT: any) {
    return this.tf.tidy(() => {
      const preds = this.tf.squeeze(this.actor.predict(observationT), 0);
      let action;
      if (this.envActionSpace.class == 'Discrete') {
        action = this.tf.squeeze(this.tf.multinomial(preds, 1), 0); // > []
      } else if (this.envActionSpace.class == 'Box') {
        action = this.tf.add(
          this.tf.mul(
            this.tf.randomStandardNormal([this.envActionSpace.shape[0]]),
            this.tf.exp(this.logStd)
          ),
          preds
        ); // > [actionSpace.shape[0]]
      }
      return [preds, action];
    });
  }

  logProbCategorical(logits: { shape: string | any[] }, x: any) {
    return this.tf.tidy(() => {
      const numActions = logits.shape[logits.shape.length - 1];
      const logprobabilitiesAll = this.tf.logSoftmax(logits);
      return this.tf.sum(
        this.tf.mul(this.tf.oneHot(x, numActions), logprobabilitiesAll),
        logprobabilitiesAll.shape.length - 1
      );
    });
  }

  logProbNormal(loc: any, scale: any, x: any) {
    return this.tf.tidy(() => {
      const logUnnormalized = this.tf.mul(
        -0.5,
        this.tf.square(
          this.tf.sub(this.tf.div(x, scale), this.tf.div(loc, scale))
        )
      );
      const logNormalization = this.tf.add(
        this.tf.scalar(0.5 * Math.log(2.0 * Math.PI)),
        this.tf.log(scale)
      );
      return this.tf.sum(
        this.tf.sub(logUnnormalized, logNormalization),
        logUnnormalized.shape.length - 1
      );
    });
  }

  logProb(preds: any, actions: any) {
    // Preds can be logits or means
    if (this.envActionSpace.class == 'Discrete') {
      return this.logProbCategorical(preds, actions);
    } else if (this.envActionSpace.class == 'Box') {
      return this.logProbNormal(preds, this.tf.exp(this.logStd), actions);
    }
  }

  predict(observation: any, deterministic = false) {
    return this.actor.predict(observation);
  }

  trainPolicy(
    observationBufferT: any,
    actionBufferT: any,
    logprobabilityBufferT: any,
    advantageBufferT: any
  ) {
    const optFunc = () => {
      const predsT = this.actor.predict(observationBufferT); // -> Logits or means
      const diffT = this.tf.sub(
        this.logProb(predsT, actionBufferT),
        logprobabilityBufferT
      );
      const ratioT = this.tf.exp(diffT);
      const minAdvantageT = this.tf.where(
        this.tf.greater(advantageBufferT, 0),
        this.tf.mul(this.tf.add(1, this.config.clipRatio), advantageBufferT),
        this.tf.mul(this.tf.sub(1, this.config.clipRatio), advantageBufferT)
      );
      const policyLoss = this.tf.neg(
        this.tf.mean(
          this.tf.minimum(this.tf.mul(ratioT, advantageBufferT), minAdvantageT)
        )
      );
      return policyLoss;
    };

    return this.tf.tidy(() => {
      const { values, grads } = this.optPolicy.computeGradients(optFunc);
      this.optPolicy.applyGradients(grads);
      const kl = this.tf.mean(
        this.tf.sub(
          logprobabilityBufferT,
          this.logProb(this.actor.predict(observationBufferT), actionBufferT)
        )
      );
      return kl.arraySync();
    });
  }

  trainValue(observationBufferT: any, returnBufferT: any) {
    const optFunc = () => {
      const valuesPredT = this.critic.predict(observationBufferT);
      return this.tf.losses.meanSquaredError(returnBufferT, valuesPredT);
    };

    this.tf.tidy(() => {
      const { values, grads } = this.optValue.computeGradients(optFunc);
      this.optValue.applyGradients(grads);
    });
  }

  async collectRollouts() {
    if (this.lastObservation === null) {
      this.lastObservation = await this.env.reset();
    }

    this.buffer.reset();

    let sumReturn = 0;
    let numEpisodes = 0;

    const allPreds = [];
    const allActions = [];
    const allClippedActions = [];

    for (let step = 0; step < this.config.nSteps; step++) {
      this.eventEmitter.emitStepStart({
        step,
        nSteps: this.config.nSteps,
      });
      // Predict action, value and logprob from last observation
      const [preds, action, value, logprobability] = this.tf.tidy(() => {
        const lastObservationT = this.tf.tensor([this.lastObservation]);
        const [predsT, actionT] = this.sampleAction(lastObservationT);
        const valueT = this.critic.predict(lastObservationT);
        const logprobabilityT = this.logProb(predsT, actionT);
        return [
          predsT.arraySync(), // -> Discrete: [actionSpace.n] or Box: [actionSpace.shape[0]]
          actionT.arraySync(), // -> Discrete: [] or Box: [actionSpace.shape[0]]
          (valueT as any).arraySync()[0][0],
          logprobabilityT.arraySync(),
        ];
      });
      allPreds.push(preds);
      allActions.push(action);

      // Rescale for continuous action space
      let clippedAction = action;
      allClippedActions.push(clippedAction);

      // Take action in environment
      const [newObservation, reward, done] = await this.env.step(clippedAction);
      sumReturn += reward;

      // Update global timestep counter
      this.numTimesteps += 1;

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
          : this.tf.tidy(() =>
              (
                this.critic.predict(this.tf.tensor([newObservation])) as any
              ).arraySync()
            )[0][0];
        this.buffer.finishTrajectory(lastValue);
        numEpisodes += 1;
        this.lastObservation = await this.env.reset();
      }
      this.eventEmitter.emitStepEnd({
        step,
        nSteps: this.config.nSteps,
        episodes: numEpisodes,
        reward,
        done,
        value,
        sumReturn,
      });
    }
  }

  async train() {
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
    ] = this.tf.tidy(() => [
      this.tf.tensor(observationBuffer),
      this.tf.tensor(actionBuffer, null, this.envActionSpace.dtype),
      this.tf.tensor(advantageBuffer),
      this.tf.tensor(returnBuffer).reshape([-1, 1]),
      this.tf.tensor(logprobabilityBuffer),
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

    this.tf.dispose([
      observationBufferT,
      actionBufferT,
      advantageBufferT,
      returnBufferT,
      logprobabilityBufferT,
    ]);
  }

  async learn(trainTimes = 64) {
    let iteration = 0;
    while (iteration < trainTimes) {
      this.eventEmitter.emitRolloutStart({ iteration });
      await this.collectRollouts();
      this.eventEmitter.emitRolloutEnd({ iteration });
      iteration += 1;
      this.eventEmitter.emitTrainingStart({ iteration });
      this.train();
      this.eventEmitter.emitTrainingEnd({ iteration });
    }
  }

  async exportModels() {
    const data: ModelsData = {
      actor: { modelTopology: {}, weightSpecs: [], weightData: '' },
      critic: { modelTopology: {}, weightSpecs: [], weightData: '' },
    };
    data.actor = await exportModel(this.actor);
    data.critic = await exportModel(this.critic);
    return data;
  }

  async importModels({ actor, critic }: ModelsData) {
    this.actor = await importModel(actor, this.tf);
    this.critic = await importModel(critic, this.tf);
  }

  predictAction(ob: StepObservation) {
    const [action] = this.tf.tidy(() => {
      const lastObservationT = this.tf.tensor([ob]);
      const [predsT, actionT] = this.sampleAction(lastObservationT);
      return [
        actionT.arraySync(), // -> Discrete: [] or Box: [actionSpace.shape[0]]
      ];
    });
    return action;
  }
}
