// Implements a buffer for storing state information in a reinforcement
// learning algorithm

import assert from 'assert';
import { StepObservation, TF } from './defines';

interface StateBufferParams {
  tf: TF;
  gamma?: number;
  lam?: number;
}

export class StateBuffer {
  tf: TF;
  gamma = 0.99;
  lam = 0.95;
  observationBuffer: StepObservation[] = [];
  actionBuffer: any[] = [];
  advantageBuffer: any[] = [];
  rewardBuffer: any[] = [];
  returnBuffer: any[] = [];
  valueBuffer: any[] = [];
  logprobabilityBuffer: any[] = [];
  trajectoryStartIndex = 0;
  pointer = 0;

  constructor({ tf, gamma = 0.99, lam = 0.95 }: StateBufferParams) {
    this.tf = tf;
    this.gamma = gamma;
    this.lam = lam;
    this.reset();
  }

  add(
    observation: StepObservation,
    action: any,
    reward: any,
    value: any,
    logprobability: any
  ) {
    this.observationBuffer.push(observation.slice(0));
    this.actionBuffer.push(action);
    this.rewardBuffer.push(reward);
    this.valueBuffer.push(value);
    this.logprobabilityBuffer.push(logprobability);
    this.pointer += 1;
  }

  discountedCumulativeSums(arr: any[], coeff: number) {
    let res: number[] = [];
    let s = 0;
    arr.reverse().forEach((v: number) => {
      s = v + s * coeff;
      res.push(s);
    });
    return res.reverse();
  }

  finishTrajectory(lastValue: number) {
    const rewards = this.rewardBuffer
      .slice(this.trajectoryStartIndex, this.pointer)
      .concat(lastValue * this.gamma);
    const values = this.valueBuffer
      .slice(this.trajectoryStartIndex, this.pointer)
      .concat(lastValue);
    const deltas = rewards
      .slice(0, -1)
      .map((reward, ri) => reward - (values[ri] - this.gamma * values[ri + 1]));
    this.advantageBuffer = this.advantageBuffer.concat(
      this.discountedCumulativeSums(deltas, this.gamma * this.lam)
    );
    this.returnBuffer = this.returnBuffer.concat(
      this.discountedCumulativeSums(rewards, this.gamma).slice(0, -1)
    );
    this.trajectoryStartIndex = this.pointer;
  }

  get() {
    const [advantageMean, advantageStd] = this.tf.tidy(() => [
      this.tf.mean(this.advantageBuffer).arraySync(),
      this.tf.moments(this.advantageBuffer).variance.sqrt().arraySync(),
    ]);
    assert(typeof advantageMean === 'number');
    assert(typeof advantageStd === 'number');
    this.advantageBuffer = this.advantageBuffer.map(
      (advantage) => (advantage - advantageMean) / advantageStd
    );

    return [
      this.observationBuffer,
      this.actionBuffer,
      this.advantageBuffer,
      this.returnBuffer,
      this.logprobabilityBuffer,
    ];
  }

  reset() {
    this.observationBuffer = [];
    this.actionBuffer = [];
    this.advantageBuffer = [];
    this.rewardBuffer = [];
    this.returnBuffer = [];
    this.valueBuffer = [];
    this.logprobabilityBuffer = [];
    this.trajectoryStartIndex = 0;
    this.pointer = 0;
  }
}
