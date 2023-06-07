import ora, { Ora } from 'ora';
import { sleep } from '../../../utils/miscs';
import { Env } from '../../src/Env';
import {
  BoxActionSpace,
  DiscreteActionSpace,
  ObservationSpace,
  StepAction,
  StepObservation,
  StepReturns,
} from '../../src/defines';
import { Wma } from '../../src/Wma';
import { formatParams } from '../../src/utils';

export class GameEnv extends Env {
  numWin = 0;
  numLoss = 0;
  wma = new Wma();
  agent: number[];
  goal: number[];
  i: number;
  spinner: Ora;
  constructor() {
    super();
    this.agent = [0, 0];
    this.goal = [0, 0];
    this.i = 0;
    this.reset();
    this.spinner = ora('').start();
  }
  getActionSpace(): BoxActionSpace | DiscreteActionSpace {
    return {
      class: 'Box',
      shape: [2],
      low: [-1, -1],
      high: [1, 1],
      dtype: 'float32',
    };
  }
  getObservationSpace(): ObservationSpace {
    return {
      class: 'Box',
      shape: [4],
      dtype: 'float32',
    };
  }
  async step(action: StepAction) {
    if (!Array.isArray(action)) {
      throw new Error(`Step action must be an array`);
    }
    const [x, y] = this.agent;
    this.agent = [x + action[0] * 0.05, y + action[1] * 0.05];
    this.i += 1;
    const reward = -Math.sqrt(
      Math.pow(this.agent[0] - this.goal[0], 2) +
        Math.pow(this.agent[1] - this.goal[1], 2)
    );
    const win = reward > -0.01;
    if (win) {
      this.numWin++;
    }
    const loss = this.i > 30;
    if (loss && !win) {
      this.numLoss++;
    }
    const done = win || loss;
    await sleep(1);
    this.wma.update(reward);
    this.spinner.text =
      `[ENV] ` +
      formatParams({
        wma: this.wma.getValue(),
        reward: reward,
        win: this.numWin,
        loss: this.numLoss,
      });
    return [
      [this.agent[0], this.agent[1], this.goal[0], this.goal[1]],
      reward,
      done,
    ] as StepReturns;
  }
  reset(): StepObservation {
    this.agent = [Math.random(), Math.random()];
    this.goal = [Math.random(), Math.random()];
    this.i = 0;
    return [this.agent[0], this.agent[1], this.goal[0], this.goal[1]];
  }

  dispose() {
    this.spinner.succeed();
  }
}
