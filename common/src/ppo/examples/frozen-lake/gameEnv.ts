import lodash from 'lodash';
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
  i = 0;
  spinner: Ora;

  currentPosition = 0;
  row = 0;
  col = 0;

  constructor() {
    super();
    this.reset();
    this.spinner = ora('').start();
  }
  getActionSpace(): BoxActionSpace | DiscreteActionSpace {
    return {
      class: 'Discrete',
      n: 4,
      dtype: 'int32',
    };
  }
  getObservationSpace(): ObservationSpace {
    return {
      class: 'Discrete',
      shape: [1],
      dtype: 'int32',
    };
  }
  async step(action: StepAction) {
    // 计数
    this.i += 1;

    // 延时
    await sleep(1);

    if (typeof action !== 'number') {
      throw new Error('action must be a number');
    }

    let reward = 0;
    let done = false;
    let newPos = this.move(this.row, this.col, action);
    this.row = newPos[0];
    this.col = newPos[1];
    const currentState = map[this.row][this.col];
    this.currentPosition = this.row * mapSize + this.col;
    if (currentState === 'H') {
      done = true;
      this.numLoss++;
      // reward = -1;
    } else if (currentState === 'G') {
      done = true;
      this.numWin++;
      reward = 1;
    }

    this.wma.update(Math.abs(7 - this.row) + Math.abs(7 - this.col));

    this.spinner.text =
      '[ENV] ' +
      formatParams({
        wma: this.wma.getValue(),
        row: this.row,
        col: this.col,
        win: this.numWin,
        loss: this.numLoss,
      });
    return [this.getCurrentObservation(), reward, done] as StepReturns;
  }

  reset(): StepObservation {
    return this.getCurrentObservation();
  }

  getCurrentObservation(): StepObservation {
    return [this.currentPosition];
  }

  dispose() {
    this.spinner.succeed();
  }

  private inMap(row: number, col: number, action: number) {
    if (row === 0 && action === Direction.Up) {
      return false;
    }
    if (row === mapSize - 1 && action === Direction.Down) {
      return false;
    }
    if (col === 0 && action === Direction.Left) {
      return false;
    }
    if (col === mapSize - 1 && action === Direction.Right) {
      return false;
    }
    return true;
  }

  private move(row: number, col: number, action: number): [number, number] {
    if (Math.random() > 1 / 3) {
      action = lodash.random(0, 3);
    }
    if (this.inMap(row, col, action)) {
      if (action === Direction.Up) {
        row -= 1;
      }
      if (action === Direction.Down) {
        row += 1;
      }
      if (action === Direction.Right) {
        col += 1;
      }
      if (action === Direction.Left) {
        col -= 1;
      }
    }
    return [row, col];
  }
}

const mapSize = 8;
const map = [
  ['S', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
  ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
  ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
  ['F', 'F', 'F', 'F', 'F', 'H', 'F', 'F'],
  ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F'],
  ['F', 'H', 'H', 'F', 'F', 'F', 'H', 'F'],
  ['F', 'H', 'F', 'F', 'H', 'F', 'H', 'F'],
  ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'G'],
];

enum Direction {
  Up,
  Right,
  Left,
  Down,
}
