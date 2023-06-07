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
  gravity = 9.8;
  masscart = 1.0;
  masspole = 0.1;
  total_mass = 1.1;
  length = 0.5;
  polemass_length = 0.05;
  force_mag = 10.0;
  tau = 0.02;
  theta_threshold_radians = (12 * 2 * Math.PI) / 360;
  x_threshold = 2.4;
  tick_threshold = 200;

  theta_dot = 0;
  x = 0;
  x_dot = 0;
  theta = 0;

  numWin = 0;
  numLoss = 0;
  wma = new Wma();
  i = 0;
  spinner: Ora;
  agent: any;
  goal: any;

  constructor() {
    super();
    this.reset();
    this.spinner = ora('').start();
  }
  getActionSpace(): BoxActionSpace | DiscreteActionSpace {
    return {
      class: 'Discrete',
      n: 2,
      dtype: 'int32',
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
    this.i += 1;
    // 计算力的大小和方向
    const force = action ? this.force_mag : -this.force_mag;

    // 计算小车加速度和角加速度
    const costheta = Math.cos(this.theta);
    const sintheta = Math.sin(this.theta);
    const temp =
      (force +
        this.polemass_length * this.theta_dot * this.theta_dot * sintheta) /
      this.total_mass;
    const thetaacc =
      (this.gravity * sintheta - costheta * temp) /
      (this.length *
        (4.0 / 3.0 - (this.masspole * costheta * costheta) / this.total_mass));
    const xacc =
      temp - (this.polemass_length * thetaacc * costheta) / this.total_mass;

    // 更新小车位置、速度和角度、角速度
    this.x += this.tau * this.x_dot;
    this.x_dot += this.tau * xacc;
    this.theta += this.tau * this.theta_dot;
    this.theta_dot += this.tau * thetaacc;

    // 坚持一定时间后胜利
    const win = this.i > this.tick_threshold;

    // 判断是否结束游戏
    const done =
      this.i > this.tick_threshold ||
      this.x < -this.x_threshold ||
      this.x > this.x_threshold ||
      this.theta < -this.theta_threshold_radians ||
      this.theta > this.theta_threshold_radians;

    // 根据游戏是否结束，计算奖励
    const reward = done ? 0 : 1;

    // 胜利计数
    const loss = !win && done;
    if (win) {
      this.numWin++;
    }
    if (loss) {
      this.numLoss++;
    }

    // 延时
    await sleep(1000 / 60);

    if (done) {
      this.wma.update(this.i);
    }
    this.spinner.text =
      `[ENV] ` +
      formatParams({
        wma: this.wma.getValue(),
        theta: this.theta,
        win: this.numWin,
        loss: this.numLoss,
      });
    return [this.getCurrentObservation(), reward, done] as StepReturns;
  }

  reset(): StepObservation {
    this.i = 0;
    // 小车初始位置在[-0.05, 0.05]之间随机选取
    this.x = Math.random() * 0.1 - 0.05;
    // 小车初始速度为0
    this.x_dot = 0;
    // 杆子初始角度在[-0.05, 0.05]之间随机选取
    this.theta = Math.random() * 0.1 - 0.05;
    // 杆子初始角速度为0
    this.theta_dot = 0;
    return this.getCurrentObservation();
  }

  getCurrentObservation(): StepObservation {
    return [this.x, this.x_dot, this.theta, this.theta_dot];
  }

  dispose() {
    this.spinner.succeed();
  }
}
