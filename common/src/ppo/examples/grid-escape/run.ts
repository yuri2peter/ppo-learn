import * as tfNode from '@tensorflow/tfjs-node';
import fs from 'fs';
import { PPO } from '../../src/PPO';
import { GameEnv } from './gameEnv';

main();
async function main() {
  const env = new GameEnv();
  const ppo = new PPO({ env, tf: tfNode });

  console.log('读取数据');
  const dataString = fs.readFileSync(__dirname + '/data.json', 'utf-8');
  await ppo.importModels(JSON.parse(dataString));

  console.log('开始运行');
  let ob = env.reset();
  for (let index = 0; index < 2000; index++) {
    const action = ppo.predictAction(ob);
    const [newOb, reward, done] = await env.step(action);

    ob = done ? env.reset() : newOb;
  }

  env.dispose();
  console.log('完成');
}
