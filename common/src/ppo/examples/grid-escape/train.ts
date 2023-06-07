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

  console.log('开始训练');
  await ppo.learn();

  console.log('保存数据...');
  const data = await ppo.exportModels();
  fs.writeFileSync(__dirname + '/data.json', JSON.stringify(data));

  env.dispose();
  console.log('完成');
}
