import tf from '@tensorflow/tfjs';
import { Env } from './Env';

type ClassBox = 'Box';
type ClassDiscrete = 'Discrete';
type DtypeFloat = 'float32';
type DtypeInt = 'int32';

//  the class of the action space (Box or Discrete)
export type SpaceClass = ClassBox | ClassDiscrete;
export type Dtype = DtypeFloat | DtypeInt;

export interface BoxActionSpace {
  class: ClassBox; // SpaceClass
  shape: [number]; // the shape of the action space. e.g.[2]
  low: number[]; // the lower bound of the action space. e.g.[-1, -1]
  high: number[]; // the upper bound of the action space. e.g.[1, 1]
  dtype: DtypeFloat; // the data type of the action space
}

export interface DiscreteActionSpace {
  class: ClassDiscrete; // SpaceClass
  n: number; // the number of actions in the action space
  dtype: DtypeInt; // the data type of the action space
}

export interface ObservationSpace {
  class: SpaceClass;
  shape: [number];
  dtype: Dtype;
}

// step returns
export type StepAction = number[] | number; // agent's action
export type StepObservation = number[]; // the current state of the environment
export type StepReward = number; // the reward for the current step
export type StepDone = boolean; // a boolean indicating if the episode is over
export type StepReturns = [StepObservation, StepReward, StepDone];

export type TF = typeof tf;
export type ActivationIdentifier =
  | 'elu'
  | 'hardSigmoid'
  | 'linear'
  | 'relu'
  | 'relu6'
  | 'selu'
  | 'sigmoid'
  | 'softmax'
  | 'softplus'
  | 'softsign'
  | 'tanh'
  | 'swish'
  | 'mish';

export interface PPOParams {
  config?: Partial<TrainingConfig>;
  tf?: TF;
  env: Env;
}
export interface TrainingConfig {
  nSteps: number;
  nEpochs: number;
  policyLearningRate: number;
  valueLearningRate: number;
  clipRatio: number;
  targetKL: number;
  useSDE: false;
  netArch: {
    pi: number[];
    vf: number[];
  };
  activation: string;
  verbose: number;
  gamma: number;
  lam: number;
}

export const trainingConfigDefault: TrainingConfig = {
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
  gamma: 0.99,
  lam: 0.95,
};

export interface ModelArtifacts {
  modelTopology: { [key: string]: any };
  weightSpecs: any[];
  weightData: string;
}

export interface ModelsData {
  actor: ModelArtifacts;
  critic: ModelArtifacts;
}
