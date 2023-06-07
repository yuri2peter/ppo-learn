import {
  BoxActionSpace,
  DiscreteActionSpace,
  ObservationSpace,
  StepAction,
  StepReturns,
  StepObservation,
} from './defines';

export abstract class Env {
  abstract getActionSpace(): BoxActionSpace | DiscreteActionSpace;
  abstract getObservationSpace(): ObservationSpace;
  abstract step(action: StepAction): Promise<StepReturns> | StepReturns;
  abstract reset(): Promise<StepObservation> | StepObservation;
}
