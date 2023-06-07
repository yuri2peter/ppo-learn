import Emitter from 'event-emitter';
type Cb<T> = (data: T) => void;

export class PPOEvent {
  ev = Emitter();

  emitStepStart(data: EventStepStartData) {
    this.ev.emit('stepStart', data);
  }
  onStepStart(cb: Cb<EventStepStartData>) {
    this.ev.on('stepStart', cb);
  }
  offStepStart(cb: Cb<EventStepStartData>) {
    this.ev.off('stepStart', cb);
  }

  emitStepEnd(data: EventStepEndData) {
    this.ev.emit('stepEnd', data);
  }
  onStepEnd(cb: Cb<EventStepEndData>) {
    this.ev.on('stepEnd', cb);
  }
  offStepEnd(cb: Cb<EventStepEndData>) {
    this.ev.off('stepEnd', cb);
  }

  emitTrainingStart(data: EventTrainingData) {
    this.ev.emit('trainingStart', data);
  }
  onTrainingStart(cb: Cb<EventTrainingData>) {
    this.ev.on('trainingStart', cb);
  }
  offTrainingStart(cb: Cb<EventTrainingData>) {
    this.ev.off('trainingStart', cb);
  }

  emitTrainingEnd(data: EventTrainingData) {
    this.ev.emit('trainingEnd');
  }
  onTrainingEnd(cb: Cb<EventTrainingData>) {
    this.ev.on('trainingEnd', cb);
  }
  offTrainingEnd(cb: Cb<EventTrainingData>) {
    this.ev.off('trainingEnd', cb);
  }

  emitRolloutStart(data: EventRolloutData) {
    this.ev.emit('rolloutStart', data);
  }
  onRolloutStart(cb: Cb<EventRolloutData>) {
    this.ev.on('rolloutStart', cb);
  }
  offRolloutStart(cb: Cb<EventRolloutData>) {
    this.ev.off('rolloutStart', cb);
  }

  emitRolloutEnd(data: EventRolloutData) {
    this.ev.emit('rolloutEnd', data);
  }
  onRolloutEnd(cb: Cb<EventRolloutData>) {
    this.ev.on('rolloutEnd', cb);
  }
  offRolloutEnd(cb: Cb<EventRolloutData>) {
    this.ev.off('rolloutEnd', cb);
  }
}

export type EventType =
  | 'stepStart'
  | 'stepEnd'
  | 'trainingStart'
  | 'trainingEnd'
  | 'rolloutStart'
  | 'rolloutEnd';

export interface EventStepStartData {
  step: number;
  nSteps: number;
}

export interface EventStepEndData {
  step: number;
  nSteps: number;
  reward: number;
  done: boolean;
  value: number;
  episodes: number;
  sumReturn: number;
}

export interface EventRolloutData {
  iteration: number;
}

export interface EventTrainingData {
  iteration: number;
}
