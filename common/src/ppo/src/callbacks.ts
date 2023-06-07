export class BaseCallback {
  nCalls: number;
  constructor() {
    this.nCalls = 0;
  }

  _onStep(alg: any) {
    return true;
  }
  onStep(alg: any) {
    this.nCalls += 1;
    return this._onStep(alg);
  }

  _onTrainingStart(alg: any) {}
  onTrainingStart(alg: any) {
    this._onTrainingStart(alg);
  }

  _onTrainingEnd(alg: any) {}
  onTrainingEnd(alg: any) {
    this._onTrainingEnd(alg);
  }

  _onRolloutStart(alg: any) {}
  onRolloutStart(alg: any) {
    this._onRolloutStart(alg);
  }

  _onRolloutEnd(alg: any) {}
  onRolloutEnd(alg: any) {
    this._onRolloutEnd(alg);
  }
}

export class FunctionalCallback extends BaseCallback {
  callback: any;
  constructor(callback: any) {
    super();
    this.callback = callback;
  }

  _onStep(alg: any) {
    if (this.callback) {
      return this.callback(alg);
    }
    return true;
  }
}

export class DictCallback extends BaseCallback {
  callback: any;
  constructor(callback: any) {
    super();
    this.callback = callback;
  }

  _onStep(alg: any) {
    if (this.callback && this.callback.onStep) {
      return this.callback.onStep(alg);
    }
    return true;
  }

  _onTrainingStart(alg: any) {
    if (this.callback && this.callback.onTrainingStart) {
      this.callback.onTrainingStart(alg);
    }
  }

  _onTrainingEnd(alg: any) {
    if (this.callback && this.callback.onTrainingEnd) {
      this.callback.onTrainingEnd(alg);
    }
  }

  _onRolloutStart(alg: any) {
    if (this.callback && this.callback.onRolloutStart) {
      this.callback.onRolloutStart(alg);
    }
  }

  _onRolloutEnd(alg: any) {
    if (this.callback && this.callback.onRolloutEnd) {
      this.callback.onRolloutEnd(alg);
    }
  }
}
