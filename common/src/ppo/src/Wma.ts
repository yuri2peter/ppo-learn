// 平滑指数计算（Weighted Moving Average）

export class Wma {
  private alpha = 0;
  private value = 0;
  private inited = false;
  constructor(alpha = 0.01) {
    this.alpha = alpha;
  }

  update(num: number) {
    if (!this.inited) {
      this.inited = true;
      this.value = num;
    }
    this.value = this.value * (1 - this.alpha) + num * this.alpha;
    return this.value;
  }

  getValue() {
    return this.value.toFixed(3);
  }
}
