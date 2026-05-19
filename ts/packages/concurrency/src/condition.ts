/**
 * ConditionVariable —— 条件变量。
 * 经典生产者-消费者模式中的等待点。
 *
 *   await cond.wait();    // 阻塞直到 notify
 *   cond.notify();        // 唤醒一个 waiter
 *   cond.notifyAll();     // 唤醒全部
 */
export class ConditionVariable {
  private waiters: Array<() => void> = [];

  wait(): Promise<void> {
    return new Promise<void>((resolve) => {
      this.waiters.push(resolve);
    });
  }

  notify(): boolean {
    const w = this.waiters.shift();
    if (w) {
      w();
      return true;
    }
    return false;
  }

  notifyAll(): number {
    const n = this.waiters.length;
    const all = this.waiters;
    this.waiters = [];
    for (const w of all) w();
    return n;
  }

  get pendingWaiters(): number {
    return this.waiters.length;
  }
}
