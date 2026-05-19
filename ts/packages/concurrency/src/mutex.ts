/**
 * Mutex —— async 互斥锁。FIFO 公平。
 *
 * 用法：
 *   const release = await mutex.acquire();
 *   try { ... } finally { release(); }
 *
 * 或：
 *   await mutex.runExclusive(async () => { ... });
 */
export class Mutex {
  private locked = false;
  private waiters: Array<(release: () => void) => void> = [];

  async acquire(): Promise<() => void> {
    if (!this.locked) {
      this.locked = true;
      return this.makeRelease();
    }
    return new Promise<() => void>((resolve) => {
      this.waiters.push((release) => resolve(release));
    });
  }

  async runExclusive<T>(fn: () => Promise<T> | T): Promise<T> {
    const release = await this.acquire();
    try {
      return await fn();
    } finally {
      release();
    }
  }

  get isLocked(): boolean {
    return this.locked;
  }

  get waitersCount(): number {
    return this.waiters.length;
  }

  private makeRelease(): () => void {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const next = this.waiters.shift();
      if (next) {
        next(this.makeRelease());
      } else {
        this.locked = false;
      }
    };
  }
}

/** Semaphore —— 计数信号量（最多 n 个并发持有者）。 */
export class Semaphore {
  private permits: number;
  private waiters: Array<() => void> = [];

  constructor(permits: number) {
    if (permits < 1) throw new Error("Semaphore permits must be >= 1");
    this.permits = permits;
  }

  async acquire(): Promise<() => void> {
    if (this.permits > 0) {
      this.permits--;
      return this.makeRelease();
    }
    return new Promise<() => void>((resolve) => {
      this.waiters.push(() => resolve(this.makeRelease()));
    });
  }

  async runExclusive<T>(fn: () => Promise<T> | T): Promise<T> {
    const release = await this.acquire();
    try {
      return await fn();
    } finally {
      release();
    }
  }

  get available(): number {
    return this.permits;
  }

  get waitersCount(): number {
    return this.waiters.length;
  }

  private makeRelease(): () => void {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const next = this.waiters.shift();
      if (next) next();
      else this.permits++;
    };
  }
}
