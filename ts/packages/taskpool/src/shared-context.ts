import { Mutex } from "@openintj/concurrency";

/**
 * SharedContext —— 多 Agent 共享的上下文键值存储，写入互斥。
 *
 * 用例：
 *  - Planner 把任务拆解后写入 plan key
 *  - 各子 Agent 读 plan，写各自的 partial result
 *  - Synthesizer 读全部 partial result 做最终合成
 */
export class SharedContext {
  private store = new Map<string, unknown>();
  private mutex = new Mutex();

  async set<T>(key: string, value: T): Promise<void> {
    await this.mutex.runExclusive(() => {
      this.store.set(key, value);
    });
  }

  /** 同步读不需要锁；引用语义。 */
  get<T = unknown>(key: string): T | undefined {
    return this.store.get(key) as T | undefined;
  }

  has(key: string): boolean {
    return this.store.has(key);
  }

  /** 原子更新（get + transform + set）。 */
  async update<T>(
    key: string,
    transform: (current: T | undefined) => T | undefined,
  ): Promise<void> {
    await this.mutex.runExclusive(() => {
      const cur = this.store.get(key) as T | undefined;
      const next = transform(cur);
      if (next === undefined) this.store.delete(key);
      else this.store.set(key, next);
    });
  }

  /** 一次性获取多个键（不上锁，读取时刻的快照）。 */
  snapshot(): Record<string, unknown> {
    return Object.fromEntries(this.store.entries());
  }

  async delete(key: string): Promise<boolean> {
    return this.mutex.runExclusive(() => this.store.delete(key));
  }

  async clear(): Promise<void> {
    await this.mutex.runExclusive(() => this.store.clear());
  }

  get size(): number {
    return this.store.size;
  }
}
