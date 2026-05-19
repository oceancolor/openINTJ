import { Semaphore } from "./mutex.js";

/**
 * AgentPool —— 并发 Agent 工作池（worker pool 模式）。
 *
 * - maxConcurrent：同时运行的 worker 上限
 * - submit(jobFn)：把任务投入队列，返回 Promise<T>
 * - shutdown：等待所有任务完成
 *
 * 用于多智能体编排：把 N 个独立任务并行跑，受 maxConcurrent 限制。
 */
export interface AgentPoolStats {
  active: number;
  pending: number;
  completed: number;
  failed: number;
}

export class AgentPool {
  private readonly sem: Semaphore;
  private active = 0;
  private completed = 0;
  private failed = 0;

  constructor(maxConcurrent: number) {
    this.sem = new Semaphore(maxConcurrent);
  }

  async submit<TIn, TOut>(job: (input: TIn) => Promise<TOut>, input: TIn): Promise<TOut> {
    const release = await this.sem.acquire();
    this.active++;
    try {
      const r = await job(input);
      this.completed++;
      return r;
    } catch (e) {
      this.failed++;
      throw e;
    } finally {
      this.active--;
      release();
    }
  }

  /** 一次性投入多个任务，返回与输入顺序对应的结果数组。 */
  async map<TItem, TResult>(
    items: readonly TItem[],
    fn: (item: TItem, index: number) => Promise<TResult>,
  ): Promise<TResult[]> {
    return Promise.all(
      items.map((item, i) => this.submit<TItem, TResult>(async (it: TItem) => fn(it, i), item)),
    );
  }

  /** Settled 版本：失败也不 reject 整体，返回 PromiseSettledResult。 */
  async mapSettled<TItem, TResult>(
    items: readonly TItem[],
    fn: (item: TItem, index: number) => Promise<TResult>,
  ): Promise<PromiseSettledResult<TResult>[]> {
    const results: PromiseSettledResult<TResult>[] = [];
    await Promise.all(
      items.map(async (item, i) => {
        try {
          const value = await this.submit<TItem, TResult>(async (it: TItem) => fn(it, i), item);
          results[i] = { status: "fulfilled", value };
        } catch (reason) {
          results[i] = { status: "rejected", reason };
        }
      }),
    );
    return results;
  }

  get stats(): AgentPoolStats {
    return {
      active: this.active,
      pending: this.sem.waitersCount,
      completed: this.completed,
      failed: this.failed,
    };
  }
}
