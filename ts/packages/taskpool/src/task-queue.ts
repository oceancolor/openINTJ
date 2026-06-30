import { ConditionVariable, Mutex } from "@openintj/concurrency";
import type { HookBus } from "@openintj/core";

/**
 * Task —— DAG 任务节点。
 */
export interface TaskNode<T = unknown> {
  id: string;
  /** 优先级（越大越高），同优先级按入队顺序。 */
  priority: number;
  /** 依赖任务的 id，全部完成后此任务才能进入 ready。 */
  deps: readonly string[];
  /** 任务负载（用户自定义）。 */
  payload: T;
  /** 完成回调（被 worker 调用）。 */
  run: (payload: T) => Promise<unknown> | unknown;
}

interface InternalTask {
  node: TaskNode<unknown>;
  state: "waiting" | "ready" | "running" | "completed" | "failed";
  enqueueOrder: number;
  result?: unknown;
  error?: unknown;
}

/**
 * TaskQueue —— 带依赖图（DAG）的优先级任务队列。
 *
 * - submit(task)：注册任务（依赖未满足时进入 waiting）
 * - dequeue()：取下一个 ready 优先级最高的任务（阻塞等待）
 * - complete(taskId, result)：标记完成，触发依赖项检查
 *
 * 此实现是单进程内 in-memory（不跨进程持久化）。
 *
 * 可观测性：传入 `opts.hooks` 后，submit/dequeue/complete/fail 会发出
 * `task.enqueue` / `task.beforeRun` / `task.afterRun` 事件（在临界区外 emit，避免再入死锁）。
 */
export interface TaskQueueOpts {
  /** 接 HookBus 后发出 task.* 可观测事件。 */
  hooks?: HookBus;
  /** 队列名（写进事件）。默认 "task-queue"。 */
  name?: string;
}

export class TaskQueue {
  private tasks = new Map<string, InternalTask>();
  private order = 0;
  private mutex = new Mutex();
  private readyCv = new ConditionVariable();
  private closed = false;
  private readonly hooks: HookBus | undefined;
  private readonly name: string;
  /** taskId → 取出执行的时刻，用于算 afterRun 耗时。 */
  private readonly runStart = new Map<string, number>();

  constructor(opts: TaskQueueOpts = {}) {
    this.hooks = opts.hooks;
    this.name = opts.name ?? "task-queue";
  }

  async submit<T>(node: TaskNode<T>): Promise<void> {
    const ready = await this.mutex.runExclusive(() => {
      if (this.tasks.has(node.id)) {
        throw new Error(`TaskQueue: duplicate task id ${node.id}`);
      }
      const isReady = node.deps.every((d) => {
        const dep = this.tasks.get(d);
        return dep && dep.state === "completed";
      });
      this.tasks.set(node.id, {
        node: node as TaskNode<unknown>,
        state: isReady ? "ready" : "waiting",
        enqueueOrder: this.order++,
      });
      if (isReady) this.readyCv.notify();
      return isReady;
    });
    if (this.hooks) {
      await this.hooks.emit("task.enqueue", {
        queue: this.name,
        taskId: node.id,
        priority: node.priority,
        depCount: node.deps.length,
        ready,
      });
    }
  }

  async dequeue(): Promise<InternalTask | undefined> {
    while (true) {
      const next = await this.mutex.runExclusive(() => {
        const candidates: InternalTask[] = [];
        for (const t of this.tasks.values()) {
          if (t.state === "ready") candidates.push(t);
        }
        if (candidates.length === 0) return undefined;
        candidates.sort(
          (a, b) => b.node.priority - a.node.priority || a.enqueueOrder - b.enqueueOrder,
        );
        const picked = candidates[0]!;
        picked.state = "running";
        return picked;
      });
      if (next) {
        this.runStart.set(next.node.id, Date.now());
        if (this.hooks) {
          await this.hooks.emit("task.beforeRun", {
            queue: this.name,
            taskId: next.node.id,
            priority: next.node.priority,
          });
        }
        return next;
      }
      if (this.closed && this.allDone()) return undefined;
      await this.readyCv.wait();
    }
  }

  async complete(taskId: string, result: unknown): Promise<void> {
    const existed = await this.mutex.runExclusive(() => {
      const t = this.tasks.get(taskId);
      if (!t) return false;
      t.state = "completed";
      t.result = result;
      // 解锁依赖此任务的下游
      for (const other of this.tasks.values()) {
        if (other.state !== "waiting") continue;
        if (other.node.deps.includes(taskId)) {
          const ready = other.node.deps.every((d) => {
            const dep = this.tasks.get(d);
            return dep && dep.state === "completed";
          });
          if (ready) other.state = "ready";
        }
      }
      this.readyCv.notifyAll();
      return true;
    });
    await this.emitAfterRun(taskId, existed, true);
  }

  async fail(taskId: string, error: unknown): Promise<void> {
    const existed = await this.mutex.runExclusive(() => {
      const t = this.tasks.get(taskId);
      if (!t) return false;
      t.state = "failed";
      t.error = error;
      // 失败级联：所有依赖此任务的也标 failed
      const cascade = (id: string): void => {
        for (const other of this.tasks.values()) {
          if (other.state === "completed" || other.state === "failed") continue;
          if (other.node.deps.includes(id)) {
            other.state = "failed";
            other.error = new Error(`upstream failed: ${id}`);
            cascade(other.node.id);
          }
        }
      };
      cascade(taskId);
      this.readyCv.notifyAll();
      return true;
    });
    await this.emitAfterRun(taskId, existed, false);
  }

  /** 在临界区外发出 task.afterRun（带耗时）。 */
  private async emitAfterRun(taskId: string, existed: boolean, success: boolean): Promise<void> {
    const startedAt = this.runStart.get(taskId);
    this.runStart.delete(taskId);
    if (!existed || !this.hooks) return;
    await this.hooks.emit("task.afterRun", {
      queue: this.name,
      taskId,
      success,
      durationMs: startedAt !== undefined ? Date.now() - startedAt : 0,
    });
  }

  /** 直接拿任务结果（必须是 completed）。 */
  result(taskId: string): { result?: unknown; error?: unknown; state: string } {
    const t = this.tasks.get(taskId);
    if (!t) return { state: "missing" };
    return { result: t.result, error: t.error, state: t.state };
  }

  close(): void {
    this.closed = true;
    this.readyCv.notifyAll();
  }

  private allDone(): boolean {
    for (const t of this.tasks.values()) {
      if (t.state !== "completed" && t.state !== "failed") return false;
    }
    return true;
  }

  get pending(): number {
    let n = 0;
    for (const t of this.tasks.values()) {
      if (t.state === "waiting" || t.state === "ready") n++;
    }
    return n;
  }
}
