import { ConditionVariable, Mutex } from "@openintj/concurrency";

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
 */
export class TaskQueue {
  private tasks = new Map<string, InternalTask>();
  private order = 0;
  private mutex = new Mutex();
  private readyCv = new ConditionVariable();
  private closed = false;

  async submit<T>(node: TaskNode<T>): Promise<void> {
    await this.mutex.runExclusive(() => {
      if (this.tasks.has(node.id)) {
        throw new Error(`TaskQueue: duplicate task id ${node.id}`);
      }
      const ready = node.deps.every((d) => {
        const dep = this.tasks.get(d);
        return dep && dep.state === "completed";
      });
      this.tasks.set(node.id, {
        node: node as TaskNode<unknown>,
        state: ready ? "ready" : "waiting",
        enqueueOrder: this.order++,
      });
      if (ready) this.readyCv.notify();
    });
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
      if (next) return next;
      if (this.closed && this.allDone()) return undefined;
      await this.readyCv.wait();
    }
  }

  async complete(taskId: string, result: unknown): Promise<void> {
    await this.mutex.runExclusive(() => {
      const t = this.tasks.get(taskId);
      if (!t) return;
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
    });
  }

  async fail(taskId: string, error: unknown): Promise<void> {
    await this.mutex.runExclusive(() => {
      const t = this.tasks.get(taskId);
      if (!t) return;
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
