import type { HookBus } from "@openintj/core";
import type { TaskGraph, TaskGraphNode } from "./plan-graph-adapter.js";
import { SharedContext } from "./shared-context.js";
import type { TaskPoolRecoveryPolicy } from "./synthesizer.js";
import type { StoredTaskNode, StoredTaskRun, TaskStore } from "./task-store.js";

export type TaskRunState =
  | "pending"
  | "ready"
  | "running"
  | "completed"
  | "failed"
  | "timed_out"
  | "cancelled";

export interface TaskPoolOpts {
  hooks?: HookBus;
  /** 并行 Tao worker 上限（默认 3）。 */
  maxConcurrency?: number;
  /** 单 task 超时 ms（默认 120_000）。 */
  taskTimeoutMs?: number;
  retry?: RetryPolicy;
  store?: TaskStore;
  name?: string;
}

export interface RetryPolicy {
  maxRetries?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
  shouldRetry?: (error: unknown, node: TaskGraphNode) => boolean;
}

export interface TaskWorkerContext {
  runId: string;
  /** Original input persisted with the graph; stable across restart recovery. */
  goalInput: string;
  /** Stable correlation id for this task attempt's Tao/tool trace. */
  traceId: string;
  shared: SharedContext;
  signal: AbortSignal;
  attempt: number;
}

export type TaskWorker<T = unknown> = (node: TaskGraphNode, ctx: TaskWorkerContext) => Promise<T>;

export interface TaskRunResult<T = unknown> {
  runId: string;
  planId: string;
  status: "completed" | "failed" | "cancelled";
  results: ReadonlyMap<string, T>;
  errors: ReadonlyMap<string, unknown>;
  states: ReadonlyMap<string, TaskRunState>;
  attempts: ReadonlyMap<string, number>;
}

export interface SubmitRunOptions {
  signal?: AbortSignal;
  traceId?: string;
  runId?: string;
}

export interface TaskPoolRecoverySummary {
  policy: TaskPoolRecoveryPolicy;
  found: number;
  resumed: number;
  completed: number;
  cancelled: number;
  failed: number;
}

export class TaskGraphValidationError extends Error {}
export class TaskTimeoutError extends Error {}
export class TaskCancelledError extends Error {}

const terminal = new Set<TaskRunState>(["completed", "failed", "timed_out", "cancelled"]);

/** Validate dependencies and return a stable topological order. */
export const topologicalTaskOrder = (graph: TaskGraph): readonly TaskGraphNode[] => {
  const byId = new Map<string, TaskGraphNode>();
  for (const node of graph.nodes) {
    if (byId.has(node.id)) throw new TaskGraphValidationError(`duplicate task id: ${node.id}`);
    byId.set(node.id, node);
  }
  for (const node of graph.nodes) {
    for (const dep of node.deps) {
      if (!byId.has(dep)) {
        throw new TaskGraphValidationError(`task ${node.id} has missing dependency: ${dep}`);
      }
    }
  }
  const remaining = new Map(graph.nodes.map((node) => [node.id, node.deps.length]));
  const ordered: TaskGraphNode[] = [];
  while (ordered.length < graph.nodes.length) {
    const next = graph.nodes.find(
      (node) => remaining.get(node.id) === 0 && !ordered.some((done) => done.id === node.id),
    );
    if (!next) throw new TaskGraphValidationError("task graph contains a cycle");
    ordered.push(next);
    for (const node of graph.nodes) {
      if (node.deps.includes(next.id)) remaining.set(node.id, remaining.get(node.id)! - 1);
    }
  }
  return ordered;
};

const errorText = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);

const wait = (ms: number, signal: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new TaskCancelledError("run cancelled"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new TaskCancelledError("run cancelled"));
      },
      { once: true },
    );
  });

export class TaskRun<T = unknown> {
  readonly result: Promise<TaskRunResult<T>>;

  constructor(
    readonly runId: string,
    readonly planId: string,
    readonly shared: SharedContext,
    private readonly controller: AbortController,
    private readonly stateMap: Map<string, TaskRunState>,
    private readonly attemptMap: Map<string, number>,
    execute: () => Promise<TaskRunResult<T>>,
  ) {
    this.result = execute();
  }

  cancel(reason = "cancelled by caller"): void {
    this.controller.abort(new TaskCancelledError(reason));
  }

  getState(taskId: string): TaskRunState | undefined {
    return this.stateMap.get(taskId);
  }

  snapshot(): {
    states: ReadonlyMap<string, TaskRunState>;
    attempts: ReadonlyMap<string, number>;
  } {
    return {
      states: new Map(this.stateMap),
      attempts: new Map(this.attemptMap),
    };
  }
}

/** Dependency-aware, bounded task executor. */
export class TaskPool {
  private readonly hooks: HookBus | undefined;
  private readonly maxConcurrency: number;
  private readonly taskTimeoutMs: number;
  private readonly retry: Required<Omit<RetryPolicy, "shouldRetry">> &
    Pick<RetryPolicy, "shouldRetry">;
  private readonly store: TaskStore | undefined;
  private readonly name: string;

  constructor(opts: TaskPoolOpts = {}) {
    this.hooks = opts.hooks;
    this.maxConcurrency = Math.max(1, opts.maxConcurrency ?? 3);
    this.taskTimeoutMs = opts.taskTimeoutMs ?? 120_000;
    this.retry = {
      maxRetries: Math.max(0, opts.retry?.maxRetries ?? 0),
      initialBackoffMs: Math.max(0, opts.retry?.initialBackoffMs ?? 100),
      maxBackoffMs: Math.max(0, opts.retry?.maxBackoffMs ?? 10_000),
      ...(opts.retry?.shouldRetry ? { shouldRetry: opts.retry.shouldRetry } : {}),
    };
    this.store = opts.store;
    this.name = opts.name ?? "task-pool";
  }

  submit<T>(
    graph: TaskGraph,
    worker: TaskWorker<T>,
    opts: SubmitRunOptions = {},
    seed?: StoredTaskRun,
  ): TaskRun<T> {
    const ordered = topologicalTaskOrder(graph);
    const runId = opts.runId ?? crypto.randomUUID();
    const seedNodes = new Map(seed?.nodes.map((node) => [node.taskId, node]) ?? []);
    const initialResults = new Map<string, T>();
    const initialErrors = new Map<string, unknown>();
    for (const node of seed?.nodes ?? []) {
      if (node.state === "completed") initialResults.set(node.taskId, node.result as T);
      else if (terminal.has(node.state) && node.error) {
        initialErrors.set(node.taskId, new Error(node.error));
      }
    }
    const shared = new SharedContext(
      [...initialResults].map(([taskId, result]) => [`task:${taskId}:result`, result] as const),
    );
    const states = new Map(
      ordered.map((node) => {
        const previous = seedNodes.get(node.id)?.state;
        return [
          node.id,
          previous && terminal.has(previous) ? previous : ("pending" as TaskRunState),
        ] as const;
      }),
    );
    const attempts = new Map(
      ordered.map((node) => [node.id, seedNodes.get(node.id)?.attempt ?? 0] as const),
    );
    const controller = new AbortController();
    if (opts.signal) {
      if (opts.signal.aborted) controller.abort(opts.signal.reason);
      else
        opts.signal.addEventListener("abort", () => controller.abort(opts.signal!.reason), {
          once: true,
        });
    }
    return new TaskRun(runId, graph.planId, shared, controller, states, attempts, () =>
      this.execute(
        graph,
        ordered,
        worker,
        shared,
        states,
        attempts,
        controller,
        runId,
        initialResults,
        initialErrors,
        opts.traceId,
        seed?.createdAt,
      ),
    );
  }

  async submitRun<T>(
    graph: TaskGraph,
    worker: TaskWorker<T>,
    opts: SubmitRunOptions = {},
  ): Promise<TaskRunResult<T>> {
    return this.submit(graph, worker, opts).result;
  }

  /** Re-submit a durable incomplete run; completed nodes are retained. */
  recover<T>(
    stored: StoredTaskRun,
    worker: TaskWorker<T>,
    opts: SubmitRunOptions = {},
  ): TaskRun<T> {
    return this.submit(stored.graph, worker, { ...opts, runId: stored.runId }, stored);
  }

  /**
   * Resolve snapshots left in `running` state by a previous process.
   *
   * `cancel` is the safe default because replaying an interrupted node cannot
   * guarantee exactly-once external side effects. `resume` must be explicit;
   * legacy snapshots without the original input are cancelled rather than
   * reconstructed from the lossy intent label.
   */
  async recoverIncomplete<T>(
    worker: TaskWorker<T>,
    policy: TaskPoolRecoveryPolicy = "cancel",
  ): Promise<TaskPoolRecoverySummary> {
    const summary: TaskPoolRecoverySummary = {
      policy,
      found: 0,
      resumed: 0,
      completed: 0,
      cancelled: 0,
      failed: 0,
    };
    if (!this.store) return summary;

    const incomplete = await this.store.listIncompleteRuns();
    summary.found = incomplete.length;
    for (const stored of incomplete) {
      if (policy === "cancel" || !stored.graph.goalInput) {
        const reason =
          policy === "cancel"
            ? "interrupted by process restart"
            : "cannot resume legacy snapshot without original goal input";
        const status = await this.cancelStoredRun(stored, reason);
        if (status === "completed") summary.completed++;
        else summary.cancelled++;
        continue;
      }

      summary.resumed++;
      try {
        const result = await this.recover(stored, worker, {
          traceId: `${stored.runId}:recovery`,
        }).result;
        if (result.status === "completed") summary.completed++;
        else if (result.status === "cancelled") summary.cancelled++;
        else summary.failed++;
      } catch (error) {
        summary.failed++;
        await this.cancelStoredRun(stored, `restart recovery failed: ${errorText(error)}`).catch(
          () => undefined,
        );
      }
    }
    return summary;
  }

  private async cancelStoredRun(
    stored: StoredTaskRun,
    reason: string,
  ): Promise<"completed" | "cancelled"> {
    if (!this.store) throw new Error("TaskPool recovery requires a TaskStore");
    const now = Date.now();
    const cancelledTaskIds: string[] = [];
    const nodes: StoredTaskNode[] = stored.nodes.map((node) => {
      if (terminal.has(node.state)) return node;
      cancelledTaskIds.push(node.taskId);
      return {
        ...node,
        state: "cancelled",
        error: reason,
        updatedAt: now,
      };
    });
    const status = nodes.every((node) => node.state === "completed") ? "completed" : "cancelled";
    await this.store.saveRun({
      ...stored,
      status,
      nodes,
      updatedAt: now,
    });
    for (const taskId of cancelledTaskIds) {
      await this.hooks?.emit(
        "taskpool.task.cancel",
        { pool: this.name, runId: stored.runId, taskId, reason },
        { traceId: `${stored.runId}:recovery` },
      );
    }
    await this.hooks?.emit(
      "taskpool.run.complete",
      {
        pool: this.name,
        runId: stored.runId,
        planId: stored.planId,
        status,
        completed: nodes.filter((node) => node.state === "completed").length,
        failed: nodes.filter((node) => node.state === "failed").length,
        cancelled: nodes.filter((node) => node.state === "cancelled").length,
        timedOut: nodes.filter((node) => node.state === "timed_out").length,
      },
      { traceId: `${stored.runId}:recovery` },
    );
    return status;
  }

  private async execute<T>(
    graph: TaskGraph,
    ordered: readonly TaskGraphNode[],
    worker: TaskWorker<T>,
    shared: SharedContext,
    states: Map<string, TaskRunState>,
    attempts: Map<string, number>,
    controller: AbortController,
    runId: string,
    initialResults: ReadonlyMap<string, T>,
    initialErrors: ReadonlyMap<string, unknown>,
    suppliedTraceId?: string,
    suppliedCreatedAt?: number,
  ): Promise<TaskRunResult<T>> {
    const id = runId;
    const traceId = suppliedTraceId ?? runId;
    const results = new Map(initialResults);
    const errors = new Map(initialErrors);
    const createdAt = suppliedCreatedAt ?? Date.now();
    let active = 0;
    let wake: (() => void) | undefined;
    let persistChain = Promise.resolve();
    const notify = (): void => {
      wake?.();
      wake = undefined;
    };
    const persist = (status: StoredTaskRun["status"] = "running"): Promise<void> => {
      if (!this.store) return Promise.resolve();
      const nodes: StoredTaskNode[] = ordered.map((node) => ({
        taskId: node.id,
        state: states.get(node.id)!,
        attempt: attempts.get(node.id)!,
        ...(results.has(node.id) ? { result: results.get(node.id) } : {}),
        ...(errors.has(node.id) ? { error: errorText(errors.get(node.id)) } : {}),
        updatedAt: Date.now(),
      }));
      const snapshot: StoredTaskRun = {
        runId: id,
        planId: graph.planId,
        status,
        graph,
        nodes,
        createdAt,
        updatedAt: Date.now(),
      };
      // Full-run snapshots must reach the store in transition order. Atomic
      // upserts alone do not prevent a slower stale write from winning.
      const save = persistChain.then(() => this.store!.saveRun(snapshot));
      persistChain = save.catch(() => undefined);
      return save;
    };
    const emit = async <K extends keyof import("@openintj/core").HookEventMap>(
      event: K,
      payload: import("@openintj/core").HookEventMap[K],
    ): Promise<void> => {
      await this.hooks?.emit(event, payload, { traceId });
    };
    const transition = async (node: TaskGraphNode, state: TaskRunState): Promise<void> => {
      states.set(node.id, state);
      await persist();
    };
    const cancelPending = async (reason: string): Promise<void> => {
      for (const node of ordered) {
        const state = states.get(node.id)!;
        if (!terminal.has(state) && state !== "running") {
          await transition(node, "cancelled");
          await emit("taskpool.task.cancel", {
            pool: this.name,
            runId: id,
            taskId: node.id,
            reason,
          });
        }
      }
    };
    const executeNode = async (node: TaskGraphNode): Promise<void> => {
      active++;
      try {
        while (!controller.signal.aborted) {
          const attempt = attempts.get(node.id)! + 1;
          const workerTraceId = `${traceId}:${node.id}:${attempt}`;
          attempts.set(node.id, attempt);
          await transition(node, "running");
          await emit("taskpool.task.start", {
            pool: this.name,
            runId: id,
            taskId: node.id,
            action: node.action,
            attempt,
            workerTraceId,
          });
          const taskController = new AbortController();
          const abortTask = (): void => taskController.abort(controller.signal.reason);
          controller.signal.addEventListener("abort", abortTask, { once: true });
          const timeoutMs = node.timeoutMs ?? this.taskTimeoutMs;
          let timedOut = false;
          const timer = setTimeout(() => {
            timedOut = true;
            taskController.abort(new TaskTimeoutError(`${node.id} timeout after ${timeoutMs}ms`));
          }, timeoutMs);
          try {
            const out = await Promise.race([
              worker(node, {
                runId: id,
                goalInput: graph.goalInput ?? graph.goalIntent,
                traceId: workerTraceId,
                shared,
                signal: taskController.signal,
                attempt,
              }),
              new Promise<never>((_, reject) => {
                taskController.signal.addEventListener(
                  "abort",
                  () =>
                    reject(
                      timedOut
                        ? new TaskTimeoutError(`${node.id} timeout after ${timeoutMs}ms`)
                        : new TaskCancelledError("run cancelled"),
                    ),
                  { once: true },
                );
              }),
            ]);
            results.set(node.id, out);
            await shared.set(`task:${node.id}:result`, out);
            await transition(node, "completed");
            await emit("taskpool.task.complete", {
              pool: this.name,
              runId: id,
              taskId: node.id,
              success: true,
              status: "completed",
              attempt,
            });
            return;
          } catch (error) {
            if (timedOut) {
              errors.set(node.id, error);
              await transition(node, "timed_out");
              await emit("taskpool.task.timeout", {
                pool: this.name,
                runId: id,
                taskId: node.id,
                attempt,
                timeoutMs,
              });
              await emit("taskpool.task.complete", {
                pool: this.name,
                runId: id,
                taskId: node.id,
                success: false,
                status: "timed_out",
                attempt,
                error: errorText(error),
              });
              return;
            }
            if (controller.signal.aborted) {
              errors.set(node.id, error);
              await transition(node, "cancelled");
              await emit("taskpool.task.cancel", {
                pool: this.name,
                runId: id,
                taskId: node.id,
                reason: errorText(controller.signal.reason ?? error),
              });
              return;
            }
            const retry =
              attempt <= this.retry.maxRetries && (this.retry.shouldRetry?.(error, node) ?? true);
            if (retry) {
              const delayMs = Math.min(
                this.retry.maxBackoffMs,
                this.retry.initialBackoffMs * 2 ** (attempt - 1),
              );
              await transition(node, "ready");
              await emit("taskpool.task.retry", {
                pool: this.name,
                runId: id,
                taskId: node.id,
                attempt,
                delayMs,
                error: errorText(error),
              });
              await emit("taskpool.task.ready", {
                pool: this.name,
                runId: id,
                taskId: node.id,
                attempt: attempt + 1,
              });
              await wait(delayMs, controller.signal);
              continue;
            }
            errors.set(node.id, error);
            await transition(node, "failed");
            await emit("taskpool.task.complete", {
              pool: this.name,
              runId: id,
              taskId: node.id,
              success: false,
              status: "failed",
              attempt,
              error: errorText(error),
            });
            return;
          } finally {
            clearTimeout(timer);
            controller.signal.removeEventListener("abort", abortTask);
          }
        }
      } finally {
        active--;
        notify();
      }
    };

    await emit("taskpool.run.submit", {
      pool: this.name,
      runId: id,
      planId: graph.planId,
      taskCount: ordered.length,
    });
    await persist();
    while ([...states.values()].some((state) => !terminal.has(state))) {
      if (controller.signal.aborted) await cancelPending("run cancelled");
      // Dependency failures deterministically cancel descendants.
      for (const node of ordered) {
        if (states.get(node.id) !== "pending") continue;
        const badDep = node.deps.find(
          (dep) => terminal.has(states.get(dep)!) && states.get(dep) !== "completed",
        );
        if (badDep) {
          const propagatedState = states.get(badDep) === "cancelled" ? "cancelled" : "failed";
          const reason = `dependency ${badDep} ended as ${states.get(badDep)}`;
          errors.set(node.id, new Error(reason));
          await transition(node, propagatedState);
          if (propagatedState === "cancelled") {
            await emit("taskpool.task.cancel", {
              pool: this.name,
              runId: id,
              taskId: node.id,
              reason,
            });
          } else {
            await emit("taskpool.task.complete", {
              pool: this.name,
              runId: id,
              taskId: node.id,
              success: false,
              status: "failed",
              attempt: attempts.get(node.id)!,
              error: reason,
            });
          }
        }
      }
      const ready = ordered.filter(
        (node) =>
          states.get(node.id) === "pending" &&
          node.deps.every((dep) => states.get(dep) === "completed"),
      );
      for (const node of ready) {
        await transition(node, "ready");
        await emit("taskpool.task.ready", {
          pool: this.name,
          runId: id,
          taskId: node.id,
          attempt: attempts.get(node.id)! + 1,
        });
      }
      for (const node of ordered) {
        if (active >= this.maxConcurrency) break;
        if (states.get(node.id) === "ready") void executeNode(node);
      }
      if ([...states.values()].some((state) => !terminal.has(state))) {
        await new Promise<void>((resolve) => {
          wake = resolve;
        });
      }
    }
    const status = controller.signal.aborted
      ? "cancelled"
      : [...states.values()].some((state) => state === "failed" || state === "timed_out")
        ? "failed"
        : "completed";
    await persist(status);
    await emit("taskpool.run.complete", {
      pool: this.name,
      runId: id,
      planId: graph.planId,
      status,
      completed: [...states.values()].filter((state) => state === "completed").length,
      failed: [...states.values()].filter((state) => state === "failed").length,
      cancelled: [...states.values()].filter((state) => state === "cancelled").length,
      timedOut: [...states.values()].filter((state) => state === "timed_out").length,
    });
    const orderedResults = new Map<string, T>();
    for (const node of ordered) {
      if (results.has(node.id)) orderedResults.set(node.id, results.get(node.id)!);
    }
    return {
      runId: id,
      planId: graph.planId,
      status,
      results: orderedResults,
      errors,
      states: new Map(states),
      attempts: new Map(attempts),
    };
  }
}
