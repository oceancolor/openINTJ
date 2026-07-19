import { DEFAULT_REACT_CONFIG, HookBus, type LlmClient, ReactStateMachine } from "@openintj/core";
import { GoalParser, Planner } from "@openintj/plane-control";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { AgentInstancePool } from "../src/agent-instance-pool.js";
import { Channel } from "../src/channel.js";
import { planGraphToTaskGraph } from "../src/plan-graph-adapter.js";
import { resolveOrchestrationMode, resolveTaskPoolRecoveryPolicy } from "../src/synthesizer.js";
import { TaskGraphValidationError, TaskPool, topologicalTaskOrder } from "../src/task-pool.js";
import { MemoryTaskStore, type StoredTaskRun, type TaskStore } from "../src/task-store.js";

describe("TaskPool MVD", () => {
  it("PlanGraph → TaskGraph 适配确定性", () => {
    const parser = new GoalParser();
    const planner = new Planner();
    const goal = parser.parse("帮我规划迁移方案", "planning");
    const plan = planner.createPlan(goal);
    const graph = planGraphToTaskGraph(plan);
    expect(graph.goalInput).toBe("帮我规划迁移方案");
    expect(graph.nodes.length).toBeGreaterThanOrEqual(3);
    expect(graph.nodes[0]!.deps).toEqual([]);
  });

  it("钻石 DAG 按依赖顺序完成", async () => {
    const order: string[] = [];
    const graph = {
      planId: "p1",
      goalIntent: "plan",
      nodes: [
        { id: "a", deps: [], action: "think", description: "A" },
        { id: "b", deps: ["a"], action: "think", description: "B" },
        { id: "c", deps: ["a"], action: "think", description: "C" },
        { id: "d", deps: ["b", "c"], action: "respond", description: "D" },
      ],
    };
    const pool = new TaskPool({ maxConcurrency: 2 });
    const result = await pool.submitRun(graph, async (node) => {
      order.push(node.id);
      return node.id;
    });
    expect(result.status).toBe("completed");
    expect(order.indexOf("a")).toBeLessThan(order.indexOf("b"));
    expect(order.indexOf("a")).toBeLessThan(order.indexOf("c"));
    expect(order.indexOf("d")).toBe(order.length - 1);
  });

  it("validates missing dependencies and cycles", () => {
    expect(() =>
      topologicalTaskOrder({
        planId: "missing",
        goalIntent: "test",
        nodes: [{ id: "a", deps: ["nope"], action: "x", description: "x" }],
      }),
    ).toThrow(TaskGraphValidationError);
    expect(() =>
      topologicalTaskOrder({
        planId: "cycle",
        goalIntent: "test",
        nodes: [
          { id: "a", deps: ["b"], action: "x", description: "x" },
          { id: "b", deps: ["a"], action: "x", description: "x" },
        ],
      }),
    ).toThrow(/cycle/);
  });

  it("bounds peak concurrency and stores partial results", async () => {
    let active = 0;
    let peak = 0;
    const graph = {
      planId: "wide",
      goalIntent: "test",
      nodes: Array.from({ length: 6 }, (_, i) => ({
        id: `${i}`,
        deps: [],
        action: "x",
        description: "x",
      })),
    };
    const result = await new TaskPool({ maxConcurrency: 2 }).submitRun(graph, async (node, ctx) => {
      active++;
      peak = Math.max(peak, active);
      await new Promise((resolve) => setTimeout(resolve, 5));
      active--;
      expect(ctx.signal.aborted).toBe(false);
      return node.id;
    });
    expect(peak).toBe(2);
    expect(result.results.size).toBe(6);
  });

  it("retries with ready transition and bounded attempts", async () => {
    const hooks = new HookBus();
    const events: string[] = [];
    hooks.on("taskpool.task.retry", () => events.push("retry"));
    hooks.on("taskpool.task.ready", () => events.push("ready"));
    let calls = 0;
    const result = await new TaskPool({
      hooks,
      retry: { maxRetries: 2, initialBackoffMs: 1 },
    }).submitRun(
      {
        planId: "retry",
        goalIntent: "test",
        nodes: [{ id: "a", deps: [], action: "x", description: "x" }],
      },
      async () => {
        calls++;
        if (calls < 3) throw new Error("transient");
        return "ok";
      },
    );
    expect(result.status).toBe("completed");
    expect(result.attempts.get("a")).toBe(3);
    expect(events).toEqual(["ready", "retry", "ready", "retry", "ready"]);
  });

  it("distinguishes timeout and cancellation and cascades descendants", async () => {
    const timeout = await new TaskPool({ taskTimeoutMs: 5 }).submitRun(
      {
        planId: "timeout",
        goalIntent: "test",
        nodes: [
          { id: "a", deps: [], action: "x", description: "x" },
          { id: "b", deps: ["a"], action: "x", description: "x" },
        ],
      },
      async (_node, ctx) =>
        new Promise((_resolve, reject) =>
          ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason)),
        ),
    );
    expect(timeout.states.get("a")).toBe("timed_out");
    expect(timeout.states.get("b")).toBe("failed");

    const handle = new TaskPool().submit(
      {
        planId: "cancel",
        goalIntent: "test",
        nodes: [
          { id: "a", deps: [], action: "x", description: "x" },
          { id: "b", deps: ["a"], action: "x", description: "x" },
        ],
      },
      async (_node, ctx) =>
        new Promise((_resolve, reject) =>
          ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason)),
        ),
    );
    handle.cancel();
    const cancelled = await handle.result;
    expect(cancelled.status).toBe("cancelled");
    expect(cancelled.states.get("b")).toBe("cancelled");
  });

  it("propagates TaskPool cancellation into an in-flight ReAct LLM call", async () => {
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    let llmSignal: AbortSignal | undefined;
    const llm: LlmClient = {
      chat: async (_messages, opts) => {
        llmSignal = opts?.signal;
        markStarted();
        return await new Promise<string>((_resolve, reject) => {
          opts?.signal?.addEventListener("abort", () => reject(opts.signal?.reason), {
            once: true,
          });
        });
      },
      visionChat: async () => "unused",
      getStatus: () => ({
        provider: "test",
        model: "test",
        available: true,
        mode: "live",
        status: "connected",
        visionSupported: false,
      }),
    };
    const react = new ReactStateMachine({
      config: DEFAULT_REACT_CONFIG,
      hooks: new HookBus(),
      llm,
      toolRunner: async () => {
        throw new Error("unexpected tool call");
      },
    });
    const handle = new TaskPool().submit(
      {
        planId: "llm-cancel",
        goalIntent: "test",
        goalInput: "cancel the request",
        nodes: [{ id: "a", deps: [], action: "x", description: "x" }],
      },
      async (_node, ctx) =>
        react.runSingle(
          {
            messages: [{ role: "user", content: ctx.goalInput }],
            availableTools: [],
            taoIteration: 1,
            systemPrompt: "",
          },
          { signal: ctx.signal },
        ),
    );

    await started;
    handle.cancel("stop LLM");
    const result = await handle.result;

    expect(result.status).toBe("cancelled");
    expect(llmSignal?.aborted).toBe(true);
  });

  it("persists lifecycle snapshots and lists no completed run", async () => {
    const store = new MemoryTaskStore();
    const result = await new TaskPool({ store }).submitRun(
      {
        planId: "persist",
        goalIntent: "test",
        nodes: [{ id: "a", deps: [], action: "x", description: "x" }],
      },
      async () => "saved",
    );
    expect((await store.loadRun(result.runId))?.status).toBe("completed");
    expect(await store.listIncompleteRuns()).toEqual([]);
  });

  it("cancels interrupted snapshots by default without replaying workers", async () => {
    const store = new MemoryTaskStore();
    await store.saveRun({
      runId: "interrupted",
      planId: "safe-default",
      status: "running",
      graph: {
        planId: "safe-default",
        goalIntent: "plan",
        goalInput: "规划数据库迁移",
        nodes: [{ id: "a", deps: [], action: "write", description: "写迁移文件" }],
      },
      nodes: [{ taskId: "a", state: "running", attempt: 1, updatedAt: 2 }],
      createdAt: 1,
      updatedAt: 2,
    });
    const worker = async (): Promise<string> => {
      throw new Error("safe recovery must not replay side effects");
    };

    const summary = await new TaskPool({ store }).recoverIncomplete(worker);

    expect(summary).toEqual({
      policy: "cancel",
      found: 1,
      resumed: 0,
      completed: 0,
      cancelled: 1,
      failed: 0,
    });
    const saved = await store.loadRun("interrupted");
    expect(saved?.status).toBe("cancelled");
    expect(saved?.nodes[0]).toMatchObject({
      state: "cancelled",
      error: "interrupted by process restart",
    });
  });

  it("refuses to resume legacy snapshots that lack the original input", async () => {
    const store = new MemoryTaskStore();
    await store.saveRun({
      runId: "legacy-run",
      planId: "legacy-plan",
      status: "running",
      graph: {
        planId: "legacy-plan",
        goalIntent: "plan",
        nodes: [{ id: "a", deps: [], action: "x", description: "legacy" }],
      },
      nodes: [{ taskId: "a", state: "running", attempt: 1, updatedAt: 2 }],
      createdAt: 1,
      updatedAt: 2,
    });
    let workerCalls = 0;

    const summary = await new TaskPool({ store }).recoverIncomplete(async () => {
      workerCalls++;
      return "unexpected";
    }, "resume");

    expect(workerCalls).toBe(0);
    expect(summary).toMatchObject({ resumed: 0, cancelled: 1, failed: 0 });
    expect((await store.loadRun("legacy-run"))?.nodes[0]?.error).toContain(
      "without original goal input",
    );
  });

  it("explicitly resumes incomplete nodes with the persisted original input", async () => {
    const store = new MemoryTaskStore();
    await store.saveRun({
      runId: "resume-run",
      planId: "resume-plan",
      status: "running",
      graph: {
        planId: "resume-plan",
        goalIntent: "plan",
        goalInput: "规划 TypeScript 迁移",
        nodes: [
          { id: "a", deps: [], action: "analyze", description: "分析" },
          { id: "b", deps: ["a"], action: "respond", description: "总结" },
        ],
      },
      nodes: [
        { taskId: "a", state: "completed", attempt: 1, result: "A", updatedAt: 1 },
        { taskId: "b", state: "running", attempt: 1, updatedAt: 2 },
      ],
      createdAt: 1,
      updatedAt: 2,
    });
    const calls: string[] = [];

    const summary = await new TaskPool({ store }).recoverIncomplete(async (node, ctx) => {
      calls.push(node.id);
      expect(ctx.goalInput).toBe("规划 TypeScript 迁移");
      expect(ctx.shared.get("task:a:result")).toBe("A");
      return "B";
    }, "resume");

    expect(calls).toEqual(["b"]);
    expect(summary).toEqual({
      policy: "resume",
      found: 1,
      resumed: 1,
      completed: 1,
      cancelled: 0,
      failed: 0,
    });
    expect(await store.loadRun("resume-run")).toMatchObject({
      status: "completed",
      createdAt: 1,
    });
  });

  it("does not replay terminally failed nodes during explicit resume", async () => {
    const store = new MemoryTaskStore();
    await store.saveRun({
      runId: "resume-failed",
      planId: "resume-failed-plan",
      status: "running",
      graph: {
        planId: "resume-failed-plan",
        goalIntent: "plan",
        goalInput: "执行有依赖的计划",
        nodes: [
          { id: "a", deps: [], action: "write", description: "已失败步骤" },
          { id: "b", deps: ["a"], action: "respond", description: "下游步骤" },
        ],
      },
      nodes: [
        { taskId: "a", state: "failed", attempt: 1, error: "write failed", updatedAt: 1 },
        { taskId: "b", state: "running", attempt: 1, updatedAt: 2 },
      ],
      createdAt: 1,
      updatedAt: 2,
    });
    let workerCalls = 0;

    const summary = await new TaskPool({ store }).recoverIncomplete(async () => {
      workerCalls++;
      return "unexpected";
    }, "resume");

    expect(workerCalls).toBe(0);
    expect(summary.failed).toBe(1);
    expect((await store.loadRun("resume-failed"))?.nodes).toMatchObject([
      { taskId: "a", state: "failed", error: "write failed" },
      { taskId: "b", state: "failed" },
    ]);
  });

  it("serializes durable snapshots across parallel transitions", async () => {
    class DetectConcurrentStore implements TaskStore {
      readonly memory = new MemoryTaskStore();
      active = 0;
      peak = 0;

      async saveRun(run: StoredTaskRun): Promise<void> {
        this.active++;
        this.peak = Math.max(this.peak, this.active);
        await new Promise((resolve) => setTimeout(resolve, 2));
        await this.memory.saveRun(run);
        this.active--;
      }

      loadRun(runId: string): Promise<StoredTaskRun | undefined> {
        return this.memory.loadRun(runId);
      }

      listIncompleteRuns(): Promise<readonly StoredTaskRun[]> {
        return this.memory.listIncompleteRuns();
      }
    }

    const store = new DetectConcurrentStore();
    const result = await new TaskPool({ store, maxConcurrency: 3 }).submitRun(
      {
        planId: "serialized-store",
        goalIntent: "test",
        nodes: ["a", "b", "c"].map((id) => ({
          id,
          deps: [],
          action: "x",
          description: id,
        })),
      },
      async (node) => node.id,
    );
    expect(result.status).toBe("completed");
    expect(store.peak).toBe(1);
    expect((await store.loadRun(result.runId))?.status).toBe("completed");
  });

  it("gives eligible TaskPool explicit precedence over self-consistency", () => {
    expect(resolveOrchestrationMode(true, true)).toBe("taskpool");
    expect(resolveOrchestrationMode(false, true)).toBe("self-consistency");
    expect(resolveOrchestrationMode(false, false)).toBe("simple");
  });

  it("requires explicit opt-in before restart recovery replays workers", () => {
    expect(resolveTaskPoolRecoveryPolicy(undefined, {})).toBe("cancel");
    expect(
      resolveTaskPoolRecoveryPolicy(undefined, {
        OPENINTJ_TASK_POOL_RECOVERY: "resume",
      }),
    ).toBe("resume");
    expect(
      resolveTaskPoolRecoveryPolicy("cancel", {
        OPENINTJ_TASK_POOL_RECOVERY: "resume",
      }),
    ).toBe("cancel");
  });
});

describe("multi-agent opt-in primitives", () => {
  it("bounds role acquisition and hands leases to waiters", async () => {
    const pool = new AgentInstancePool(
      async (role: "researcher") => ({ id: crypto.randomUUID(), role }),
      1,
    );
    const first = await pool.acquire("researcher");
    let acquired = false;
    const secondPromise = pool.acquire("researcher").then((lease) => {
      acquired = true;
      return lease;
    });
    await Promise.resolve();
    expect(acquired).toBe(false);
    expect(pool.stats("researcher").created).toBe(1);
    first.release();
    const second = await secondPromise;
    expect(second.agent.id).toBe(first.agent.id);
    second.release();
  });

  it("validates messages before constrained reduction", () => {
    const channel = new Channel(
      z.object({ role: z.enum(["researcher", "reviewer"]), text: z.string().min(1) }),
      { count: 0 },
      (state) => ({ count: state.count + 1 }),
    );
    channel.send({ role: "reviewer", text: "ok" });
    expect(channel.state()).toEqual({ count: 1 });
    expect(() => channel.send({ role: "admin", text: "" })).toThrow();
    expect(channel.state()).toEqual({ count: 1 });
  });
});
