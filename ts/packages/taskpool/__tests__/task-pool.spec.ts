import { HookBus } from "@openintj/core";
import { GoalParser, Planner } from "@openintj/plane-control";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { AgentInstancePool } from "../src/agent-instance-pool.js";
import { Channel } from "../src/channel.js";
import { planGraphToTaskGraph } from "../src/plan-graph-adapter.js";
import { resolveOrchestrationMode } from "../src/synthesizer.js";
import { TaskGraphValidationError, TaskPool, topologicalTaskOrder } from "../src/task-pool.js";
import { MemoryTaskStore, type StoredTaskRun, type TaskStore } from "../src/task-store.js";

describe("TaskPool MVD", () => {
  it("PlanGraph → TaskGraph 适配确定性", () => {
    const parser = new GoalParser();
    const planner = new Planner();
    const goal = parser.parse("帮我规划迁移方案", "planning");
    const plan = planner.createPlan(goal);
    const graph = planGraphToTaskGraph(plan);
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
