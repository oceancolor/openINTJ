/**
 * TaskQueue 可观测事件断言：task.enqueue / task.beforeRun / task.afterRun。
 */
import { HookBus, type HookEventMap } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { TaskQueue } from "../src/task-queue.js";

const collect = <E extends keyof HookEventMap>(bus: HookBus, event: E): HookEventMap[E][] => {
  const out: HookEventMap[E][] = [];
  bus.on(event, (ctx) => {
    out.push(ctx.payload);
  });
  return out;
};

describe("TaskQueue observability", () => {
  it("submit → enqueue 事件；dequeue → beforeRun；complete → afterRun(success)", async () => {
    const bus = new HookBus();
    const enq = collect(bus, "task.enqueue");
    const before = collect(bus, "task.beforeRun");
    const after = collect(bus, "task.afterRun");
    const q = new TaskQueue({ hooks: bus, name: "q1" });

    await q.submit({ id: "a", priority: 1, deps: [], payload: 0, run: () => 1 });
    await q.submit({ id: "b", priority: 5, deps: ["a"], payload: 0, run: () => 2 });

    expect(enq).toHaveLength(2);
    expect(enq.find((e) => e.taskId === "a")).toMatchObject({ ready: true, depCount: 0 });
    expect(enq.find((e) => e.taskId === "b")).toMatchObject({ ready: false, depCount: 1 });

    // a 先就绪（b 依赖 a）
    const t1 = await q.dequeue();
    expect(t1?.node.id).toBe("a");
    expect(before).toHaveLength(1);
    expect(before[0]).toMatchObject({ taskId: "a", priority: 1, queue: "q1" });

    await q.complete("a", "ra");
    expect(after).toHaveLength(1);
    expect(after[0]).toMatchObject({ taskId: "a", success: true });
    expect(after[0]!.durationMs).toBeGreaterThanOrEqual(0);

    // a 完成后 b 就绪
    const t2 = await q.dequeue();
    expect(t2?.node.id).toBe("b");
    await q.complete("b", "rb");
    expect(after).toHaveLength(2);
    expect(after[1]).toMatchObject({ taskId: "b", success: true });
  });

  it("fail → afterRun(success=false)", async () => {
    const bus = new HookBus();
    const after = collect(bus, "task.afterRun");
    const q = new TaskQueue({ hooks: bus });

    await q.submit({ id: "x", priority: 1, deps: [], payload: 0, run: () => 1 });
    await q.dequeue();
    await q.fail("x", new Error("nope"));

    expect(after).toHaveLength(1);
    expect(after[0]).toMatchObject({ taskId: "x", success: false });
  });

  it("不传 hooks 时零事件（向后兼容）", async () => {
    const q = new TaskQueue();
    await q.submit({ id: "a", priority: 1, deps: [], payload: 0, run: () => 1 });
    const t = await q.dequeue();
    expect(t?.node.id).toBe("a");
    await q.complete("a", 1);
    expect(q.result("a").state).toBe("completed");
  });
});
