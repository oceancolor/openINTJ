/**
 * AgentPool / forkJoin 可观测事件断言：传入真实 HookBus，捕获 pool.* / forkjoin.* 事件。
 */
import { HookBus, type HookEventMap } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { AgentPool } from "../src/agent-pool.js";
import { forkJoin } from "../src/fork-join.js";

const collect = <E extends keyof HookEventMap>(bus: HookBus, event: E): HookEventMap[E][] => {
  const out: HookEventMap[E][] = [];
  bus.on(event, (ctx) => {
    out.push(ctx.payload);
  });
  return out;
};

describe("AgentPool observability", () => {
  it("每个 job 发出 pool.beforeJob / pool.afterJob（成功）", async () => {
    const bus = new HookBus();
    const before = collect(bus, "pool.beforeJob");
    const after = collect(bus, "pool.afterJob");
    const pool = new AgentPool(2, { hooks: bus, name: "test-pool" });

    const results = await pool.map([1, 2, 3], async (n) => n * 2);
    expect(results).toEqual([2, 4, 6]);

    expect(before).toHaveLength(3);
    expect(after).toHaveLength(3);
    expect(before.every((b) => b.pool === "test-pool")).toBe(true);
    expect(after.every((a) => a.success)).toBe(true);
    // jobId 配对
    expect(new Set(after.map((a) => a.jobId))).toEqual(new Set(before.map((b) => b.jobId)));
    // 最终累计 completed=3 failed=0
    const last = after[after.length - 1]!;
    expect(last.completed).toBe(3);
    expect(last.failed).toBe(0);
  });

  it("失败 job 的 pool.afterJob.success=false 且 failed 计数递增", async () => {
    const bus = new HookBus();
    const after = collect(bus, "pool.afterJob");
    const pool = new AgentPool(1, { hooks: bus, name: "p" });

    await expect(
      pool.submit(async () => {
        throw new Error("boom");
      }, undefined),
    ).rejects.toThrow("boom");

    expect(after).toHaveLength(1);
    expect(after[0]!.success).toBe(false);
    expect(after[0]!.failed).toBe(1);
  });

  it("不传 hooks 时零事件（向后兼容）", async () => {
    const pool = new AgentPool(2);
    const r = await pool.map([1, 2], async (n) => n + 1);
    expect(r).toEqual([2, 3]);
  });
});

describe("forkJoin observability", () => {
  it("发出 forkjoin.beforeFork / forkjoin.afterJoin（含 fulfilled/rejected 计数）", async () => {
    const bus = new HookBus();
    const before = collect(bus, "forkjoin.beforeFork");
    const after = collect(bus, "forkjoin.afterJoin");

    const res = await forkJoin(
      [1, 2, 3],
      async (n) => {
        if (n === 2) throw new Error("fail-2");
        return n;
      },
      { hooks: bus, group: "vote" },
    );
    expect(res.fulfilled).toEqual([1, 3]);
    expect(res.rejected).toHaveLength(1);

    expect(before).toHaveLength(1);
    expect(before[0]).toMatchObject({ group: "vote", total: 3 });
    expect(after).toHaveLength(1);
    expect(after[0]).toMatchObject({ group: "vote", total: 3, fulfilled: 2, rejected: 1 });
  });
});
