import { describe, expect, it, vi } from "vitest";
import { DormantRuntime } from "../src/dormant-runtime.js";
import { PatternMiner } from "../src/pattern-miner.js";
import type { PassiveEvent } from "../src/types.js";
import { type OffThreadRunner, mineWithWorkerFallback } from "../src/worker-miner.js";

const mkEvents = (): PassiveEvent[] =>
  Array.from({ length: 6 }, (_, i) => ({
    eventId: `e${i}`,
    ts: Date.now() + i,
    source: "user" as const,
    text: "绿 茶 健 康",
    metadata: {},
  }));

const OPTS = { ngramSize: 2, minFrequency: 2, minConfidence: 0.3 };

describe("mineWithWorkerFallback（worker 下放 + 内联回退编排）", () => {
  it("runner 成功 → usedWorker=true，透传其结果", async () => {
    const sentinel = [
      {
        patternId: "p-worker",
        description: "from worker",
        evidenceIds: ["e0"],
        frequency: 6,
        confidence: 1,
        category: "other" as const,
        ts: 1,
      },
    ];
    const runner: OffThreadRunner = vi.fn(async () => sentinel);
    const r = await mineWithWorkerFallback(mkEvents(), OPTS, { runner });
    expect(r.usedWorker).toBe(true);
    expect(r.patterns).toBe(sentinel);
    expect(runner).toHaveBeenCalledOnce();
  });

  it("runner 抛错 → usedWorker=false，回退内联挖掘且与 PatternMiner 等价", async () => {
    const events = mkEvents();
    const runner: OffThreadRunner = vi.fn(async () => {
      throw new Error("worker unavailable");
    });
    const r = await mineWithWorkerFallback(events, OPTS, { runner });
    expect(r.usedWorker).toBe(false);
    // 与直接内联跑 PatternMiner 的模式集一致（忽略非确定性 patternId/ts）。
    const inline = await new PatternMiner(OPTS).mine(events);
    expect(r.patterns.map((p) => p.description).sort()).toEqual(
      inline.map((p) => p.description).sort(),
    );
    expect(r.patterns.length).toBe(inline.length);
    expect(r.patterns.length).toBeGreaterThan(0);
  });

  it("空事件回退路径也安全（0 模式）", async () => {
    const runner: OffThreadRunner = vi.fn(async () => {
      throw new Error("boom");
    });
    const r = await mineWithWorkerFallback([], OPTS, { runner });
    expect(r.usedWorker).toBe(false);
    expect(r.patterns).toEqual([]);
  });
});

describe("DormantRuntime.mineRunner 接线", () => {
  it("配了 mineRunner 且无 llmExtract → mine() 走 worker，lastMineUsedWorker=true", async () => {
    const runner: OffThreadRunner = vi.fn((events, opts) => new PatternMiner(opts).mine(events));
    const rt = new DormantRuntime({
      minerOpts: OPTS,
      mineRunner: runner,
    });
    for (let i = 0; i < 6; i++) rt.record("绿 茶 健 康", "user");
    const res = await rt.mine();
    expect(runner).toHaveBeenCalledOnce();
    expect(rt.lastMineUsedWorker).toBe(true);
    expect(res.patterns.length).toBeGreaterThan(0);
  });

  it("配了 llmExtract → 不走 worker（函数无法跨边界），lastMineUsedWorker=false", async () => {
    const runner: OffThreadRunner = vi.fn(async () => []);
    const rt = new DormantRuntime({
      minerOpts: {
        ...OPTS,
        llmExtract: async (ngram) => ({
          description: `pref: ${ngram}`,
          category: "preference",
        }),
      },
      mineRunner: runner,
    });
    for (let i = 0; i < 6; i++) rt.record("绿 茶 健 康", "user");
    const res = await rt.mine();
    expect(runner).not.toHaveBeenCalled();
    expect(rt.lastMineUsedWorker).toBe(false);
    // llmExtract 打了 preference 类别 → 有可建议的提案
    expect(res.proposals.length).toBeGreaterThan(0);
  });

  it("未配 mineRunner → 内联挖掘，lastMineUsedWorker 恒 false", async () => {
    const rt = new DormantRuntime({ minerOpts: OPTS });
    for (let i = 0; i < 6; i++) rt.record("绿 茶 健 康", "user");
    await rt.mine();
    expect(rt.lastMineUsedWorker).toBe(false);
  });
});
