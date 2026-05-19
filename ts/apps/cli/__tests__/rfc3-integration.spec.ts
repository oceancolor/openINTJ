/**
 * RFC-003 三方向集成测试
 *
 * 验证：
 *  1. 多线程 Agent 编排：AgentPool + ForkJoin 并发跑 N 个 query，结果有序
 *  2. 任务池调度：TaskQueue + ObjectPool 缓存检索结果，DAG 顺序正确
 *  3. Dormant pattern 提取与审批回路：从交互流水中挖掘 pattern → propose → approve
 */

import { AgentPool, forkJoin, majorityVote } from "@openintj/concurrency";
import { InternalizationManager, PassiveStore, PatternMiner } from "@openintj/dormant";
import { HybridRetriever, ObjectPool, SharedContext, TaskQueue } from "@openintj/taskpool";
import { describe, expect, it } from "vitest";
import { assembleAgent } from "../src/agent.js";

describe("RFC-003 方向 1: 多线程 Agent 编排", () => {
  it("AgentPool 并发跑 4 个 mock agent，结果按提交顺序返回", async () => {
    const pool = new AgentPool(2);
    const queries = ["问题A", "问题B", "问题C", "问题D"];
    const results = await pool.map(queries, async (q) => {
      const agent = assembleAgent({ llmProvider: "mock" });
      const r = await agent.run(q);
      return { q, answer: r.finalAnswer };
    });
    expect(results).toHaveLength(4);
    for (let i = 0; i < 4; i++) {
      expect(results[i]!.q).toBe(queries[i]);
      expect(results[i]!.answer.length).toBeGreaterThan(0);
    }
  });

  it("ForkJoin majority vote 选最多的回答", async () => {
    const r = await forkJoin(
      [1, 2, 3, 4, 5],
      async (n) => (n % 2 === 0 ? "answer-X" : "answer-Y"),
      { reducer: majorityVote },
    );
    expect(r.reduced).toBe("answer-Y"); // 3 个奇数
    expect(r.fulfilled).toHaveLength(5);
  });

  it("ForkJoin minSuccess 强制最低成功数", async () => {
    await expect(
      forkJoin(
        [1, 2, 3],
        async (n) => {
          if (n === 1) return "ok";
          throw new Error("fail");
        },
        { minSuccess: 2 },
      ),
    ).rejects.toThrow(/required 2/);
  });
});

describe("RFC-003 方向 2: 任务池 + 对象池 + 混合检索", () => {
  it("SharedContext 在多个 worker 之间安全更新", async () => {
    const ctx = new SharedContext();
    await ctx.set("counter", 0);
    const pool = new AgentPool(4);
    await pool.map(new Array(20).fill(0), async () => {
      await ctx.update<number>("counter", (n) => (n ?? 0) + 1);
    });
    expect(ctx.get<number>("counter")).toBe(20);
  });

  it("TaskQueue + AgentPool 跑 DAG 任务，依赖顺序正确", async () => {
    const q = new TaskQueue();
    const log: string[] = [];
    await q.submit({
      id: "fetch",
      priority: 5,
      deps: [],
      payload: undefined,
      run: () => {
        log.push("fetch");
      },
    });
    await q.submit({
      id: "parse",
      priority: 3,
      deps: ["fetch"],
      payload: undefined,
      run: () => {
        log.push("parse");
      },
    });
    await q.submit({
      id: "summarize",
      priority: 1,
      deps: ["parse"],
      payload: undefined,
      run: () => {
        log.push("summarize");
      },
    });
    q.close();

    while (true) {
      const t = await q.dequeue();
      if (!t) break;
      try {
        await t.node.run(t.node.payload);
        await q.complete(t.node.id, undefined);
      } catch (e) {
        await q.fail(t.node.id, e);
      }
    }
    expect(log).toEqual(["fetch", "parse", "summarize"]);
  });

  it("ObjectPool hot/warm/cold 分层缓存，热点提升", async () => {
    const op = new ObjectPool<string>({
      hotPromoteAt: 2,
      hotCapacity: 1,
      warmCapacity: 2,
      coldCapacity: 5,
    });
    await op.set("hot_key", "value");
    await op.get("hot_key");
    await op.get("hot_key"); // hits hotPromoteAt
    expect(op.stats().hot).toBe(1);
  });

  it("HybridRetriever 检索质量优于纯向量", () => {
    const r = new HybridRetriever();
    r.index([
      {
        id: "1",
        text: "machine learning frameworks pytorch tensorflow",
        vector: [1, 0, 0, 0],
      },
      {
        id: "2",
        text: "deep learning research paper survey",
        vector: [0, 1, 0, 0],
      },
      {
        id: "3",
        text: "machine learning tutorials beginner",
        vector: [0, 0, 1, 0],
      },
    ]);
    // 关键词匹配 "machine learning" 应该把 1 和 3 拉到前面
    const out = r.search("machine learning", [0, 0, 0, 1], 3);
    const ids = out.slice(0, 2).map((o) => o.doc.id);
    expect(ids).toContain("1");
    expect(ids).toContain("3");
  });
});

describe("RFC-003 方向 3: Dormant Memory Learning（审批回路）", () => {
  it("用户对话流水 → PatternMiner → 提案 → 用户审批 → 写入 PersonaConfig", async () => {
    const passive = new PassiveStore();
    // 用户多次提及偏好（语料带轻微变化以模拟真实交互）
    const variants = [
      "我喜欢喝绿茶",
      "今天我喜欢喝绿茶",
      "我喜欢喝绿茶啊",
      "总是喜欢喝绿茶",
      "晚饭后喜欢喝绿茶",
      "其实我喜欢喝绿茶",
      "你知道我喜欢喝绿茶",
      "我也喜欢喝绿茶",
    ];
    variants.forEach((text, i) => {
      passive.record({
        eventId: `e${i}`,
        ts: Date.now() + i,
        source: "user",
        text,
        metadata: {},
      });
    });
    const miner = new PatternMiner({
      ngramSize: 3,
      minFrequency: 5,
      minConfidence: 0.5,
    });
    const patterns = await miner.mine(passive.exportAll());
    expect(patterns.length).toBeGreaterThan(0);

    const im = new InternalizationManager();
    const proposals = im.proposeBatch(
      patterns.map((p) => ({ ...p, category: "preference" as const })),
    );
    expect(proposals.length).toBeGreaterThan(0);

    // 模拟用户在 UI 审批
    const first = proposals[0]!;
    const approved = im.approve(first.proposalId);
    expect(approved?.status).toBe("applied");

    const cfg = im.snapshot();
    expect(Object.keys(cfg.preferences).length).toBeGreaterThan(0);
    expect(cfg.meta.version).toBe(1);
  });

  it("用户拒绝提案后 PersonaConfig 不变", async () => {
    const im = new InternalizationManager();
    const p = im.propose({
      patternId: "p1",
      description: "有争议的偏好",
      evidenceIds: ["e1"],
      frequency: 10,
      confidence: 0.9,
      category: "preference",
      ts: Date.now(),
    });
    im.reject(p!.proposalId);
    const cfg = im.snapshot();
    expect(Object.keys(cfg.preferences).length).toBe(0);
    expect(im.listProposals("rejected")).toHaveLength(1);
  });

  it("LLM 抽取增强 pattern description（注入式）", async () => {
    const passive = new PassiveStore();
    for (let i = 0; i < 5; i++) {
      passive.record({
        eventId: `e${i}`,
        ts: Date.now(),
        source: "user",
        text: "记 得 提 醒 我",
        metadata: {},
      });
    }
    const miner = new PatternMiner({
      ngramSize: 3,
      minFrequency: 3,
      minConfidence: 0.4,
      llmExtract: async (ngram) => ({
        description: `用户习惯让我记住事项: ${ngram}`,
        category: "habit",
      }),
    });
    const patterns = await miner.mine(passive.exportAll());
    expect(patterns[0]!.description).toContain("用户习惯");
    expect(patterns[0]!.category).toBe("habit");
  });
});
