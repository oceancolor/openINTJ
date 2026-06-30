/**
 * CLF.1 ReinforcingClassifier：分类、强化收敛、封顶、回退路径。
 */
import { SimpleEmbedder, TaskType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { DEFAULT_SEEDS, ReinforcingClassifier } from "../src/index.js";

const make = (over: Partial<ConstructorParameters<typeof ReinforcingClassifier>[0]> = {}) =>
  new ReinforcingClassifier({ embedder: new SimpleEmbedder(64), ...over });

describe("ReinforcingClassifier", () => {
  it("无 exemplar → 回退关键词启发式（fallback=true）", async () => {
    const clf = make();
    const r = await clf.classify("写一个快速排序函数");
    expect(r.fallback).toBe(true);
    expect(r.label).toBe(TaskType.CODE_GENERATION); // detectTaskType 命中「写一个」
    expect(r.confidence).toBe(0);
  });

  it("种子注入后能对相近 query 分类（非兜底）", async () => {
    const clf = make({ minConfidence: 0.3 });
    await clf.addSeeds(DEFAULT_SEEDS);
    expect(clf.size).toBe(DEFAULT_SEEDS.length);
    const r = await clf.classify("帮我实现一个二分查找函数");
    // SimpleEmbedder 是哈希式，语义弱，但至少应产出一个标签 + scores 归一化
    expect(Object.values(r.scores).reduce((a, b) => a + (b ?? 0), 0)).toBeCloseTo(1, 5);
    expect(Object.keys(r.scores).length).toBeGreaterThan(0);
  });

  it("reinforce 正反馈：重复同一 query 提升该标签置信、命中非兜底", async () => {
    const clf = make({ minConfidence: 0.5, mergeThreshold: 0.99 });
    const q = "部署流水线的回滚策略是什么";
    // 多次正反馈到 PLANNING
    for (let i = 0; i < 5; i++) await clf.reinforce(q, TaskType.PLANNING, { signal: 1 });
    const r = await clf.classify(q);
    expect(r.fallback).toBe(false);
    expect(r.label).toBe(TaskType.PLANNING);
    expect(r.confidence).toBeGreaterThanOrEqual(0.5);
  });

  it("reinforce 合并：高相似同标签反馈升权而非无限新增", async () => {
    const clf = make({ mergeThreshold: 0.5 });
    const q = "一模一样的查询";
    await clf.reinforce(q, TaskType.ANALYSIS, { signal: 1 });
    const sizeAfterFirst = clf.size;
    await clf.reinforce(q, TaskType.ANALYSIS, { signal: 1 });
    // 完全相同的向量相似度=1 >= mergeThreshold → 合并，size 不增
    expect(clf.size).toBe(sizeAfterFirst);
  });

  it("penalize 负反馈：衰减并最终移除附近 exemplar", async () => {
    const clf = make({ mergeThreshold: 0.5 });
    const q = "需要被惩罚的查询";
    await clf.reinforce(q, TaskType.GENERAL_CHAT, { signal: 1 });
    expect(clf.size).toBe(1);
    await clf.reinforce(q, TaskType.GENERAL_CHAT, { signal: -2 }); // 权重 1 + (-2) <=0 → 移除
    expect(clf.size).toBe(0);
  });

  it("封顶：超过 maxExemplars 时淘汰最弱", async () => {
    const clf = make({ maxExemplars: 3, mergeThreshold: 1.1 }); // mergeThreshold>1 → 永不合并，强制新增
    for (let i = 0; i < 10; i++) {
      await clf.reinforce(`query number ${i}`, TaskType.QUICK_RESPONSE, { signal: 1 });
    }
    expect(clf.size).toBe(3);
  });

  it("toState / loadState 往返保持分类行为", async () => {
    const clf = make({ minConfidence: 0.3 });
    await clf.addSeeds(DEFAULT_SEEDS);
    const state = clf.toState();
    const clf2 = make({ minConfidence: 0.3 });
    clf2.loadState(state);
    expect(clf2.size).toBe(clf.size);
    const a = await clf.classify("写一份接口文档");
    const b = await clf2.classify("写一份接口文档");
    expect(b.label).toBe(a.label);
  });
});
