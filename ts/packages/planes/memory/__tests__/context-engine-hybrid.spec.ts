/**
 * A1.3 ContextEngine.candidateRetrieve：opt-in 候选召回器走 hybrid，
 * 默认仍走 MemoryRetriever（行为零变化）；候选解析仍经原 ShaderPipeline。
 */
import { TaskType } from "@openintj/core";
import { describe, expect, it, vi } from "vitest";
import { ContextEngine, MemoryStore, type RankedMemory, fragmentsToRanked } from "../src/index.js";

describe("ContextEngine candidateRetrieve (A1.3)", () => {
  it("默认不注入：走 MemoryRetriever，候选召回器不被调用", async () => {
    const store = new MemoryStore();
    store.addLongTerm("the cat likes fish");
    const spy = vi.fn();
    const engine = new ContextEngine({ store });
    // 没有 candidateRetrieve → spy 永不触发
    await engine.build({
      query: "cat",
      history: [],
      taskType: TaskType.GENERAL_CHAT,
      systemPrompt: "p",
    });
    expect(spy).not.toHaveBeenCalled();
  });

  it("注入后：build 用 candidateRetrieve 的结果（并仍经着色）", async () => {
    const store = new MemoryStore();
    const f = store.addLongTerm("alpha beta gamma", { importance: 0.9 });
    const candidateRetrieve = vi.fn(
      async (): Promise<RankedMemory[]> => [
        { fragment: f, score: 0.99, components: { relevance: 0.99, keyword: 0, recency: 0 } },
      ],
    );
    const engine = new ContextEngine({ store, candidateRetrieve });
    const win = await engine.build({
      query: "alpha",
      history: [],
      taskType: TaskType.GENERAL_CHAT,
      systemPrompt: "p",
      topK: 6,
    });
    expect(candidateRetrieve).toHaveBeenCalledOnce();
    expect(candidateRetrieve.mock.calls[0]?.[1]).toMatchObject({ topK: 6 });
    // 着色后注入到 system prompt
    expect(win.systemPrompt).toContain("alpha beta gamma");
    expect(win.memoryFragments.length).toBe(1);
  });

  it("fragmentsToRanked：解析回片段、taskType 命中加成、bump accessCount", () => {
    const store = new MemoryStore();
    const tagged = store.addLongTerm("topic A", { taskTags: [TaskType.CODE_GENERATION] });
    const plain = store.addLongTerm("topic B");
    const before = tagged.accessCount;
    const ranked = fragmentsToRanked(
      store,
      [
        { id: tagged.fragmentId, score: 1 },
        { id: plain.fragmentId, score: 1 },
        { id: "missing", score: 1 },
      ],
      { taskType: TaskType.CODE_GENERATION },
    );
    expect(ranked.map((r) => r.fragment.fragmentId)).toEqual([tagged.fragmentId, plain.fragmentId]);
    // 命中 taskType 的片段 ×1.3
    expect(ranked[0]?.score).toBeCloseTo(1.3, 6);
    expect(ranked[1]?.score).toBe(1);
    // 命中即 bump
    expect(tagged.accessCount).toBe(before + 1);
  });
});
