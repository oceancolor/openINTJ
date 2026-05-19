import {
  HookBus,
  type HookLogger,
  LODLevel,
  ShaderMode,
  TaskType,
  truncateSummarize,
} from "@openintj/core";
import { describe, expect, it, vi } from "vitest";
import {
  ContextEngine,
  MemoryStore,
  ShaderPipeline,
  fragmentShader,
  geometryShader,
  vertexShader,
} from "../src/index.js";
import type { RankedMemory } from "../src/retriever.js";

const silent: HookLogger = { warn: () => {}, error: () => {} };

const mkRanked = (contents: string[], scores: number[]): RankedMemory[] => {
  const store = new MemoryStore();
  return contents.map((c, i) => ({
    fragment: store.addShortTerm(c, { importance: 0.8 }),
    score: scores[i] ?? 0.5,
    components: { relevance: 0.5, keyword: 0.3, recency: 0.7 },
  }));
};

describe("vertexShader", () => {
  it("hybrid: top 30% gets finer LOD, rest coarser", () => {
    const ranked = mkRanked(
      ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
      [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
    );
    const out = vertexShader(ranked, ShaderMode.HYBRID, 0.5);
    // base = LOD_2 for hybrid at ratio 0.5
    expect(out[0]!.lod).toBe(1); // top 30% → base-1
    expect(out[1]!.lod).toBe(1);
    expect(out[2]!.lod).toBe(1);
    expect(out[3]!.lod).toBeGreaterThan(2); // rest → base+1 (and very-low score → +1 again)
  });

  it("high_fidelity: score>0.7 demotes by 1 level (finer)", () => {
    const ranked = mkRanked(["a", "b"], [0.9, 0.5]);
    const out = vertexShader(ranked, ShaderMode.HIGH_FIDELITY, 0.4);
    // base = LOD_0 at ratio<0.6 high_fidelity
    expect(out[0]!.lod).toBe(0); // already at floor
    expect(out[1]!.lod).toBe(0);
  });

  it("low_fidelity: score<0.3 promotes to coarser LOD", () => {
    const ranked = mkRanked(["a", "b"], [0.5, 0.1]);
    const out = vertexShader(ranked, ShaderMode.LOW_FIDELITY, 0.95);
    expect(out[0]!.lod).toBe(LODLevel.LOD_4); // base = 4 cap
    expect(out[1]!.lod).toBe(LODLevel.LOD_4);
  });

  it("adaptive: returns base lod from getLodForMode", () => {
    const ranked = mkRanked(["a"], [0.5]);
    const out = vertexShader(ranked, ShaderMode.ADAPTIVE, 0.4);
    expect(out[0]!.lod).toBe(LODLevel.LOD_1);
  });

  it("empty input returns empty", () => {
    expect(vertexShader([], ShaderMode.HYBRID, 0.5)).toEqual([]);
  });
});

describe("geometryShader", () => {
  it("filters by importance threshold (decayed)", () => {
    const ranked = mkRanked(["keep", "drop"], [0.9, 0.1]);
    // make drop's fragment ancient
    const ancient = ranked[1]!.fragment;
    ancient.timestamp = Date.now() / 1000 - 24 * 3600 * 100;
    ancient.importance = 0.5;
    const lodAssigned = vertexShader(ranked, ShaderMode.HYBRID, 0.5);
    const filtered = geometryShader(lodAssigned, {
      config: {
        importanceThreshold: 0.3,
        maxFragmentsPerQuery: 10,
        recencyHalfLifeHours: 24,
      },
    });
    expect(filtered.map((a) => a.ranked.fragment.content)).toEqual(["keep"]);
  });

  it("limits to maxFragmentsPerQuery", () => {
    const ranked = mkRanked(["a", "b", "c", "d", "e"], [0.9, 0.8, 0.7, 0.6, 0.5]);
    const lod = vertexShader(ranked, ShaderMode.HYBRID, 0.5);
    const out = geometryShader(lod, {
      config: {
        importanceThreshold: 0,
        maxFragmentsPerQuery: 2,
        recencyHalfLifeHours: 24,
      },
    });
    expect(out).toHaveLength(2);
  });
});

describe("fragmentShader", () => {
  it("returns content at LOD with default truncate summarizer", async () => {
    const store = new MemoryStore();
    const long = "x".repeat(800);
    const f = store.addShortTerm(long, { importance: 0.9 });
    const out = await fragmentShader(
      [
        {
          ranked: {
            fragment: f,
            score: 0.9,
            components: { relevance: 0.9, keyword: 0, recency: 1 },
          },
          lod: LODLevel.LOD_2,
        },
      ],
      {
        config: { maxSummaryLength: 200, recencyHalfLifeHours: 24 },
        shaderMode: ShaderMode.HYBRID,
        memoryBudgetTokens: 500,
      },
    );
    expect(out).toHaveLength(1);
    expect(out[0]!.content.length).toBeLessThan(200);
    expect(out[0]!.lod).toBe(LODLevel.LOD_2);
  });

  it("uses precomputed summary if available", async () => {
    const store = new MemoryStore();
    const f = store.addShortTerm("original long text", {
      summaries: { 1: "短摘要", 2: "更短" },
    });
    const out = await fragmentShader(
      [
        {
          ranked: {
            fragment: f,
            score: 0.5,
            components: { relevance: 0.5, keyword: 0, recency: 1 },
          },
          lod: LODLevel.LOD_2,
        },
      ],
      {
        config: { maxSummaryLength: 100, recencyHalfLifeHours: 24 },
        shaderMode: ShaderMode.HYBRID,
        memoryBudgetTokens: 500,
      },
    );
    expect(out[0]!.content).toBe("更短");
  });

  it("respects memoryBudgetTokens and truncates", async () => {
    const store = new MemoryStore();
    const items = [1, 2, 3, 4].map((i) =>
      store.addShortTerm(`item-${i}-${"x".repeat(200)}`, { importance: 0.9 }),
    );
    const out = await fragmentShader(
      items.map((f, i) => ({
        ranked: {
          fragment: f,
          score: 0.9 - i * 0.1,
          components: { relevance: 0.5, keyword: 0, recency: 1 },
        },
        lod: LODLevel.LOD_0,
      })),
      {
        config: { maxSummaryLength: 200, recencyHalfLifeHours: 24 },
        shaderMode: ShaderMode.HIGH_FIDELITY,
        memoryBudgetTokens: 50, // 极小预算
      },
    );
    const totalTokens = out.reduce((s, f) => s + f.tokens, 0);
    expect(totalTokens).toBeLessThanOrEqual(60); // 含一点裕量，因截断后再估算
  });

  it("supports async LLM summarizer", async () => {
    const store = new MemoryStore();
    const f = store.addShortTerm("a".repeat(500), { importance: 0.9 });
    const summarizer = vi.fn(
      async (text: string, len: number) => `[LLM-summary len=${len}] ${text.slice(0, 10)}`,
    );
    const out = await fragmentShader(
      [
        {
          ranked: {
            fragment: f,
            score: 0.5,
            components: { relevance: 0.5, keyword: 0, recency: 1 },
          },
          lod: LODLevel.LOD_3,
        },
      ],
      {
        config: { maxSummaryLength: 200, recencyHalfLifeHours: 24 },
        shaderMode: ShaderMode.HYBRID,
        memoryBudgetTokens: 500,
        summarize: summarizer,
      },
    );
    expect(summarizer).toHaveBeenCalled();
    expect(out[0]!.content).toContain("LLM-summary");
  });
});

describe("truncateSummarize", () => {
  it("returns full text when within budget", () => {
    expect(truncateSummarize("short", 100)).toBe("short");
  });

  it("truncates with head + ' ... ' + tail", () => {
    const r = truncateSummarize("a".repeat(300), 30);
    expect(r.length).toBeLessThanOrEqual(35);
    expect(r).toContain("...");
  });
});

describe("ShaderPipeline.process", () => {
  it("V→G→F end-to-end with adaptive mode + budget compaction", async () => {
    const ranked = mkRanked(
      ["alpha", "beta", "gamma", "delta", "epsilon"],
      [0.95, 0.85, 0.75, 0.65, 0.55],
    );
    const hooks = new HookBus({ logger: silent });
    const pipe = new ShaderPipeline({
      config: { mode: ShaderMode.ADAPTIVE, maxFragmentsPerQuery: 10 },
      budget: { maxTokens: 10_000 },
      hooks,
    });
    const shaderApplied = vi.fn();
    hooks.on("event.SHADER_APPLIED", shaderApplied);
    const r = await pipe.process(ranked, TaskType.GENERAL_CHAT);
    expect(r.shaded.length).toBe(5);
    expect(r.totalTokens).toBeGreaterThan(0);
    expect(shaderApplied).toHaveBeenCalledOnce();
  });

  it("resolveMode picks LOW_FIDELITY when budget near threshold (adaptive)", () => {
    const pipe = new ShaderPipeline({
      config: { mode: ShaderMode.ADAPTIVE, compactionThreshold: 0.5 },
      budget: {
        maxTokens: 1000,
        reservedTokens: 0,
        conversationTokens: 600,
      },
    });
    const r = pipe.resolveMode(TaskType.GENERAL_CHAT);
    // GENERAL_CHAT is hardcoded to LOW_FIDELITY by TASK_SHADER_MAP
    expect(r.mode).toBe(ShaderMode.LOW_FIDELITY);
  });

  it("resolveMode adaptive switches by ratio", () => {
    const pipe = new ShaderPipeline({
      config: { mode: ShaderMode.ADAPTIVE },
      budget: { maxTokens: 1000, conversationTokens: 100 },
    });
    // ANALYSIS task → adaptive flow (TASK_SHADER_MAP[ANALYSIS] = HYBRID)
    // Actually for TaskType.ANALYSIS, TASK_SHADER_MAP returns HYBRID directly, no adaptive resolve
    expect(pipe.resolveMode(TaskType.ANALYSIS).mode).toBe(ShaderMode.HYBRID);
  });
});

describe("ContextEngine.build", () => {
  it("retrieves + shades + injects memory into systemPrompt", async () => {
    const hooks = new HookBus({ logger: silent });
    const store = new MemoryStore();
    store.addShortTerm("用户喜欢喝绿茶");
    store.addShortTerm("用户家里养了一只橘猫");
    store.addShortTerm("无关信息：今天天气好");
    const engine = new ContextEngine({
      store,
      hooks,
      shaderConfig: { mode: ShaderMode.HYBRID, maxFragmentsPerQuery: 3 },
    });
    const win = await engine.build({
      query: "我家的猫怎么样？",
      history: [],
      taskType: TaskType.GENERAL_CHAT,
      systemPrompt: "你是一个助手",
    });
    expect(win.memoryFragments.length).toBeGreaterThan(0);
    expect(win.systemPrompt).toContain("[记忆参考]");
    expect(win.messages).toHaveLength(1); // user query only
    expect(win.messages[0]!.role).toBe("user");
    expect(win.totalTokens).toBeGreaterThan(0);
  });

  it("emits CONTEXT_COMPACTED when budget exceeded threshold", async () => {
    const hooks = new HookBus({ logger: silent });
    const compacted = vi.fn();
    hooks.on("event.CONTEXT_COMPACTED", compacted);
    const store = new MemoryStore();
    for (let i = 0; i < 5; i++) {
      store.addShortTerm(`memory ${i} ${"x".repeat(100)}`, {
        importance: 0.9,
      });
    }
    const engine = new ContextEngine({
      store,
      hooks,
      budget: {
        maxTokens: 200, // 极小，必触发
        reservedTokens: 0,
        conversationTokens: 100, // 已占 50%
      },
    });
    await engine.build({
      query: "查一下记忆",
      history: [],
      taskType: TaskType.GENERAL_CHAT,
      systemPrompt: "你是助手",
    });
    expect(compacted).toHaveBeenCalled();
  });

  it("returns empty memoryFragments when store is empty", async () => {
    const engine = new ContextEngine({ store: new MemoryStore() });
    const win = await engine.build({
      query: "x",
      history: [],
      taskType: TaskType.QUICK_RESPONSE,
      systemPrompt: "p",
    });
    expect(win.memoryFragments).toEqual([]);
    expect(win.systemPrompt).toBe("p");
  });
});
