import { HookBus, type HookLogger, TaskType } from "@openintj/core";
import { describe, expect, it, vi } from "vitest";
import {
  MemoryPlane,
  MemoryRetriever,
  MemoryStore,
  cosineSimilarity,
  simpleEmbedding,
} from "../src/index.js";

const silentLogger: HookLogger = { warn: () => {}, error: () => {} };

describe("simpleEmbedding", () => {
  it("is deterministic for same input", () => {
    const a = simpleEmbedding("hello world");
    const b = simpleEmbedding("hello world");
    expect(a).toEqual(b);
  });

  it("yields normalized vectors", () => {
    const v = simpleEmbedding("test");
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it("respects requested dim", () => {
    expect(simpleEmbedding("x", 32)).toHaveLength(32);
    expect(simpleEmbedding("x", 128)).toHaveLength(128);
  });
});

describe("cosineSimilarity", () => {
  it("returns 1 for identical vectors", () => {
    const v = simpleEmbedding("hello");
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 6);
  });

  it("returns 0 for empty input", () => {
    expect(cosineSimilarity([], [])).toBe(0);
  });

  it("returns 0 for length mismatch", () => {
    expect(cosineSimilarity([1, 0], [1, 0, 0])).toBe(0);
  });
});

describe("MemoryStore", () => {
  it("adds across three tiers and counts correctly", () => {
    const s = new MemoryStore();
    s.addShortTerm("u-1");
    s.addWorking("w-1");
    s.addLongTerm("l-1");
    expect(s.totalCount).toBe(3);
    expect(s.countsByTier()).toEqual({ short_term: 1, working: 1, long_term: 1 });
  });

  it("migrates oldest short_term to long_term on overflow", () => {
    const s = new MemoryStore({ maxShortTerm: 2 });
    s.addShortTerm("a");
    s.addShortTerm("b");
    s.addShortTerm("c");
    expect(s.shortTerm.map((f) => f.content)).toEqual(["b", "c"]);
    expect(s.longTerm.map((f) => f.content)).toEqual(["a"]);
  });

  it("drops oldest working memory on overflow (no migration)", () => {
    const s = new MemoryStore({ maxWorking: 2 });
    s.addWorking("a");
    s.addWorking("b");
    s.addWorking("c");
    expect(s.working.map((f) => f.content)).toEqual(["b", "c"]);
    expect(s.longTerm).toHaveLength(0);
  });

  it("removes by fragmentId", () => {
    const s = new MemoryStore();
    const f = s.addShortTerm("x");
    expect(s.remove(f.fragmentId)).toBe(true);
    expect(s.totalCount).toBe(0);
  });

  it("clearWorking empties working tier only", () => {
    const s = new MemoryStore();
    s.addShortTerm("a");
    s.addWorking("b");
    s.clearWorking();
    expect(s.shortTerm).toHaveLength(1);
    expect(s.working).toHaveLength(0);
  });
});

describe("MemoryRetriever", () => {
  it("returns top-k by combined score, descending", () => {
    const s = new MemoryStore();
    s.addShortTerm("the cat sat on the mat");
    s.addShortTerm("dogs are loyal animals");
    s.addShortTerm("cats and dogs are pets");
    const r = new MemoryRetriever(s);
    const ranked = r.retrieve("cat dog pets", { topK: 2 });
    expect(ranked).toHaveLength(2);
    expect(ranked[0]!.score).toBeGreaterThanOrEqual(ranked[1]!.score);
  });

  it("uses CJK bigrams for zero-dependency keyword recall", () => {
    const s = new MemoryStore();
    s.addShortTerm("另外，我所在的城市是杭州，习惯用公制单位。");
    s.addShortTerm("约束 A：数据库必须用 SQLite，不能引入外部服务。");
    const r = new MemoryRetriever(s, {
      relevanceWeight: 0,
      recencyWeight: 1,
      importanceWeight: 0,
    });
    const ranked = r.retrieve("我在哪个城市？", { topK: 2 });
    expect(ranked[0]!.fragment.content).toContain("杭州");
    expect(ranked[0]!.components.keyword).toBeGreaterThan(ranked[1]!.components.keyword);
  });

  it("filters by minImportance after decay", () => {
    const now = 1_000_000;
    const s = new MemoryStore();
    s.addShortTerm("recent", { importance: 0.9 });
    // very old fragment
    const old = s.addShortTerm("ancient", { importance: 0.8 });
    old.timestamp = now - 24 * 3600 * 10; // 10 half-lives ago
    const r = new MemoryRetriever(s, { recencyHalfLifeHours: 24 }, { clock: () => now });
    const ranked = r.retrieve("topic", { minImportance: 0.5 });
    expect(ranked.map((x) => x.fragment.content)).toContain("recent");
    expect(ranked.map((x) => x.fragment.content)).not.toContain("ancient");
  });

  it("applies task tag boost (×1.3)", () => {
    const s = new MemoryStore();
    s.addShortTerm("tagged", { taskTags: [TaskType.CODE_GENERATION] });
    s.addShortTerm("untagged");
    const r = new MemoryRetriever(s);
    // 同样 query；同样向量分；只有 tag 不同
    const noTask = r.retrieve("xx");
    const withTask = r.retrieve("xx", { taskType: TaskType.CODE_GENERATION });
    const taggedNoBoost = noTask.find((x) => x.fragment.content === "tagged");
    const taggedBoosted = withTask.find((x) => x.fragment.content === "tagged");
    expect(taggedBoosted!.score).toBeCloseTo(taggedNoBoost!.score * 1.3, 5);
  });

  it("uses recencyHalfLifeHours independently of maxSummaryLength (RFC-003 fix)", () => {
    const now = 1_000_000;
    const s = new MemoryStore();
    s.addShortTerm("old item", { importance: 1 }).timestamp = now - 24 * 3600;
    s.addShortTerm("very old", { importance: 1 }).timestamp = now - 24 * 3600 * 100;

    // 短半衰期 → very old 几乎归零
    const rShort = new MemoryRetriever(
      s,
      { recencyHalfLifeHours: 24, importanceWeight: 1, relevanceWeight: 0, recencyWeight: 0 },
      { clock: () => now },
    );
    const rankedShort = rShort.retrieve("query");
    const veryOldShort = rankedShort.find((x) => x.fragment.content === "very old");

    // 长半衰期 → very old 仍接近 1
    // 重置访问计数（避免之前 retrieve 的副作用影响）
    for (const f of s.all) {
      f.accessCount = 0;
    }
    const rLong = new MemoryRetriever(
      s,
      {
        recencyHalfLifeHours: 24 * 365,
        importanceWeight: 1,
        relevanceWeight: 0,
        recencyWeight: 0,
      },
      { clock: () => now },
    );
    const rankedLong = rLong.retrieve("query");
    const veryOldLong = rankedLong.find((x) => x.fragment.content === "very old");

    expect(veryOldLong!.components.recency).toBeGreaterThan(veryOldShort!.components.recency);
  });

  it("increments accessCount on retrieved fragments", () => {
    const s = new MemoryStore();
    const f = s.addShortTerm("hit me");
    const r = new MemoryRetriever(s);
    r.retrieve("hit me");
    expect(f.accessCount).toBe(1);
    r.retrieve("hit me");
    expect(f.accessCount).toBe(2);
  });
});

describe("MemoryPlane", () => {
  it("emits event.MEMORY_LOADED on retrieve", async () => {
    const hooks = new HookBus({ logger: silentLogger });
    const plane = new MemoryPlane({ hooks });
    plane.store.addShortTerm("alpha");
    plane.store.addShortTerm("beta");
    const handler = vi.fn();
    hooks.on("event.MEMORY_LOADED", handler);
    const ranked = await plane.retrieve("alpha");
    expect(handler).toHaveBeenCalledOnce();
    expect(ranked.length).toBeGreaterThan(0);
  });

  it("recordUserInput tags as user_input", () => {
    const plane = new MemoryPlane();
    const f = plane.recordUserInput("hello");
    expect(f.taskTags).toContain("user_input");
    expect(plane.getStats().total).toBe(1);
  });
});
