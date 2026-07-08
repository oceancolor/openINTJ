import { SimpleEmbedder, contentHash } from "@openintj/core";
import { describe, expect, it } from "vitest";
import {
  InMemoryVectorStore,
  type VectorRow,
  VectorRowSchema,
  type VectorSearchResult,
  hybridVectorSearch,
  rrfFuse,
} from "../src/index.js";

const e = new SimpleEmbedder(16);

const mkRow = (id: string, content: string, tags: string[] = []): VectorRow =>
  VectorRowSchema.parse({
    fragmentId: id,
    content,
    embedding: e.embed(content),
    memoryType: "long_term",
    importance: 0.5,
    taskTags: tags,
    contentHash: contentHash({ content }),
    timestamp: Date.now() / 1000,
  });

const hit = (id: string, score: number): VectorSearchResult => ({
  row: mkRow(id, id),
  score,
  distance: 0,
});

describe("rrfFuse", () => {
  it("融合按名次而非分数量纲（异构分数也能合理排序）", () => {
    // 向量榜：a 第一（分极高），b 第二。文本榜：b 第一，c 第二。
    // b 两榜都靠前 → RRF 应把 b 顶上来。
    const vector = [hit("a", 0.99), hit("b", 0.5)];
    const text = [hit("b", 30), hit("c", 12)];
    const fused = rrfFuse([vector, text]);
    expect(fused[0]!.row.fragmentId).toBe("b");
    const ids = fused.map((f) => f.row.fragmentId).sort();
    expect(ids).toEqual(["a", "b", "c"]);
  });

  it("同一文档跨榜的贡献累加", () => {
    const only = rrfFuse([[hit("a", 1)]]);
    const twice = rrfFuse([[hit("a", 1)], [hit("a", 1)]]);
    expect(twice[0]!.score).toBeCloseTo(only[0]!.score * 2, 10);
  });

  it("topK 截断", () => {
    const fused = rrfFuse([[hit("a", 1), hit("b", 1), hit("c", 1)]], { topK: 2 });
    expect(fused).toHaveLength(2);
  });

  it("rrfK 越大头部名次差异越弱", () => {
    const listA = [hit("a", 1), hit("b", 1)];
    const small = rrfFuse([listA], { rrfK: 1 });
    const large = rrfFuse([listA], { rrfK: 1000 });
    const gapSmall = small[0]!.score - small[1]!.score;
    const gapLarge = large[0]!.score - large[1]!.score;
    expect(gapLarge).toBeLessThan(gapSmall);
  });

  it("空输入返回空", () => {
    expect(rrfFuse([])).toEqual([]);
    expect(rrfFuse([[]])).toEqual([]);
  });
});

describe("hybridVectorSearch", () => {
  const seed = async (): Promise<InMemoryVectorStore> => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([
      mkRow("db1", "postgres database index query optimization"),
      mkRow("db2", "sql transaction isolation acid database"),
      mkRow("ck1", "recipe pasta tomato garlic basil"),
      mkRow("ck2", "baking bread flour yeast oven"),
    ]);
    return s;
  };

  it("向量 + FTS 融合召回相关文档", async () => {
    const s = await seed();
    const out = await hybridVectorSearch(s, {
      query: "database query optimization",
      queryEmbedding: e.embed("database query optimization"),
      topK: 2,
    });
    expect(out.length).toBeGreaterThan(0);
    // 头部应是 database 主题
    expect(out[0]!.row.fragmentId.startsWith("db")).toBe(true);
  });

  it("FTS 命中纯词法项（向量可能漏掉的精确关键词）", async () => {
    const s = await seed();
    // "yeast" 是 ck2 独有词；确保 FTS 路把它召回
    const out = await hybridVectorSearch(s, {
      query: "yeast",
      queryEmbedding: e.embed("yeast"),
      topK: 4,
    });
    expect(out.some((r) => r.row.fragmentId === "ck2")).toBe(true);
  });

  it("store 无 searchText 时降级为纯向量检索", async () => {
    const s = await seed();
    // 模拟不支持 FTS 的 store：删掉 searchText
    const noFts = Object.assign(Object.create(Object.getPrototypeOf(s)), s, {
      searchText: undefined,
    }) as InMemoryVectorStore;
    const out = await hybridVectorSearch(noFts, {
      query: "database",
      queryEmbedding: e.embed("database query optimization"),
      topK: 2,
    });
    expect(out.length).toBeGreaterThan(0);
    expect(out.length).toBeLessThanOrEqual(2);
  });

  it("过滤语义（taskTags）两路一致生效", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([
      mkRow("a", "database index tuning", ["work"]),
      mkRow("b", "database index tuning", ["personal"]),
    ]);
    const out = await hybridVectorSearch(s, {
      query: "database index",
      queryEmbedding: e.embed("database index"),
      topK: 10,
      taskTags: ["work"],
    });
    expect(out).toHaveLength(1);
    expect(out[0]!.row.fragmentId).toBe("a");
  });
});
