import { SimpleEmbedder, contentHash } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { InMemoryVectorStore, VectorRowSchema } from "../src/index.js";

const e = new SimpleEmbedder(8);

const mkRow = (id: string, content: string, tags: string[] = []) =>
  VectorRowSchema.parse({
    fragmentId: id,
    content,
    embedding: e.embed(content),
    memoryType: "short_term",
    importance: 0.5,
    taskTags: tags,
    contentHash: contentHash({ content }),
    timestamp: Date.now() / 1000,
  });

describe("InMemoryVectorStore", () => {
  it("init/upsert/count cycle works", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    expect(await s.count()).toBe(0);
    await s.upsert([mkRow("a", "alpha"), mkRow("b", "beta")]);
    expect(await s.count()).toBe(2);
    expect(s.dimension).toBe(8);
  });

  it("upsert is idempotent on same fragmentId", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([mkRow("a", "v1")]);
    await s.upsert([mkRow("a", "v2")]);
    expect(await s.count()).toBe(1);
    const all = await s.scanAll();
    expect(all[0]!.content).toBe("v2");
  });

  it("delete removes selected ids", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([mkRow("a", "1"), mkRow("b", "2"), mkRow("c", "3")]);
    const n = await s.delete(["a", "c", "missing"]);
    expect(n).toBe(2);
    expect(await s.count()).toBe(1);
  });

  it("search returns top-K cosine results", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([
      mkRow("apple", "I love apples and bananas"),
      mkRow("car", "Cars and trucks are vehicles"),
      mkRow("apple2", "apples are sweet fruits"),
    ]);
    const q = e.embed("I love apples and bananas");
    const out = await s.search(q, { topK: 2 });
    expect(out).toHaveLength(2);
    expect(out[0]!.row.fragmentId).toBe("apple");
    expect(out[0]!.score).toBeGreaterThan(0.9);
  });

  it("search filters by memoryTypes", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([mkRow("a", "x"), { ...mkRow("b", "y"), memoryType: "long_term" as const }]);
    const out = await s.search(e.embed("x"), {
      topK: 10,
      memoryTypes: ["long_term"],
    });
    expect(out).toHaveLength(1);
    expect(out[0]!.row.fragmentId).toBe("b");
  });

  it("search filters by taskTags", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([mkRow("a", "tea time", ["tea"]), mkRow("b", "coffee break", ["coffee"])]);
    const out = await s.search(e.embed("drink"), {
      topK: 10,
      taskTags: ["tea"],
    });
    expect(out).toHaveLength(1);
    expect(out[0]!.row.fragmentId).toBe("a");
  });

  it("search filters by minImportance", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([
      { ...mkRow("a", "x"), importance: 0.2 },
      { ...mkRow("b", "y"), importance: 0.8 },
    ]);
    const out = await s.search(e.embed("x"), {
      topK: 10,
      minImportance: 0.5,
    });
    expect(out).toHaveLength(1);
    expect(out[0]!.row.fragmentId).toBe("b");
  });

  it("scanAll returns full set", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([mkRow("a", "1"), mkRow("b", "2"), mkRow("c", "3")]);
    const all = await s.scanAll();
    expect(all.map((r) => r.fragmentId).sort()).toEqual(["a", "b", "c"]);
  });

  it("searchText 按 BM25 词法命中并排序", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([
      mkRow("db", "postgres database index query optimization"),
      mkRow("ck", "recipe pasta tomato garlic"),
      mkRow("db2", "database transaction isolation"),
    ]);
    const out = await s.searchText("database index query", { topK: 10 });
    expect(out.length).toBeGreaterThan(0);
    // 词法上 "db" 命中 3 个 query 词 → 应排第一
    expect(out[0]!.row.fragmentId).toBe("db");
    // cooking 文档不含任何 query 词 → 不应出现
    expect(out.some((r) => r.row.fragmentId === "ck")).toBe(false);
  });

  it("searchText 尊重 memoryTypes / taskTags 过滤", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([
      mkRow("a", "database index", ["work"]),
      mkRow("b", "database index", ["home"]),
    ]);
    const out = await s.searchText("database index", { topK: 10, taskTags: ["work"] });
    expect(out).toHaveLength(1);
    expect(out[0]!.row.fragmentId).toBe("a");
  });

  it("searchText 空查询 / 无命中返回空", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([mkRow("a", "database index")]);
    expect(await s.searchText("", { topK: 5 })).toEqual([]);
    expect(await s.searchText("astronomy telescope", { topK: 5 })).toEqual([]);
  });

  it("supportsFts=true 且 ensureFtsIndex 幂等 no-op", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    expect(s.supportsFts).toBe(true);
    await s.ensureFtsIndex();
    await s.ensureFtsIndex();
  });

  it("rejects dimension mismatch", async () => {
    const s = new InMemoryVectorStore();
    await s.init();
    await s.upsert([mkRow("a", "1")]);
    const bad = mkRow("b", "2");
    bad.embedding = [1, 2, 3]; // wrong dim
    await expect(s.upsert([bad])).rejects.toThrow(/dim mismatch/);
  });
});
