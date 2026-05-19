import { SimpleEmbedder } from "@openintj/core";
import { InMemoryVectorStore } from "@openintj/storage-lance";
import { InMemoryMetadataStore } from "@openintj/storage-sqlite";
import { describe, expect, it } from "vitest";
import { PersistentMemoryStore } from "../src/index.js";

const mkStore = (opts?: { hydrate?: boolean }) =>
  new PersistentMemoryStore({
    vectorStore: new InMemoryVectorStore(),
    metadataStore: new InMemoryMetadataStore(),
    embedder: new SimpleEmbedder(8),
    storeConfig: { embeddingDim: 8, maxShortTerm: 3 },
    hydrateOnInit: opts?.hydrate,
  });

describe("PersistentMemoryStore", () => {
  it("init runs migrate and seeds empty state", async () => {
    const s = mkStore();
    await s.init();
    expect(s.totalCount).toBe(0);
    expect(await s.vectorStore.count()).toBe(0);
  });

  it("dual-writes addShortTerm to vector + metadata stores", async () => {
    const s = mkStore();
    await s.init();
    s.addShortTerm("hello world");
    s.addShortTerm("another memory");
    await s.awaitPendingWrites();

    expect(await s.vectorStore.count()).toBe(2);
    const list = await s.metadataStore.listFragmentMeta();
    expect(list).toHaveLength(2);
  });

  it("hydrate restores fragments after re-instantiation (round-trip)", async () => {
    const vec = new InMemoryVectorStore();
    const meta = new InMemoryMetadataStore();
    const s1 = new PersistentMemoryStore({
      vectorStore: vec,
      metadataStore: meta,
      embedder: new SimpleEmbedder(8),
      storeConfig: { embeddingDim: 8 },
    });
    await s1.init();
    s1.addShortTerm("alpha");
    s1.addShortTerm("beta");
    s1.addLongTerm("gamma persistent");
    await s1.awaitPendingWrites();
    const before = s1.all.map((f) => f.fragmentId).sort();

    // 模拟重启：复用同一 vector + metadata store（持久化层）
    const s2 = new PersistentMemoryStore({
      vectorStore: vec,
      metadataStore: meta,
      embedder: new SimpleEmbedder(8),
      storeConfig: { embeddingDim: 8 },
      hydrateOnInit: true,
    });
    await s2.init();
    const after = s2.all.map((f) => f.fragmentId).sort();
    expect(after).toEqual(before);
    expect(s2.shortTerm.length + s2.longTerm.length).toBe(3);
  });

  it("vectorSearch uses persistent layer (not memory)", async () => {
    const s = mkStore();
    await s.init();
    await s.addShortTermAsync("apple banana cherry");
    await s.addShortTermAsync("dog elephant fox");
    const e = new SimpleEmbedder(8);
    const out = await s.vectorSearch(e.embed("apple"), { topK: 1 });
    expect(out).toHaveLength(1);
    expect(out[0]!.row.content).toContain("apple");
  });

  it("remove() deletes from both layers", async () => {
    const s = mkStore();
    await s.init();
    const f = s.addShortTerm("removable");
    await s.awaitPendingWrites();
    expect(s.remove(f.fragmentId)).toBe(true);
    await s.awaitPendingWrites();
    expect(await s.vectorStore.count()).toBe(0);
    expect(await s.metadataStore.getFragmentMeta(f.fragmentId)).toBeUndefined();
  });

  it("short-term overflow promotes to long_term and persists new memoryType", async () => {
    const s = mkStore(); // maxShortTerm=3
    await s.init();
    s.addShortTerm("m1");
    s.addShortTerm("m2");
    s.addShortTerm("m3");
    s.addShortTerm("m4"); // m1 evicted to long_term
    await s.awaitPendingWrites();
    // sync to push current memoryType state
    await s.sync();
    const longTerm = await s.metadataStore.listFragmentMeta({
      memoryType: "long_term",
    });
    const m1 = longTerm.find((r) => r);
    expect(m1).toBeDefined();
  });

  it("sync() rewrites all in-memory state to persistent layer", async () => {
    const s = mkStore();
    await s.init();
    s.addShortTerm("a");
    s.addShortTerm("b");
    await s.awaitPendingWrites();
    const r = await s.sync();
    expect(r.fragments).toBe(2);
    expect(await s.vectorStore.count()).toBe(2);
  });

  it("close cleans up", async () => {
    const s = mkStore();
    await s.init();
    s.addShortTerm("x");
    await s.close();
    expect(await s.vectorStore.count()).toBe(0); // in-memory store cleared
  });

  it("metadata is preserved through round-trip", async () => {
    const vec = new InMemoryVectorStore();
    const meta = new InMemoryMetadataStore();
    const s1 = new PersistentMemoryStore({
      vectorStore: vec,
      metadataStore: meta,
      embedder: new SimpleEmbedder(8),
      storeConfig: { embeddingDim: 8 },
    });
    await s1.init();
    s1.addLongTerm("rich content", {
      importance: 0.9,
      taskTags: ["custom"],
      metadata: { source: "test", id: 42 },
      summaries: { 1: "rich", 2: "r" },
    });
    await s1.awaitPendingWrites();

    const s2 = new PersistentMemoryStore({
      vectorStore: vec,
      metadataStore: meta,
      embedder: new SimpleEmbedder(8),
      storeConfig: { embeddingDim: 8 },
    });
    await s2.init();
    const f = s2.longTerm[0]!;
    expect(f.importance).toBe(0.9);
    expect(f.taskTags).toEqual(["custom"]);
    expect(f.metadata.source).toBe("test");
    expect(f.metadata.id).toBe(42);
    expect(f.summaries[1]).toBe("rich");
  });
});

describe("MetadataStore migrate", () => {
  it("migrate is idempotent", async () => {
    const m = new InMemoryMetadataStore();
    await m.init();
    const r1 = await m.migrate();
    const r2 = await m.migrate();
    expect(r1.to).toBe(1);
    expect(r2.to).toBe(1);
  });
});
