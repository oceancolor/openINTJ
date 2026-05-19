import { type EmbeddingProvider, SimpleEmbedder, TaskType } from "@openintj/core";
import { describe, expect, it, vi } from "vitest";
import { ContextEngine, MemoryPlane, MemoryRetriever, MemoryStore } from "../src/index.js";

class StubAsyncEmbedder implements EmbeddingProvider {
  readonly name = "stub-async";
  readonly dimension = 8;
  readonly calls: string[] = [];

  embed(text: string): Promise<number[]> {
    this.calls.push(text);
    // 简单确定性：取每个 char code 求 hash
    const out = new Array(8).fill(0);
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      out[code % 8] += 1;
    }
    let n = 0;
    for (const v of out) n += v * v;
    n = Math.sqrt(n) || 1;
    return Promise.resolve(out.map((v) => v / n));
  }

  async embedBatch(texts: readonly string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (const t of texts) out.push(await this.embed(t));
    return out;
  }
}

describe("MemoryStore embedder injection", () => {
  it("uses SimpleEmbedder by default and exposes its dimension", () => {
    const s = new MemoryStore();
    expect(s.embedder.name).toBe("simple-sha256");
    expect(s.embedder.dimension).toBe(64);
    const f = s.addShortTerm("hello");
    expect(f.embedding).toHaveLength(64);
  });

  it("respects custom embedder dimension", () => {
    const e = new SimpleEmbedder(32);
    const s = new MemoryStore({ embeddingDim: 32 }, { embedder: e });
    expect(s.config.embeddingDim).toBe(32);
    const f = s.addShortTerm("x");
    expect(f.embedding).toHaveLength(32);
  });

  it("sync addShortTerm throws when embedder is async", () => {
    const e = new StubAsyncEmbedder();
    const s = new MemoryStore({ embeddingDim: 8 }, { embedder: e });
    expect(() => s.addShortTerm("hello")).toThrow(/async/);
  });

  it("addShortTermAsync works with async embedder", async () => {
    const e = new StubAsyncEmbedder();
    const s = new MemoryStore({ embeddingDim: 8 }, { embedder: e });
    const f = await s.addShortTermAsync("hello");
    expect(f.embedding).toHaveLength(8);
    expect(e.calls).toContain("hello");
  });
});

describe("MemoryRetriever async retrieval", () => {
  it("retrieveAsync auto-embeds query with async embedder", async () => {
    const e = new StubAsyncEmbedder();
    const s = new MemoryStore({ embeddingDim: 8 }, { embedder: e });
    await s.addShortTermAsync("apple banana cherry");
    await s.addShortTermAsync("dog elephant fox");
    const r = new MemoryRetriever(s);
    const out = await r.retrieveAsync("apple");
    expect(out.length).toBeGreaterThan(0);
    expect(out[0]!.fragment.content).toContain("apple");
  });

  it("retrieve (sync) throws with async embedder unless queryEmbedding given", async () => {
    const e = new StubAsyncEmbedder();
    const s = new MemoryStore({ embeddingDim: 8 }, { embedder: e });
    await s.addShortTermAsync("hi");
    const r = new MemoryRetriever(s);
    expect(() => r.retrieve("hi")).toThrow(/async/);
    // pre-embedded path works
    const qEmb = await e.embed("hi");
    const out = r.retrieve("hi", { queryEmbedding: qEmb });
    expect(out.length).toBe(1);
  });
});

describe("ContextEngine + async embedder", () => {
  it("build() works with async embedder via retrieveAsync", async () => {
    const e = new StubAsyncEmbedder();
    const plane = new MemoryPlane({
      embedder: e,
      storeConfig: { embeddingDim: 8 },
    });
    await plane.store.addShortTermAsync("用户喜欢喝绿茶");
    const engine = new ContextEngine({
      store: plane.store,
      embedder: e,
    });
    const win = await engine.build({
      query: "喝什么茶",
      history: [],
      taskType: TaskType.GENERAL_CHAT,
      systemPrompt: "你是助手",
    });
    expect(win.memoryFragments.length).toBeGreaterThan(0);
  });
});

describe("EmbeddingProvider contract", () => {
  it("SimpleEmbedder.embedBatch is consistent with embed", () => {
    const e = new SimpleEmbedder(16);
    const single = e.embed("alpha");
    const batch = e.embedBatch(["alpha", "beta"]);
    expect(batch[0]).toEqual(single);
    expect(batch[1]!.length).toBe(16);
  });

  it("custom EmbeddingProvider can be a vi.fn-driven fake", async () => {
    const fake: EmbeddingProvider = {
      name: "fake",
      dimension: 4,
      embed: vi.fn(async () => [1, 0, 0, 0]),
      embedBatch: vi.fn(async (texts) => texts.map(() => [1, 0, 0, 0])),
    };
    const s = new MemoryStore({ embeddingDim: 4 }, { embedder: fake });
    const f = await s.addShortTermAsync("any");
    expect(f.embedding).toEqual([1, 0, 0, 0]);
    expect(fake.embed).toHaveBeenCalledOnce();
  });
});
