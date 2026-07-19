/**
 * HybridRetriever 装配测试（RFC-003 方向 2 + Phase 3.3.B）。
 *
 * 覆盖：
 *  1. 默认 retrievalMode='vector'，env / opts 切到 'hybrid'
 *  2. retrieveHybrid 在多种 query 上能返回与 vector 不同（且合理）的排序
 *  3. /api/memory?mode=hybrid 走 hybrid 路径并暴露 BM25 分量
 *  4. /api/memory?mode=hybrid&rrf=true 启用 RRF 融合（components 带 rrf 字段）
 */
import { afterEach, describe, expect, it } from "vitest";
import {
  type ServerAgent,
  type ServerAgentOpts,
  assembleServerAgent as assembleServerAgentRuntime,
} from "../src/agent.js";
import { retrieveHybrid } from "../src/hybrid-retrieve.js";
import { buildApp } from "../src/routes.js";

const assembleServerAgent = (opts: ServerAgentOpts = {}): Promise<ServerAgent> =>
  assembleServerAgentRuntime({ embedProvider: "simple", ...opts });

const seed = async (agent: ServerAgent): Promise<void> => {
  // 一组刻意构造的"长尾"语料：vector 检索单看 embedding 不一定能区分，
  // BM25 通过精确词项匹配能把 keyword 命中拉前。
  await agent.persistentStore.addLongTermAsync("machine learning frameworks pytorch tensorflow", {
    taskTags: ["ml"],
    importance: 0.7,
  });
  await agent.persistentStore.addLongTermAsync("deep learning research paper survey", {
    taskTags: ["ml"],
    importance: 0.6,
  });
  await agent.persistentStore.addLongTermAsync("machine learning tutorials beginner", {
    taskTags: ["ml"],
    importance: 0.5,
  });
  await agent.persistentStore.addLongTermAsync("cooking recipe italian pasta", {
    taskTags: ["food"],
    importance: 0.4,
  });
  await agent.persistentStore.addLongTermAsync("travel guide kyoto temple", {
    taskTags: ["travel"],
    importance: 0.3,
  });
  await agent.persistentStore.awaitPendingWrites();
};

describe("agent.retrievalMode resolution", () => {
  it("默认 'vector'", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    expect(agent.retrievalMode).toBe("vector");
    await agent.close();
  });

  it("opts.retrievalMode='hybrid' 显式覆盖", async () => {
    const agent = await assembleServerAgent({
      llmProvider: "mock",
      retrievalMode: "hybrid",
    });
    expect(agent.retrievalMode).toBe("hybrid");
    await agent.close();
  });

  it("OPENINTJ_RETRIEVAL_MODE=hybrid 触发 hybrid 默认", async () => {
    const prev = process.env["OPENINTJ_RETRIEVAL_MODE"];
    process.env["OPENINTJ_RETRIEVAL_MODE"] = "hybrid";
    try {
      const agent = await assembleServerAgent({ llmProvider: "mock" });
      expect(agent.retrievalMode).toBe("hybrid");
      await agent.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_RETRIEVAL_MODE"];
      else process.env["OPENINTJ_RETRIEVAL_MODE"] = prev;
    }
  });

  it("opts 显式覆盖 env", async () => {
    const prev = process.env["OPENINTJ_RETRIEVAL_MODE"];
    process.env["OPENINTJ_RETRIEVAL_MODE"] = "hybrid";
    try {
      const agent = await assembleServerAgent({
        llmProvider: "mock",
        retrievalMode: "vector",
      });
      expect(agent.retrievalMode).toBe("vector");
      await agent.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_RETRIEVAL_MODE"];
      else process.env["OPENINTJ_RETRIEVAL_MODE"] = prev;
    }
  });
});

describe("retrieveHybrid (direct API)", () => {
  let agent: ServerAgent;
  afterEach(async () => {
    await agent.close();
  });

  it("query 'machine learning' 让两条 ML 长尾的 BM25 分量为正", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    // 让 topK 覆盖全部以避免 SimpleEmbedder 的 cosine 噪声把 BM25 命中的挤出 topK
    const hits = await retrieveHybrid(agent, "machine learning", { topK: 5 });
    const bm25Positive = hits.filter((h) => h.components.bm25 > 0);
    // 两条精确含 "machine learning" 的 doc + 一条只含 "learning" 的 doc，共 3 条 BM25 > 0
    expect(bm25Positive.length).toBe(3);
    // 两条精确含 "machine learning" 的 BM25 > 只含 "learning" 的
    const fullMatch = hits.filter((h) => h.doc.text.includes("machine learning"));
    const partialMatch = hits.filter(
      (h) => !h.doc.text.includes("machine learning") && h.components.bm25 > 0,
    );
    expect(fullMatch.length).toBe(2);
    expect(partialMatch.length).toBe(1);
    expect(Math.min(...fullMatch.map((h) => h.components.bm25))).toBeGreaterThan(
      partialMatch[0]!.components.bm25,
    );
  });

  it("BM25 分量对未命中关键词的 doc 为 0", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    // query 用一个绝对不会出现的词，BM25 必然全 0
    const hits = await retrieveHybrid(agent, "zzzzzzz_no_match", { topK: 5 });
    expect(hits.every((h) => h.components.bm25 === 0)).toBe(true);
  });

  it("memoryTypes 过滤生效", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    const hits = await retrieveHybrid(agent, "learning", {
      topK: 5,
      memoryTypes: ["long_term"],
    });
    expect(hits.every((h) => h.doc.metadata.memoryType === "long_term")).toBe(true);
  });

  it("taskTags 过滤生效", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    const hits = await retrieveHybrid(agent, "anything", {
      topK: 10,
      taskTags: ["food"],
    });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.every((h) => h.doc.metadata.taskTags.includes("food"))).toBe(true);
  });

  it("空 store 返回空数组", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    const hits = await retrieveHybrid(agent, "anything", { topK: 5 });
    expect(hits).toEqual([]);
  });

  it("useRRF=true 时 components.rrf 存在", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    const hits = await retrieveHybrid(agent, "learning", {
      topK: 3,
      config: { useRRF: true },
    });
    expect(hits[0]!.components.rrf).toBeDefined();
    expect(typeof hits[0]!.components.rrf).toBe("number");
  });

  // ---------- #10：LanceDB 原生 FTS 路径（mock 模式下由 InMemoryVectorStore 的 FTS 覆盖） ----------

  it("useLanceFts=true 走 store FTS + 向量 RRF 融合，命中精确关键词", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    const hits = await retrieveHybrid(agent, "pytorch tensorflow", {
      topK: 5,
      useLanceFts: true,
    });
    expect(hits.length).toBeGreaterThan(0);
    // "pytorch tensorflow" 是首条 ML doc 独有词 → FTS 路应把它召回并靠前
    expect(hits.some((h) => h.doc.text.includes("pytorch tensorflow"))).toBe(true);
    // RRF 分记进 components.rrf
    expect(typeof hits[0]!.components.rrf).toBe("number");
  });

  it("useLanceFts=true 尊重 taskTags 过滤", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    const hits = await retrieveHybrid(agent, "italian pasta", {
      topK: 10,
      useLanceFts: true,
      taskTags: ["food"],
    });
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.every((h) => h.doc.metadata.taskTags.includes("food"))).toBe(true);
  });

  it("OPENINTJ_LANCE_FTS=1 env 触发 FTS 路径", async () => {
    const prev = process.env["OPENINTJ_LANCE_FTS"];
    process.env["OPENINTJ_LANCE_FTS"] = "1";
    try {
      agent = await assembleServerAgent({ llmProvider: "mock" });
      await seed(agent);
      const hits = await retrieveHybrid(agent, "machine learning", { topK: 5 });
      expect(hits.length).toBeGreaterThan(0);
      expect(typeof hits[0]!.components.rrf).toBe("number");
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_LANCE_FTS"];
      else process.env["OPENINTJ_LANCE_FTS"] = prev;
    }
  });
});

describe("/api/memory mode switch", () => {
  let agent: ServerAgent;
  afterEach(async () => {
    await agent.close();
  });

  it("不传 mode 时按 agent.retrievalMode 走（默认 vector）", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    const res = await buildApp(agent).request("/api/memory?q=machine&topK=3");
    expect(res.status).toBe(200);
    const body = (await res.json()) as { mode: string; results: unknown[] };
    expect(body.mode).toBe("vector");
    expect(body.results.length).toBeGreaterThan(0);
  });

  it("?mode=hybrid 切到 BM25 + cosine 路径，分量包含 bm25", async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
    await seed(agent);
    const res = await buildApp(agent).request("/api/memory?q=machine&topK=3&mode=hybrid");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      mode: string;
      results: Array<{ components: { vector: number; bm25: number; rrf?: number } }>;
    };
    expect(body.mode).toBe("hybrid");
    expect(body.results[0]!.components.bm25).toBeDefined();
    expect(body.results[0]!.components.rrf).toBeUndefined();
  });

  it("?mode=hybrid&rrf=true 启用 RRF 融合", async () => {
    agent = await assembleServerAgent({
      llmProvider: "mock",
      retrievalMode: "hybrid",
    });
    await seed(agent);
    const res = await buildApp(agent).request("/api/memory?q=machine&topK=3&rrf=true");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      mode: string;
      results: Array<{ components: { rrf?: number } }>;
    };
    expect(body.mode).toBe("hybrid");
    expect(body.results[0]!.components.rrf).toBeDefined();
  });

  it("/api/status 暴露 retrievalMode", async () => {
    agent = await assembleServerAgent({
      llmProvider: "mock",
      retrievalMode: "hybrid",
    });
    const res = await buildApp(agent).request("/api/status");
    const body = (await res.json()) as { retrievalMode: string };
    expect(body.retrievalMode).toBe("hybrid");
  });
});
