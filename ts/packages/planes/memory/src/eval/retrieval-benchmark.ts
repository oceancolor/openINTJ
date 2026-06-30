/**
 * 可插拔的检索召回基准：给任意 {@link EmbeddingProvider} 在固定主题语料上量化
 * nDCG / recall / precision / MRR。用于 "simple vs xenova vs ollama" 三方对比（RFC roadmap #3）。
 *
 * 设计：
 *  - 语料 / query 固定（database / cooking / astronomy 三主题），相关性按主题判定。
 *  - 全异步：兼容 sync（simple）与 async（xenova/ollama）embedder；维度自动探测。
 *  - 纯函数式产出 {@link BenchmarkResult}，调用方决定打印 / 断言 / 对比。
 */
import type { EmbeddingProvider } from "@openintj/core";
import { MemoryRetriever } from "../retriever.js";
import { MemoryStore } from "../store.js";
import { type EvalCase, type EvalSummary, evaluateRanker } from "./retrieval-metrics.js";

export interface BenchmarkDoc {
  id: string;
  topic: string;
  text: string;
}

export const BENCHMARK_CORPUS: BenchmarkDoc[] = [
  { id: "db1", topic: "database", text: "postgres database indexing query optimization performance" },
  { id: "db2", topic: "database", text: "sql database transaction isolation level acid" },
  { id: "db3", topic: "database", text: "database schema migration rollback version control" },
  { id: "db4", topic: "database", text: "query planner join index scan database tuning" },
  { id: "ck1", topic: "cooking", text: "recipe pasta tomato sauce garlic basil olive oil" },
  { id: "ck2", topic: "cooking", text: "baking bread flour yeast oven dough recipe" },
  { id: "ck3", topic: "cooking", text: "grill steak salt pepper butter cooking medium rare" },
  { id: "ck4", topic: "cooking", text: "soup stock vegetable simmer recipe broth garlic" },
  { id: "as1", topic: "astronomy", text: "telescope galaxy star observation night sky nebula" },
  { id: "as2", topic: "astronomy", text: "planet orbit solar system gravity astronomy moon" },
  { id: "as3", topic: "astronomy", text: "black hole event horizon spacetime gravity star" },
  { id: "as4", topic: "astronomy", text: "comet asteroid meteor observation telescope sky" },
];

export const BENCHMARK_QUERIES: { query: string; topic: string }[] = [
  { query: "database query optimization index", topic: "database" },
  { query: "sql transaction database tuning", topic: "database" },
  { query: "recipe garlic tomato cooking", topic: "cooking" },
  { query: "baking bread recipe oven", topic: "cooking" },
  { query: "telescope galaxy star sky observation", topic: "astronomy" },
  { query: "planet orbit gravity solar system", topic: "astronomy" },
];

export interface BenchmarkResult {
  /** embedder.name。 */
  embedder: string;
  /** 探测到的向量维度。 */
  dimension: number;
  summary: EvalSummary;
}

/**
 * 在固定语料上评测一个 embedder 的默认检索路径（MemoryStore + MemoryRetriever）。
 * 维度由对 embedder 探测一次得到，兼容任意维度的 sync/async embedder。
 */
export const benchmarkRetrieval = async (
  embedder: EmbeddingProvider,
  opts: { k?: number } = {},
): Promise<BenchmarkResult> => {
  const k = opts.k ?? 4;
  // 探测维度（xenova 等 async embedder 的 dimension 在首次 embed 后才确定）。
  const probe = await Promise.resolve(embedder.embed("dimension probe"));
  const embeddingDim = probe.length;

  const store = new MemoryStore({ embeddingDim }, { embedder });
  const fragIdByDoc = new Map<string, string>();
  for (const doc of BENCHMARK_CORPUS) {
    const frag = await store.addLongTermAsync(doc.text);
    fragIdByDoc.set(doc.id, frag.fragmentId);
  }

  const retriever = new MemoryRetriever(store, undefined, { embedder });
  const rankingByQuery = new Map<string, string[]>();
  for (const q of BENCHMARK_QUERIES) {
    const ranked = await retriever.retrieveAsync(q.query, { topK: BENCHMARK_CORPUS.length });
    rankingByQuery.set(
      q.query,
      ranked.map((r) => r.fragment.fragmentId),
    );
  }

  const cases: EvalCase[] = BENCHMARK_QUERIES.map((q) => {
    const relevant = new Map<string, number>();
    for (const doc of BENCHMARK_CORPUS) {
      if (doc.topic === q.topic) relevant.set(fragIdByDoc.get(doc.id) as string, 1);
    }
    return { query: q.query, relevant };
  });

  const summary = evaluateRanker(cases, (query) => rankingByQuery.get(query) ?? [], k);
  return { embedder: embedder.name, dimension: embeddingDim, summary };
};

/** 把 {@link BenchmarkResult} 格式化成单行评分表（CI 日志 / 对比用）。 */
export const formatBenchmarkRow = (r: BenchmarkResult): string =>
  `[retrieval-benchmark] ${r.embedder}@dim${r.dimension}  ` +
  `nDCG@${r.summary.k}=${r.summary.ndcg.toFixed(3)}  ` +
  `recall@${r.summary.k}=${r.summary.recall.toFixed(3)}  ` +
  `precision@${r.summary.k}=${r.summary.precision.toFixed(3)}  ` +
  `MRR=${r.summary.mrr.toFixed(3)}  (n=${r.summary.n})`;
