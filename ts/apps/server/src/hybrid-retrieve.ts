import {
  type VectorSearchOpts,
  type VectorSearchResult,
  hybridVectorSearch,
} from "@openintj/storage-lance";
import type { HybridConfig, MemoryHybridHit } from "@openintj/taskpool";
import type { ServerAgent } from "./agent.js";

/**
 * RFC-003 方向 2：HybridRetriever 装配。
 *
 * 两条混合检索路径：
 *  1. **内存 BM25**（默认）：复用 session 级 `MemoryHybridIndex`（开局 seed + 订阅 change-feed
 *     增量维护），每次查询在内存里算 BM25 + cosine 融合。中等规模够用、零外部索引。
 *  2. **LanceDB 原生 FTS**（roadmap #10，opt-in `OPENINTJ_LANCE_FTS=1` 或 `useLanceFts`）：
 *     大规模 fragment 时把词法检索下推到 LanceDB 原生 FTS（BM25 索引），与向量检索各出一榜、
 *     RRF 融合，避免每查询扫全表。持久层 InMemory / LanceDB 均实现 `searchText`，不支持则自动
 *     降级为纯向量检索。
 */
export type HybridMemoryHit = MemoryHybridHit;

export interface RetrieveHybridOpts {
  topK?: number;
  /** 按查询覆盖融合配置（alpha/beta/useRRF/...）。仅内存路径生效。 */
  config?: Partial<HybridConfig>;
  /** 仅检索这些 memoryType。 */
  memoryTypes?: readonly string[];
  /** 任意一个标签命中即保留。 */
  taskTags?: readonly string[];
  /** 显式传入的 query embedding；不传则用 store.embedder 生成。 */
  queryEmbedding?: readonly number[];
  /**
   * 走 LanceDB 原生 FTS 路径（#10）。默认读 env `OPENINTJ_LANCE_FTS=1`。
   * 显式传值优先于 env。
   */
  useLanceFts?: boolean;
}

const lanceFtsEnabled = (opts: RetrieveHybridOpts): boolean =>
  opts.useLanceFts ?? process.env["OPENINTJ_LANCE_FTS"] === "1";

/** VectorSearchResult → MemoryHybridHit（RRF 分记进 components.rrf）。 */
const toHybridHit = (r: VectorSearchResult): HybridMemoryHit => ({
  doc: {
    id: r.row.fragmentId,
    text: r.row.content,
    vector: r.row.embedding,
    metadata: {
      memoryType: r.row.memoryType,
      taskTags: r.row.taskTags,
      importance: r.row.importance,
    },
  },
  score: r.score,
  components: { vector: 0, bm25: 0, rrf: r.score },
});

export const retrieveHybrid = async (
  agent: ServerAgent,
  query: string,
  opts: RetrieveHybridOpts = {},
): Promise<HybridMemoryHit[]> => {
  if (agent.hybridIndex.size === 0) return [];

  let qVec: readonly number[] | undefined = opts.queryEmbedding;
  if (qVec === undefined) {
    const r = agent.memory.store.embedder.embed(query);
    qVec = r instanceof Promise ? await r : r;
  }

  const topK = opts.topK ?? 10;

  // #10：大规模走 LanceDB 原生 FTS + 向量 RRF 融合（store 不支持时自动降级为纯向量）。
  if (lanceFtsEnabled(opts)) {
    const searchOpts: VectorSearchOpts = {
      topK,
      ...(opts.memoryTypes
        ? { memoryTypes: opts.memoryTypes as NonNullable<VectorSearchOpts["memoryTypes"]> }
        : {}),
      ...(opts.taskTags ? { taskTags: opts.taskTags } : {}),
    };
    const fused = await hybridVectorSearch(agent.persistentStore.vectorStore, {
      ...searchOpts,
      query,
      queryEmbedding: qVec ?? [],
    });
    return fused.map(toHybridHit);
  }

  return agent.hybridIndex.search(query, qVec, {
    topK,
    ...(opts.config ? { config: opts.config } : {}),
    ...(opts.memoryTypes ? { memoryTypes: opts.memoryTypes } : {}),
    ...(opts.taskTags ? { taskTags: opts.taskTags } : {}),
  });
};
