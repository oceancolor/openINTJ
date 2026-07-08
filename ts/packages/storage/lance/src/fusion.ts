import type { VectorSearchOpts, VectorSearchResult, VectorStore } from "./types.js";

/**
 * 混合检索融合（roadmap #10）。
 *
 * 向量检索（`store.search`）与原生全文检索（`store.searchText`）各自产出一份按相关度
 * 排序的候选列表，用 **Reciprocal Rank Fusion (RRF)** 融合成一份榜单。RRF 只依赖每个文档
 * 在各列表里的**名次**，不依赖分数量纲——因此天然适配「cosine 分」与「BM25 分」这类不可直接
 * 相加的异构分数（HybridRetriever 的加权和路径需要先 min-max 归一，RRF 不需要）。
 */

export interface RrfFuseOpts {
  /** RRF 常数 k（越大越平滑，弱化头部名次差异）。默认 60（业界常用）。 */
  rrfK?: number;
  /** 融合后截断的返回条数。默认取各列表长度之和。 */
  topK?: number;
}

/**
 * 把多份已排序的检索结果按 RRF 融合。
 * - 每份列表按当前顺序视为名次（第 1 名 rank=1）。
 * - 同一 `fragmentId` 的 RRF 分数 = Σ 1/(rrfK + rank)。
 * - 保留首次出现的 row（各列表 row 内容一致，取谁都行）。
 * - 输出的 `score` 为 RRF 分，`distance` 置 0（RRF 无距离语义）。
 */
export const rrfFuse = (
  lists: readonly (readonly VectorSearchResult[])[],
  opts: RrfFuseOpts = {},
): VectorSearchResult[] => {
  const rrfK = opts.rrfK ?? 60;
  const acc = new Map<string, { row: VectorSearchResult["row"]; score: number }>();
  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const hit = list[rank];
      if (!hit) continue;
      const id = hit.row.fragmentId;
      const contribution = 1 / (rrfK + (rank + 1));
      const prev = acc.get(id);
      if (prev) prev.score += contribution;
      else acc.set(id, { row: hit.row, score: contribution });
    }
  }
  const fused: VectorSearchResult[] = [...acc.values()]
    .map((e) => ({ row: e.row, score: e.score, distance: 0 }))
    .sort((a, b) => b.score - a.score);
  const topK = opts.topK ?? fused.length;
  return topK >= fused.length ? fused : fused.slice(0, Math.max(0, topK));
};

export interface HybridVectorSearchOpts extends VectorSearchOpts {
  /** 文本查询（走 `store.searchText`）。 */
  query: string;
  /** 查询向量（走 `store.search`）。 */
  queryEmbedding: readonly number[];
  /** RRF 常数 k。默认 60。 */
  rrfK?: number;
  /**
   * 每路召回的候选数（融合前）。默认 `max(topK * 4, topK)`——多召回再融合截断，
   * 让另一路里排名靠后的强相关项有机会靠 RRF 冒头。
   */
  candidateK?: number;
}

/**
 * 存储层混合检索：向量 + 原生 FTS，RRF 融合。
 *
 * - `store.searchText` 缺失或返回空（未建 FTS 索引 / 不支持）时**自动降级**为纯向量检索，
 *   保证任何 VectorStore 都能安全调用。
 * - 过滤语义（memoryType / taskTags / minImportance）两路一致，由各自 search 内部处理。
 */
export const hybridVectorSearch = async (
  store: VectorStore,
  opts: HybridVectorSearchOpts,
): Promise<VectorSearchResult[]> => {
  const { query, queryEmbedding, rrfK, candidateK, ...searchOpts } = opts;
  const fetchK = Math.max(candidateK ?? searchOpts.topK * 4, searchOpts.topK);
  const perPathOpts: VectorSearchOpts = { ...searchOpts, topK: fetchK };

  const vectorHits = await store.search(queryEmbedding, perPathOpts);
  const textHits =
    typeof store.searchText === "function"
      ? await store.searchText(query, perPathOpts).catch(() => [])
      : [];

  // 只有一路有结果时无需融合，直接截断（省一次 map/sort）。
  if (textHits.length === 0) return vectorHits.slice(0, searchOpts.topK);
  if (vectorHits.length === 0) return textHits.slice(0, searchOpts.topK);

  const fuseOpts: RrfFuseOpts = { topK: searchOpts.topK, ...(rrfK !== undefined ? { rrfK } : {}) };
  return rrfFuse([vectorHits, textHits], fuseOpts);
};
