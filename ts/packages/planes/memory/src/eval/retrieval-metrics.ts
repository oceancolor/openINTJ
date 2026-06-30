/**
 * 检索质量评测指标（nDCG / recall / precision / MRR）。
 *
 * 用途：
 *  - 给记忆检索（MemoryRetriever / HybridRetriever）一个可量化、可回归守护的质量基线。
 *  - 支撑 "simple vs xenova vs ollama 嵌入在固定语料上的 nDCG 对比"（RFC roadmap #3）。
 *
 * 约定：relevance 用 graded gain（0 = 不相关，1/2/3 = 相关度递增）。
 */

/** Discounted Cumulative Gain。 */
export const dcg = (gains: readonly number[]): number =>
  gains.reduce((s, g, i) => s + (2 ** g - 1) / Math.log2(i + 2), 0);

/** nDCG@k：归一化到理想排序（IDCG）。无相关文档时返回 0。 */
export const ndcgAtK = (
  rankedIds: readonly string[],
  relevance: ReadonlyMap<string, number>,
  k: number,
): number => {
  const topGains = rankedIds.slice(0, k).map((id) => relevance.get(id) ?? 0);
  const idealGains = [...relevance.values()]
    .filter((g) => g > 0)
    .sort((a, b) => b - a)
    .slice(0, k);
  const idcg = dcg(idealGains);
  return idcg === 0 ? 0 : dcg(topGains) / idcg;
};

/** recall@k：top-k 命中的相关文档占全部相关文档的比例。 */
export const recallAtK = (
  rankedIds: readonly string[],
  relevantIds: ReadonlySet<string>,
  k: number,
): number => {
  if (relevantIds.size === 0) return 0;
  const topk = new Set(rankedIds.slice(0, k));
  let hit = 0;
  for (const id of relevantIds) if (topk.has(id)) hit++;
  return hit / relevantIds.size;
};

/** precision@k：top-k 中相关文档比例。 */
export const precisionAtK = (
  rankedIds: readonly string[],
  relevantIds: ReadonlySet<string>,
  k: number,
): number => {
  const top = rankedIds.slice(0, k);
  if (top.length === 0) return 0;
  let hit = 0;
  for (const id of top) if (relevantIds.has(id)) hit++;
  return hit / top.length;
};

/** Mean Reciprocal Rank：第一个相关文档的倒数排名。 */
export const reciprocalRank = (
  rankedIds: readonly string[],
  relevantIds: ReadonlySet<string>,
): number => {
  for (let i = 0; i < rankedIds.length; i++) {
    if (relevantIds.has(rankedIds[i] as string)) return 1 / (i + 1);
  }
  return 0;
};

export interface EvalCase {
  query: string;
  /** docId → graded relevance（>0 视为相关）。 */
  relevant: ReadonlyMap<string, number>;
}

export interface EvalSummary {
  ndcg: number;
  recall: number;
  precision: number;
  mrr: number;
  /** query 数。 */
  n: number;
  k: number;
}

/**
 * 在一组 query 上评测一个排序器，返回各指标的宏平均。
 * @param rank 给定 query 返回排好序的 docId 列表（best-first）。
 */
export const evaluateRanker = (
  cases: readonly EvalCase[],
  rank: (query: string) => readonly string[],
  k: number,
): EvalSummary => {
  if (cases.length === 0) return { ndcg: 0, recall: 0, precision: 0, mrr: 0, n: 0, k };
  let ndcg = 0;
  let recall = 0;
  let precision = 0;
  let mrr = 0;
  for (const c of cases) {
    const ranked = rank(c.query);
    const relevantIds = new Set([...c.relevant.entries()].filter(([, g]) => g > 0).map(([id]) => id));
    ndcg += ndcgAtK(ranked, c.relevant, k);
    recall += recallAtK(ranked, relevantIds, k);
    precision += precisionAtK(ranked, relevantIds, k);
    mrr += reciprocalRank(ranked, relevantIds);
  }
  const n = cases.length;
  return { ndcg: ndcg / n, recall: recall / n, precision: precision / n, mrr: mrr / n, n, k };
};
