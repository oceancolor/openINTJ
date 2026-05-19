/**
 * HybridRetriever —— 向量 + BM25 + 重排合一的混合检索器。
 *
 * 这是 RFC-003 任务池方向的关键组件，解决"检索→利用"瓶颈：
 *  - vector: cosine similarity
 *  - BM25: 经典 TF-IDF + 长度归一化
 *  - rerank: 简单的加权融合（生产可换 cross-encoder）
 *
 * 输入：query + 候选 documents（带 vector），输出按融合分排序。
 */

export interface HybridDoc {
  id: string;
  text: string;
  vector?: readonly number[];
  /** 任意附加字段透传到结果。 */
  metadata?: Record<string, unknown>;
}

export interface HybridScored<D extends HybridDoc = HybridDoc> {
  doc: D;
  score: number;
  components: { vector: number; bm25: number; rrf?: number };
}

export interface HybridConfig {
  /** vector 权重，默认 0.6。 */
  alpha: number;
  /** BM25 权重，默认 0.4。 */
  beta: number;
  /** BM25 k1（saturation），默认 1.5。 */
  k1: number;
  /** BM25 b（length normalization），默认 0.75。 */
  b: number;
  /** 用 RRF（Reciprocal Rank Fusion）替代加权和，默认 false。 */
  useRRF: boolean;
  /** RRF k 参数（默认 60）。 */
  rrfK: number;
}

export const DEFAULT_HYBRID_CONFIG: HybridConfig = {
  alpha: 0.6,
  beta: 0.4,
  k1: 1.5,
  b: 0.75,
  useRRF: false,
  rrfK: 60,
};

const tokenize = (text: string): string[] =>
  text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);

const cosine = (a: readonly number[], b: readonly number[]): number => {
  if (a.length === 0 || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0;
    const bi = b[i] ?? 0;
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

export interface HybridRetrieverOpts {
  config?: Partial<HybridConfig>;
}

/**
 * 构造时遍历语料计算 BM25 必要统计量；query 时返回融合排名。
 */
export class HybridRetriever<D extends HybridDoc = HybridDoc> {
  readonly config: HybridConfig;
  private docs: D[] = [];
  private docTokens: string[][] = [];
  private avgDocLen = 0;
  private docFreq = new Map<string, number>();

  constructor(opts: HybridRetrieverOpts = {}) {
    this.config = { ...DEFAULT_HYBRID_CONFIG, ...(opts.config ?? {}) };
  }

  index(docs: readonly D[]): void {
    this.docs = [...docs];
    this.docTokens = this.docs.map((d) => tokenize(d.text));
    let total = 0;
    this.docFreq.clear();
    for (const tokens of this.docTokens) {
      total += tokens.length;
      const seen = new Set<string>();
      for (const t of tokens) {
        if (seen.has(t)) continue;
        seen.add(t);
        this.docFreq.set(t, (this.docFreq.get(t) ?? 0) + 1);
      }
    }
    this.avgDocLen = this.docs.length > 0 ? total / this.docs.length : 0;
  }

  search(query: string, queryVector: readonly number[] | undefined, topK = 10): HybridScored<D>[] {
    if (this.docs.length === 0) return [];

    const qTokens = tokenize(query);
    const N = this.docs.length;

    // BM25 + cosine 双路打分
    const vectorScores: number[] = [];
    const bm25Scores: number[] = [];
    for (let i = 0; i < this.docs.length; i++) {
      const doc = this.docs[i]!;
      const tokens = this.docTokens[i] ?? [];
      // vector
      const vScore = queryVector && doc.vector ? cosine(queryVector, doc.vector) : 0;
      vectorScores.push(vScore);
      // BM25
      let bm25 = 0;
      const tf = new Map<string, number>();
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      const docLen = tokens.length;
      const lenNorm = 1 - this.config.b + this.config.b * (docLen / Math.max(1, this.avgDocLen));
      for (const qt of qTokens) {
        const f = tf.get(qt) ?? 0;
        if (f === 0) continue;
        const df = this.docFreq.get(qt) ?? 0;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        bm25 += idf * ((f * (this.config.k1 + 1)) / (f + this.config.k1 * lenNorm));
      }
      bm25Scores.push(bm25);
    }

    // 归一化（min-max 到 [0,1]）
    const norm = (arr: number[]): number[] => {
      const max = Math.max(...arr, 0);
      if (max === 0) return arr.map(() => 0);
      return arr.map((x) => x / max);
    };
    const vN = norm(vectorScores);
    const bN = norm(bm25Scores);

    let scored: HybridScored<D>[];
    if (this.config.useRRF) {
      // RRF 融合
      const vRanked = vectorScores
        .map((s, i) => ({ i, s }))
        .sort((a, b) => b.s - a.s)
        .map((r, rank) => ({ i: r.i, rank: rank + 1 }));
      const bRanked = bm25Scores
        .map((s, i) => ({ i, s }))
        .sort((a, b) => b.s - a.s)
        .map((r, rank) => ({ i: r.i, rank: rank + 1 }));
      const rrf = new Array(this.docs.length).fill(0) as number[];
      for (const r of vRanked) rrf[r.i] = (rrf[r.i] ?? 0) + 1 / (this.config.rrfK + r.rank);
      for (const r of bRanked) rrf[r.i] = (rrf[r.i] ?? 0) + 1 / (this.config.rrfK + r.rank);
      scored = this.docs.map((doc, i) => ({
        doc,
        score: rrf[i] ?? 0,
        components: {
          vector: vectorScores[i] ?? 0,
          bm25: bm25Scores[i] ?? 0,
          rrf: rrf[i] ?? 0,
        },
      }));
    } else {
      scored = this.docs.map((doc, i) => ({
        doc,
        score: this.config.alpha * (vN[i] ?? 0) + this.config.beta * (bN[i] ?? 0),
        components: {
          vector: vectorScores[i] ?? 0,
          bm25: bm25Scores[i] ?? 0,
        },
      }));
    }

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }
}
