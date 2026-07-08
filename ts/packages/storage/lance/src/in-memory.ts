import { cosineSimilarity } from "@openintj/core";
import {
  type VectorRow,
  VectorRowSchema,
  type VectorSearchOpts,
  type VectorSearchResult,
  type VectorStore,
} from "./types.js";

const tokenize = (text: string): string[] =>
  text
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 0);

/**
 * 内存向量存储 —— 用于测试 + 兜底。
 * 无持久化，进程重启即丢。
 *
 * 也实现了 `searchText`（BM25 词法检索）——不是为了大规模性能，而是让 `hybridVectorSearch`
 * 的融合逻辑在不装 LanceDB 的环境下也可端到端测试；LanceDBVectorStore 用原生 FTS 做同一件事。
 */
export class InMemoryVectorStore implements VectorStore {
  readonly name = "in-memory";
  readonly supportsFts = true;
  private rows = new Map<string, VectorRow>();
  private _dimension = 0;

  get dimension(): number {
    return this._dimension;
  }

  async init(): Promise<void> {
    // no-op
  }

  async upsert(rows: readonly VectorRow[]): Promise<void> {
    for (const r of rows) {
      const validated = VectorRowSchema.parse(r);
      if (this._dimension === 0) this._dimension = validated.embedding.length;
      else if (validated.embedding.length !== this._dimension) {
        throw new Error(
          `InMemoryVectorStore: embedding dim mismatch ${validated.embedding.length} vs ${this._dimension}`,
        );
      }
      this.rows.set(validated.fragmentId, validated);
    }
  }

  async delete(fragmentIds: readonly string[]): Promise<number> {
    let n = 0;
    for (const id of fragmentIds) if (this.rows.delete(id)) n++;
    return n;
  }

  async search(
    queryEmbedding: readonly number[],
    opts: VectorSearchOpts,
  ): Promise<VectorSearchResult[]> {
    const memTypes = opts.memoryTypes ? new Set(opts.memoryTypes) : null;
    const tags = opts.taskTags ? new Set(opts.taskTags) : null;
    const out: VectorSearchResult[] = [];
    for (const r of this.rows.values()) {
      if (memTypes && !memTypes.has(r.memoryType)) continue;
      if (opts.minImportance !== undefined && r.importance < opts.minImportance) continue;
      if (tags && !r.taskTags.some((t) => tags.has(t))) continue;
      const score = cosineSimilarity(queryEmbedding, r.embedding);
      out.push({ row: r, score, distance: 1 - score });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, Math.max(0, opts.topK));
  }

  async ensureFtsIndex(): Promise<void> {
    // 内存实现每次查询即时算 BM25，无需预建索引。
  }

  async searchText(query: string, opts: VectorSearchOpts): Promise<VectorSearchResult[]> {
    const qTokens = tokenize(query);
    if (qTokens.length === 0) return [];
    const memTypes = opts.memoryTypes ? new Set(opts.memoryTypes) : null;
    const tags = opts.taskTags ? new Set(opts.taskTags) : null;

    // 先按同一过滤语义收集候选，再在候选集上算 BM25（df/avgLen 基于候选集，query 内一致）。
    const candidates: Array<{ row: VectorRow; tokens: string[] }> = [];
    for (const r of this.rows.values()) {
      if (memTypes && !memTypes.has(r.memoryType)) continue;
      if (opts.minImportance !== undefined && r.importance < opts.minImportance) continue;
      if (tags && !r.taskTags.some((t) => tags.has(t))) continue;
      candidates.push({ row: r, tokens: tokenize(r.content) });
    }
    if (candidates.length === 0) return [];

    const n = candidates.length;
    const df = new Map<string, number>();
    for (const c of candidates) {
      for (const t of new Set(c.tokens)) df.set(t, (df.get(t) ?? 0) + 1);
    }
    const avgLen = candidates.reduce((s, c) => s + c.tokens.length, 0) / n;
    const k1 = 1.5;
    const b = 0.75;

    const scored: VectorSearchResult[] = [];
    for (const c of candidates) {
      const tf = new Map<string, number>();
      for (const t of c.tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      const lenNorm = 1 - b + b * (c.tokens.length / Math.max(1, avgLen));
      let bm25 = 0;
      for (const qt of qTokens) {
        const f = tf.get(qt) ?? 0;
        if (f === 0) continue;
        const dfi = df.get(qt) ?? 0;
        const idf = Math.log(1 + (n - dfi + 0.5) / (dfi + 0.5));
        bm25 += idf * ((f * (k1 + 1)) / (f + k1 * lenNorm));
      }
      if (bm25 > 0) scored.push({ row: c.row, score: bm25, distance: 0 });
    }
    scored.sort((a, b2) => b2.score - a.score);
    return scored.slice(0, Math.max(0, opts.topK));
  }

  async scanAll(): Promise<VectorRow[]> {
    return [...this.rows.values()];
  }

  async count(): Promise<number> {
    return this.rows.size;
  }

  async close(): Promise<void> {
    this.rows.clear();
  }
}
