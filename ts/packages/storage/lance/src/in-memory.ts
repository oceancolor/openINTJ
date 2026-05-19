import { cosineSimilarity } from "@openintj/core";
import {
  type VectorRow,
  VectorRowSchema,
  type VectorSearchOpts,
  type VectorSearchResult,
  type VectorStore,
} from "./types.js";

/**
 * 内存向量存储 —— 用于测试 + 兜底。
 * 无持久化，进程重启即丢。
 */
export class InMemoryVectorStore implements VectorStore {
  readonly name = "in-memory";
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
