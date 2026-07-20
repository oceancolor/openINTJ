import {
  type EmbeddingProvider,
  type MemoryFragment,
  type ShaderConfig,
  ShaderConfigSchema,
  SimpleEmbedder,
  type TaskTypeType,
  cosineSimilarity,
  decayImportance,
  tokenizeLexical,
} from "@openintj/core";
import type { MemoryStore } from "./store.js";

export interface RetrieveOptions {
  topK?: number;
  taskType?: TaskTypeType;
  minImportance?: number;
  /** 注入 query embedding；不传则用 simpleEmbedding 生成。 */
  queryEmbedding?: readonly number[];
}

export interface RankedMemory {
  fragment: MemoryFragment;
  score: number;
  components: {
    relevance: number;
    keyword: number;
    recency: number;
  };
}

/**
 * 朴素混合检索：vec_score × relevance_weight
 *                 + keyword_score × recency_weight  (按 Python v2 顺序)
 *                 + recency_score × importance_weight
 *
 * 关键修复（RFC-003）：使用 ShaderConfig.recencyHalfLifeHours 作为半衰期，
 * 而非 Python 的 `max_summary_length / 10` 误用。
 */
export class MemoryRetriever {
  readonly store: MemoryStore;
  readonly shaderConfig: ShaderConfig;
  readonly embedder: EmbeddingProvider;
  private readonly clock: () => number;

  constructor(
    store: MemoryStore,
    shaderConfig?: Partial<ShaderConfig>,
    opts?: { clock?: () => number; embedder?: EmbeddingProvider },
  ) {
    this.store = store;
    this.shaderConfig = ShaderConfigSchema.parse(shaderConfig ?? {});
    this.clock = opts?.clock ?? (() => Date.now() / 1000);
    this.embedder = opts?.embedder ?? store.embedder;
  }

  retrieve(query: string, opts: RetrieveOptions = {}): RankedMemory[] {
    const topK = opts.topK ?? this.shaderConfig.maxFragmentsPerQuery;
    const minImportance = opts.minImportance ?? 0;
    let qEmb: number[];
    if (opts.queryEmbedding !== undefined) {
      qEmb = [...opts.queryEmbedding];
    } else {
      const r = this.embedder.embed(query);
      if (r instanceof Promise) {
        // sync API；调用方应 retrieveAsync 或预提供 queryEmbedding
        throw new Error(
          `MemoryRetriever.retrieve: embedder '${this.embedder.name}' is async; use retrieveAsync or pass opts.queryEmbedding`,
        );
      }
      qEmb = r;
    }
    const qKeywords = new Set(tokenizeLexical(query));
    const cjkQuery = /\p{Script=Han}/u.test(query);
    const halfLife = this.shaderConfig.recencyHalfLifeHours;
    const now = this.clock();

    const scored: RankedMemory[] = [];

    for (const fragment of this.store.all) {
      const decayed = decayImportance(fragment, halfLife, now);
      if (decayed < minImportance) continue;

      const relevance = cosineSimilarity(qEmb, fragment.embedding);
      const contentWords = new Set(tokenizeLexical(fragment.content));
      let overlap = 0;
      for (const w of qKeywords) if (contentWords.has(w)) overlap++;
      const keyword = overlap / Math.max(1, qKeywords.size);

      const recency = decayed; // 已应用半衰期

      let score =
        this.shaderConfig.relevanceWeight * relevance +
        this.shaderConfig.recencyWeight * keyword +
        this.shaderConfig.importanceWeight * recency;
      // 中文短问句常只与事实句共享一两个关键二元组；语义向量（尤其 simple/hash fallback）
      // 的随机波动不应盖过这个确定性命中。英文路径保持 Python v2 的原始权重。
      if (cjkQuery) score += 0.5 * keyword;

      // 任务标签加权（与 Python v2 一致：×1.3）
      if (opts.taskType && fragment.taskTags.includes(opts.taskType)) {
        score *= 1.3;
      }

      scored.push({
        fragment,
        score,
        components: { relevance, keyword, recency },
      });
    }

    scored.sort((a, b) => b.score - a.score);

    const top = scored.slice(0, topK);
    for (const r of top) {
      r.fragment.accessCount += 1;
      r.fragment.lastAccessed = now;
    }
    return top;
  }

  async retrieveAsync(query: string, opts: RetrieveOptions = {}): Promise<RankedMemory[]> {
    if (opts.queryEmbedding !== undefined) {
      return this.retrieve(query, opts);
    }
    const qEmb = await Promise.resolve(this.embedder.embed(query));
    return this.retrieve(query, { ...opts, queryEmbedding: qEmb });
  }
}

/** SimpleEmbedder 默认导出方便用户快速创建。 */
export { SimpleEmbedder };

/**
 * 把外部检索器（如 HybridRetriever）返回的 {id, score} 解析回 RankedMemory[]，
 * 供 ContextEngine.candidateRetrieve 使用（A1.3）。
 *
 * - 从 store 按 id 查回完整 MemoryFragment（含 importance/summaries/timestamp，供后续着色/衰减）。
 * - 复用 MemoryRetriever 的 taskType 命中 ×1.3 加成，保持与默认路径行为一致。
 * - 命中片段 bump accessCount / lastAccessed（与默认路径一致，强化「常用更近」）。
 * - 命中不到的 id 跳过（容忍 change-feed 与 store 之间的瞬时不一致）。
 */
export const fragmentsToRanked = (
  store: MemoryStore,
  scored: ReadonlyArray<{ id: string; score: number }>,
  opts: { taskType?: TaskTypeType; clock?: () => number } = {},
): RankedMemory[] => {
  const byId = new Map(store.all.map((f) => [f.fragmentId, f]));
  const now = opts.clock ? opts.clock() : Date.now() / 1000;
  const out: RankedMemory[] = [];
  for (const s of scored) {
    const fragment = byId.get(s.id);
    if (!fragment) continue;
    let score = s.score;
    if (opts.taskType && fragment.taskTags.includes(opts.taskType)) score *= 1.3;
    fragment.accessCount += 1;
    fragment.lastAccessed = now;
    out.push({ fragment, score, components: { relevance: s.score, keyword: 0, recency: 0 } });
  }
  return out;
};
