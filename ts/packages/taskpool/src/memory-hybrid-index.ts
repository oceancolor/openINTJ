/**
 * MemoryHybridIndex —— 在 HybridRetriever 之上包一层「session 级、增量维护」的记忆检索索引。
 *
 * 动机（roadmap A1）：原先 server/desktop 每次混合查询都 `new HybridRetriever().index(store.all)`
 * 全量重建。这里改为持有一个长生命周期实例，开局用现有片段 seed 一次，之后订阅
 * `event.MEMORY_WRITTEN` change-feed 做增量 upsert/remove —— 检索随对话「越用越好」且不再重建。
 *
 * 设计：
 *  - 与具体 store 解耦：只吃 MemoryFragment 与 HookBus（均为 core 类型）。
 *  - 过滤（memoryTypes / taskTags）：共享索引含全量，无法在打分前裁剪 → 过量取回再筛。
 *  - 零依赖默认：不订阅 hooks 就是个手动维护的索引。
 */

import type { HookBus, MemoryFragment } from "@openintj/core";
import { type HybridConfig, HybridRetriever, type HybridScored } from "./hybrid-retriever.js";

export interface MemoryHybridDoc {
  id: string;
  text: string;
  vector?: readonly number[];
  metadata: {
    memoryType: string;
    taskTags: readonly string[];
    importance: number;
  };
}

export type MemoryHybridHit = HybridScored<MemoryHybridDoc>;

export interface MemoryHybridSearchOpts {
  topK?: number;
  /** 仅保留这些 memoryType。 */
  memoryTypes?: readonly string[];
  /** 任意一个标签命中即保留。 */
  taskTags?: readonly string[];
  /** 按查询覆盖融合配置（alpha/beta/useRRF/...）；全是 search-time 量，不触发重建。 */
  config?: Partial<HybridConfig>;
}

const toDoc = (f: MemoryFragment): MemoryHybridDoc => ({
  id: f.fragmentId,
  text: f.content,
  vector: f.embedding,
  metadata: {
    memoryType: f.memoryType,
    taskTags: f.taskTags,
    importance: f.importance,
  },
});

export class MemoryHybridIndex {
  private readonly retriever: HybridRetriever<MemoryHybridDoc>;
  private unsub: (() => void) | undefined;

  constructor(opts: { config?: Partial<HybridConfig> } = {}) {
    this.retriever = new HybridRetriever<MemoryHybridDoc>(
      opts.config ? { config: opts.config } : {},
    );
  }

  get size(): number {
    return this.retriever.size;
  }

  /** 用现有片段全量种子（启动 hydrate 后调用一次）。 */
  seed(fragments: readonly MemoryFragment[]): void {
    this.retriever.index(fragments.map(toDoc));
  }

  /**
   * 订阅 change-feed：add/update → upsert，remove → remove。返回退订函数。
   * 重复调用会先退订旧订阅。
   */
  subscribe(hooks: HookBus): () => void {
    this.unsub?.();
    const off = hooks.on("event.MEMORY_WRITTEN", (ctx) => {
      const { fragment, op } = ctx.payload;
      if (op === "remove") this.retriever.remove(fragment.fragmentId);
      else this.retriever.upsert(toDoc(fragment));
    });
    this.unsub = off;
    return off;
  }

  /** 手动增量（不经 change-feed 时用）。 */
  upsertFragment(f: MemoryFragment): void {
    this.retriever.upsert(toDoc(f));
  }

  removeFragment(id: string): void {
    this.retriever.remove(id);
  }

  search(
    query: string,
    queryVector: readonly number[] | undefined,
    opts: MemoryHybridSearchOpts = {},
  ): MemoryHybridHit[] {
    const topK = opts.topK ?? 10;
    const hasFilter = (opts.memoryTypes?.length ?? 0) > 0 || (opts.taskTags?.length ?? 0) > 0;
    // 有过滤时过量取回（共享索引含全量），筛完再 slice。
    const fetch = hasFilter ? Math.max(topK * 4, this.retriever.size) : topK;
    let hits = this.retriever.search(query, queryVector, fetch, opts.config);
    if (opts.memoryTypes && opts.memoryTypes.length > 0) {
      const set = new Set(opts.memoryTypes);
      hits = hits.filter((h) => set.has(h.doc.metadata.memoryType));
    }
    if (opts.taskTags && opts.taskTags.length > 0) {
      const tagSet = new Set(opts.taskTags);
      hits = hits.filter((h) => h.doc.metadata.taskTags.some((t) => tagSet.has(t)));
    }
    return hits.slice(0, topK);
  }

  /** 退订 change-feed。 */
  dispose(): void {
    this.unsub?.();
    this.unsub = undefined;
  }
}
