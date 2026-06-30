import {
  type EmbeddingProvider,
  type HookBus,
  type MemoryFragment,
  MemoryFragmentSchema,
  type MemoryType,
  contentHash,
} from "@openintj/core";
import type {
  VectorRow,
  VectorSearchOpts,
  VectorSearchResult,
  VectorStore,
} from "@openintj/storage-lance";
import type { FragmentMeta, MetadataStore } from "@openintj/storage-sqlite";
import { type AddOptions, MemoryStore, type MemoryStoreConfig } from "./store.js";

export interface PersistentMemoryStoreOpts {
  vectorStore: VectorStore;
  metadataStore: MetadataStore;
  embedder?: EmbeddingProvider;
  storeConfig?: Partial<MemoryStoreConfig>;
  /** 透传给 MemoryStore 的 HookBus，用于 emit `event.MEMORY_WRITTEN` change-feed。 */
  hooks?: HookBus;
  /**
   * 启动时是否 hydrate（从持久化层把已有 fragments 加载到内存）。
   * 默认 true。
   */
  hydrateOnInit?: boolean;
}

const fragmentToVectorRow = (f: MemoryFragment): VectorRow => ({
  fragmentId: f.fragmentId,
  content: f.content,
  embedding: [...f.embedding],
  memoryType: f.memoryType,
  importance: f.importance,
  taskTags: [...f.taskTags],
  contentHash: contentHash(f),
  timestamp: f.timestamp,
  accessCount: f.accessCount,
  lastAccessed: f.lastAccessed,
  metadataJson: JSON.stringify(f.metadata),
  summariesJson: JSON.stringify(f.summaries),
});

const fragmentToMeta = (f: MemoryFragment): FragmentMeta => ({
  fragmentId: f.fragmentId,
  memoryType: f.memoryType,
  importance: f.importance,
  contentHash: contentHash(f),
  taskTagsCsv: f.taskTags.join(","),
  metadataJson: JSON.stringify(f.metadata),
  summariesJson: JSON.stringify(f.summaries),
  timestamp: f.timestamp,
  accessCount: f.accessCount,
  lastAccessed: f.lastAccessed,
});

const vectorRowToFragment = (r: VectorRow): MemoryFragment => {
  let metadata: Record<string, unknown> = {};
  let summaries: Record<number, string> = {};
  try {
    metadata = JSON.parse(r.metadataJson || "{}") as Record<string, unknown>;
  } catch {
    metadata = {};
  }
  try {
    const raw = JSON.parse(r.summariesJson || "{}") as Record<string, string>;
    summaries = Object.fromEntries(Object.entries(raw).map(([k, v]) => [Number(k), v]));
  } catch {
    summaries = {};
  }
  return MemoryFragmentSchema.parse({
    fragmentId: r.fragmentId,
    content: r.content,
    embedding: r.embedding,
    memoryType: r.memoryType,
    importance: r.importance,
    taskTags: [...r.taskTags],
    metadata,
    summaries,
    timestamp: r.timestamp,
    accessCount: r.accessCount,
    lastAccessed: r.lastAccessed,
  });
};

/**
 * PersistentMemoryStore —— 在 MemoryStore 之上叠加双写到 VectorStore + MetadataStore。
 *
 * 行为：
 *  - init(): 调用 vectorStore.init() + metadataStore.init() + migrate()，
 *    然后从 vectorStore.scanAll() hydrate 内存三层
 *  - addShortTerm/addWorking/addLongTerm: 内存 + dual-write 到持久化层
 *  - vectorSearch(): 直接走 vectorStore.search（不依赖内存）
 *  - sync(): 强制把内存全量重写到持久化层（启动后修复用）
 *
 * 注：所有 add* 方法仍是 sync，但 dual-write 在异步任务中触发。
 * 若需要确认写盘，调用 awaitPendingWrites()。
 */
export class PersistentMemoryStore extends MemoryStore {
  readonly vectorStore: VectorStore;
  readonly metadataStore: MetadataStore;
  private pending: Promise<void> = Promise.resolve();
  private readonly hydrateOnInit: boolean;
  private isInitialized = false;

  constructor(opts: PersistentMemoryStoreOpts) {
    super(opts.storeConfig ?? {}, {
      ...(opts.embedder ? { embedder: opts.embedder } : {}),
      ...(opts.hooks ? { hooks: opts.hooks } : {}),
    });
    this.vectorStore = opts.vectorStore;
    this.metadataStore = opts.metadataStore;
    this.hydrateOnInit = opts.hydrateOnInit ?? true;
  }

  async init(): Promise<void> {
    if (this.isInitialized) return;
    await this.vectorStore.init();
    await this.metadataStore.init();
    await this.metadataStore.migrate();
    if (this.hydrateOnInit) {
      const rows = await this.vectorStore.scanAll();
      for (const r of rows) {
        const f = vectorRowToFragment(r);
        if (f.memoryType === "short_term") this.shortTerm.push(f);
        else if (f.memoryType === "working") this.working.push(f);
        else this.longTerm.push(f);
      }
    }
    this.isInitialized = true;
  }

  override addShortTerm(content: string, opts: AddOptions = {}): MemoryFragment {
    const f = super.addShortTerm(content, opts);
    this.scheduleWrite(f);
    return f;
  }

  override addWorking(content: string, opts: AddOptions = {}): MemoryFragment {
    const f = super.addWorking(content, opts);
    this.scheduleWrite(f);
    return f;
  }

  override addLongTerm(content: string, opts: AddOptions = {}): MemoryFragment {
    const f = super.addLongTerm(content, opts);
    this.scheduleWrite(f);
    return f;
  }

  override async addShortTermAsync(
    content: string,
    opts: AddOptions = {},
  ): Promise<MemoryFragment> {
    const f = await super.addShortTermAsync(content, opts);
    await this.persist(f);
    return f;
  }

  override async addWorkingAsync(content: string, opts: AddOptions = {}): Promise<MemoryFragment> {
    const f = await super.addWorkingAsync(content, opts);
    await this.persist(f);
    return f;
  }

  override async addLongTermAsync(content: string, opts: AddOptions = {}): Promise<MemoryFragment> {
    const f = await super.addLongTermAsync(content, opts);
    await this.persist(f);
    return f;
  }

  override remove(fragmentId: string): boolean {
    const ok = super.remove(fragmentId);
    if (ok) {
      this.pending = this.pending.then(() =>
        Promise.all([
          this.vectorStore.delete([fragmentId]),
          this.metadataStore.deleteFragmentMeta([fragmentId]),
        ]).then(() => undefined),
      );
    }
    return ok;
  }

  /** 直接走持久化层做向量搜索（不限于内存中已加载的片段）。 */
  async vectorSearch(
    queryEmbedding: readonly number[],
    opts: VectorSearchOpts,
  ): Promise<VectorSearchResult[]> {
    return this.vectorStore.search(queryEmbedding, opts);
  }

  /** 等待所有挂起的 dual-write 完成。 */
  async awaitPendingWrites(): Promise<void> {
    await this.pending;
  }

  /** 把内存全量重写到持久化层（启动后修复 / 一次性同步用）。 */
  async sync(): Promise<{ fragments: number }> {
    const rows = this.all.map(fragmentToVectorRow);
    const metas = this.all.map(fragmentToMeta);
    if (rows.length > 0) {
      await this.vectorStore.upsert(rows);
      await this.metadataStore.putFragmentMeta(metas);
    }
    return { fragments: rows.length };
  }

  async close(): Promise<void> {
    await this.awaitPendingWrites();
    await this.vectorStore.close();
    await this.metadataStore.close();
  }

  private scheduleWrite(f: MemoryFragment): void {
    this.pending = this.pending.then(() => this.persist(f));
  }

  private async persist(f: MemoryFragment): Promise<void> {
    await this.vectorStore.upsert([fragmentToVectorRow(f)]);
    await this.metadataStore.putFragmentMeta([fragmentToMeta(f)]);
  }

  /** 用 fragmentId 显式重新指派 memoryType（短期 → 长期 promotion 时用）。 */
  async reassignMemoryType(fragmentId: string, newType: MemoryType): Promise<boolean> {
    for (const list of [this.shortTerm, this.working, this.longTerm]) {
      const idx = list.findIndex((f) => f.fragmentId === fragmentId);
      if (idx >= 0) {
        const f = list[idx]!;
        f.memoryType = newType;
        list.splice(idx, 1);
        if (newType === "short_term") this.shortTerm.push(f);
        else if (newType === "working") this.working.push(f);
        else this.longTerm.push(f);
        await this.persist(f);
        this.emitWrite(f, "update");
        return true;
      }
    }
    return false;
  }
}
