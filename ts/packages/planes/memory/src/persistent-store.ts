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
import {
  type CanonicalMemoryRecord,
  type CanonicalMemoryRepository,
  CandidateConflictError,
  candidateReceiveKeyString,
  type CandidateReceiveKey,
} from "./canonical.js";
import { type AddOptions, MemoryStore, type MemoryStoreConfig } from "./store.js";

export interface PersistentMemoryStoreOpts {
  vectorStore: VectorStore;
  metadataStore: MetadataStore;
  embedder?: EmbeddingProvider;
  storeConfig?: Partial<MemoryStoreConfig>;
  /** 透传给 MemoryStore 的 HookBus，用于 emit `event.MEMORY_WRITTEN` change-feed。 */
  hooks?: HookBus;
  /**
   * G2：canonical 仓储（SQLite 权威源）。注入后 hydrate/写入/删除全部走 canonical
   * mutation 边界，向量库降为 outbox 投影；不注入时保持 legacy 双写行为。
   */
  canonical?: CanonicalMemoryRepository;
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

const SCOPE_TAG_RE = /^(workspace|task|conversation|principal):/;

const fragmentToCanonicalRecord = (f: MemoryFragment): CanonicalMemoryRecord => ({
  fragmentId: f.fragmentId,
  content: f.content,
  memoryType: f.memoryType,
  importance: f.importance,
  taskTags: [...f.taskTags],
  scopeTags: f.taskTags.filter((t) => SCOPE_TAG_RE.test(t)),
  metadata: f.metadata,
  summaries: f.summaries,
  embedding: [...f.embedding],
  source: "runtime",
  state: "active",
  revision: 0,
  timestamp: f.timestamp,
  accessCount: f.accessCount,
  lastAccessed: f.lastAccessed,
});

const canonicalRecordToFragment = (r: CanonicalMemoryRecord): MemoryFragment =>
  MemoryFragmentSchema.parse({
    fragmentId: r.fragmentId,
    content: r.content,
    embedding: r.embedding,
    memoryType: r.memoryType,
    importance: r.importance,
    taskTags: [...new Set([...r.taskTags, ...r.scopeTags])],
    metadata: r.metadata,
    summaries: r.summaries,
    timestamp: r.timestamp,
    accessCount: r.accessCount,
    lastAccessed: r.lastAccessed,
  });

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
  readonly canonical: CanonicalMemoryRepository | undefined;
  private pending: Promise<void> = Promise.resolve();
  private readonly hydrateOnInit: boolean;
  private isInitialized = false;
  /** 最近一次持久化失败（链已隔离，不毒化后续写入；供状态面读取）。 */
  lastPersistError: string | undefined;

  constructor(opts: PersistentMemoryStoreOpts) {
    super(opts.storeConfig ?? {}, {
      ...(opts.embedder ? { embedder: opts.embedder } : {}),
      ...(opts.hooks ? { hooks: opts.hooks } : {}),
    });
    this.vectorStore = opts.vectorStore;
    this.metadataStore = opts.metadataStore;
    this.canonical = opts.canonical;
    this.hydrateOnInit = opts.hydrateOnInit ?? true;
  }

  async init(): Promise<void> {
    if (this.isInitialized) return;
    await this.vectorStore.init();
    await this.metadataStore.init();
    await this.metadataStore.migrate();
    if (this.canonical) {
      await this.canonical.init();
      if (this.hydrateOnInit) {
        // 权威恢复：只从 canonical 读正文；被 suppression 的不进内存（F12 P0 读检查）。
        const records = await this.canonical.listRecords();
        for (const r of records) {
          if (await this.canonical.isSuppressed(r.fragmentId)) continue;
          const f = canonicalRecordToFragment(r);
          if (f.memoryType === "short_term") this.shortTerm.push(f);
          else if (f.memoryType === "working") this.working.push(f);
          else this.longTerm.push(f);
        }
      }
      // 索引落后不阻塞启动：投影失败只记状态，正文/词法检索照常。
      await this.projectIndex().catch(() => undefined);
    } else if (this.hydrateOnInit) {
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
    if (f.embedding.length > 0) this.scheduleWrite(f);
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
    if (!content.trim() && opts.embedding === undefined) {
      return super.addShortTermAsync(content, opts);
    }
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
      if (this.canonical) {
        const canonical = this.canonical;
        this.enqueuePending(() =>
          canonical.applyMutation({
            writeKey: `delete:${fragmentId}:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`,
            inputHash: fragmentId,
            kind: "delete",
            fragmentId,
          }).then(() => undefined),
        );
      } else {
        this.enqueuePending(() =>
          Promise.all([
            this.vectorStore.delete([fragmentId]),
            this.metadataStore.deleteFragmentMeta([fragmentId]),
          ]).then(() => undefined),
        );
      }
    }
    return ok;
  }

  /** E06 修复：clearWorking 同步走 canonical mutation，不再只清内存留墓碑复活。 */
  override clearWorking(): void {
    super.clearWorking();
    if (this.canonical) {
      const canonical = this.canonical;
      this.enqueuePending(() =>
        canonical
          .applyMutation({
            writeKey: `clear:working:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`,
            inputHash: "working",
            kind: "clear_tier",
            tier: "working",
          })
          .then(() => undefined),
      );
    }
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
    await this.canonical?.close();
    this.isInitialized = false;
  }

  /**
   * E05 修复：pending 链逐项隔离——单条失败只记 lastPersistError，
   * 绝不让整条链进入 rejected 毒化后续所有写入。
   */
  private enqueuePending(task: () => Promise<void>): void {
    this.pending = this.pending.then(task).catch((error: unknown) => {
      this.lastPersistError = error instanceof Error ? error.message : String(error);
    });
  }

  private scheduleWrite(f: MemoryFragment): void {
    this.enqueuePending(() => this.persist(f));
  }

  private async persist(f: MemoryFragment): Promise<void> {
    if (this.canonical) {
      // canonical 是权威：先落正文+outbox，向量投影交给 projectIndex。
      await this.canonical.applyMutation({
        // 写键含内容哈希：同内容重放命中收据（replayed），内容变更是新 mutation。
        writeKey: `upsert:${f.fragmentId}:${contentHash(f)}`,
        inputHash: contentHash(f),
        kind: "upsert",
        record: fragmentToCanonicalRecord(f),
      });
      await this.projectIndex();
      return;
    }
    const row = fragmentToVectorRow(f);
    const meta = fragmentToMeta(f);
    try {
      await this.vectorStore.upsert([row]);
      await this.metadataStore.putFragmentMeta([meta]);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!/not initialized/i.test(message)) throw error;
      await this.vectorStore.init();
      await this.vectorStore.upsert([row]);
      await this.metadataStore.putFragmentMeta([meta]);
    }
  }

  /**
   * F06：把 index outbox 投影到向量库。逐 job 隔离：坏 job 只累计 attempts，
   * 不阻塞队列；过期 upsert（record 已删/修订更新）跳过，防旧投影复活（T14）。
   */
  async projectIndex(limit = 200): Promise<{ applied: number; failed: number; skipped: number }> {
    if (!this.canonical) return { applied: 0, failed: 0, skipped: 0 };
    const jobs = await this.canonical.pendingIndexJobs(limit);
    let applied = 0;
    let failed = 0;
    let skipped = 0;
    for (const job of jobs) {
      try {
        if (job.kind === "delete") {
          await this.vectorStore.delete([job.fragmentId]);
          await this.metadataStore.deleteFragmentMeta([job.fragmentId]);
          await this.canonical.markIndexJobApplied(job.jobId);
          applied++;
          continue;
        }
        const record = await this.canonical.getRecord(job.fragmentId);
        if (!record || record.state !== "active" || record.revision !== job.revision) {
          await this.canonical.markIndexJobApplied(job.jobId);
          skipped++;
          continue;
        }
        const fragment = canonicalRecordToFragment(record);
        await this.vectorStore.upsert([fragmentToVectorRow(fragment)]);
        await this.metadataStore.putFragmentMeta([fragmentToMeta(fragment)]);
        await this.canonical.markIndexJobApplied(job.jobId);
        applied++;
      } catch (error) {
        await this.canonical.markIndexJobFailed(
          job.jobId,
          error instanceof Error ? error.message : String(error),
        );
        failed++;
      }
    }
    return { applied, failed, skipped };
  }

  /** F07：空间不兼容/索引损坏时开新代次并从 canonical 全量重建投影。 */
  async rebuildIndex(spaceFingerprint: string): Promise<{ generation: number; jobs: number }> {
    if (!this.canonical) throw new Error("rebuildIndex requires canonical repository");
    const generation = await this.canonical.startIndexGeneration(spaceFingerprint);
    const records = await this.canonical.listRecords();
    for (const r of records) {
      await this.canonical.applyMutation({
        writeKey: `rebuild:${generation}:${r.fragmentId}`,
        inputHash: `${r.fragmentId}:${r.revision}`,
        kind: "upsert",
        record: r,
      });
    }
    await this.projectIndex(1000);
    return { generation, jobs: records.length };
  }

  async indexStatus(): Promise<{
    canonical: boolean;
    pending: number;
    failed: number;
    generation: number;
    generationFingerprint?: string;
    vectorDimension: number;
    embedderDimension: number;
    dimensionCompatible: boolean;
    lastPersistError?: string;
  }> {
    const stats = this.canonical
      ? await this.canonical.indexStats()
      : { pending: 0, failed: 0, generation: 0 };
    const fingerprint = this.canonical ? await this.canonical.indexGenerationFingerprint() : undefined;
    return {
      canonical: this.canonical !== undefined,
      ...stats,
      ...(fingerprint ? { generationFingerprint: fingerprint } : {}),
      vectorDimension: this.vectorStore.dimension,
      embedderDimension: this.embedder.dimension,
      dimensionCompatible:
        this.vectorStore.dimension === 0 || this.vectorStore.dimension === this.embedder.dimension,
      ...(this.lastPersistError ? { lastPersistError: this.lastPersistError } : {}),
    };
  }

  /** F12 P0：forget 一个 fragment = suppression + tombstone，同走 mutation 边界。 */
  async forgetFragment(fragmentId: string, principalId: string, reason: string): Promise<void> {
    if (!this.canonical) throw new Error("forgetFragment requires canonical repository");
    await this.canonical.addSuppression(fragmentId, principalId, reason);
    this.remove(fragmentId);
    await this.awaitPendingWrites();
  }

  async memoryEpoch(): Promise<number> {
    return this.canonical ? this.canonical.memoryEpoch() : 0;
  }

  /** F06 接收边界：候选批次幂等接收（只存待审）。 */
  async receiveCandidateBatch(input: {
    key: CandidateReceiveKey;
    inputHash: string;
    candidates: readonly Record<string, unknown>[];
  }): Promise<{ receiveKey: string; outcome: "applied" | "replayed"; candidateCount: number }> {
    if (!this.canonical) throw new Error("receiveCandidateBatch requires canonical repository");
    const receipt = await this.canonical.receiveCandidates(input).catch((error: unknown) => {
      // storage 层有自己的错误类，按 code 归一为 plane 层契约错误。
      if ((error as { code?: string })?.code === "MEMORY_CANDIDATE_CONFLICT") {
        throw new CandidateConflictError(error instanceof Error ? error.message : String(error));
      }
      throw error;
    });
    return {
      receiveKey: candidateReceiveKeyString(input.key),
      outcome: receipt.outcome,
      candidateCount: receipt.candidateCount,
    };
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
