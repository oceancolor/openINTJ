import { createHash } from "node:crypto";
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
  type CandidateOutcomeRecord,
  type CandidateReceiveKey,
} from "./canonical.js";
import { type EmbeddingSpace, embeddingSpaceForEmbedder } from "./embedding-space.js";
import { type AddOptions, MemoryStore, type MemoryStoreConfig } from "./store.js";
import type { MemoryCandidate } from "./writer.js";

export type CandidateReviewErrorCode =
  | "candidate_not_found"
  | "candidate_not_reviewable"
  | "candidate_revision_conflict"
  | "canonical_unavailable";

export class CandidateReviewError extends Error {
  readonly code: CandidateReviewErrorCode;
  constructor(code: CandidateReviewErrorCode, message: string) {
    super(message);
    this.name = "CandidateReviewError";
    this.code = code;
  }
}

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
  /** embedding 空间身份（真实 provider/model 名，供指纹与索引代次记录）。 */
  embeddingSpaceInfo?: { provider: string; model: string; transformVersion?: number };
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
  kind: f.kind,
  assertion: f.assertion,
  ...(f.validUntil !== undefined ? { validUntil: f.validUntil } : {}),
  ...(f.canonicalKey ? { canonicalKey: f.canonicalKey } : {}),
  ...(f.supersededBy ? { supersededBy: f.supersededBy } : {}),
});

/**
 * mutation 输入哈希必须覆盖完整记录（tier/importance/tags/embedding…）：
 * 只哈希正文会让晋升等状态变更被误判为重放而静默丢弃。
 */
const recordInputHash = (r: CanonicalMemoryRecord): string =>
  createHash("sha256")
    .update(
      JSON.stringify({
        fragmentId: r.fragmentId,
        content: r.content,
        memoryType: r.memoryType,
        importance: r.importance,
        taskTags: r.taskTags,
        metadata: r.metadata,
        summaries: r.summaries,
        embedding: r.embedding,
        timestamp: r.timestamp,
        kind: r.kind,
        assertion: r.assertion,
        validUntil: r.validUntil ?? null,
        canonicalKey: r.canonicalKey ?? null,
      }),
    )
    .digest("hex");

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
    kind: r.kind,
    assertion: r.assertion,
    ...(r.validUntil !== undefined ? { validUntil: r.validUntil } : {}),
    ...(r.canonicalKey ? { canonicalKey: r.canonicalKey } : {}),
    ...(r.supersededBy ? { supersededBy: r.supersededBy } : {}),
  });

/** F12：run 携带的 memory epoch 已落后（期间发生过 forget），授权点必须重建或拒绝。 */
export class MemoryEpochStaleError extends Error {
  readonly code = "MEMORY_EPOCH_STALE";
  constructor(
    readonly runEpoch: number,
    readonly currentEpoch: number,
  ) {
    super(`memory epoch stale: run=${runEpoch} current=${currentEpoch}; rebuild context before injecting`);
    this.name = "MemoryEpochStaleError";
  }
}

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
  private readonly embeddingSpaceInfo: PersistentMemoryStoreOpts["embeddingSpaceInfo"];
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
    this.embeddingSpaceInfo = opts.embeddingSpaceInfo;
    this.hydrateOnInit = opts.hydrateOnInit ?? true;
  }

  /** 当前 embedding 空间（含真实 provider/model）及其指纹。 */
  embeddingSpace(generation = 1): EmbeddingSpace {
    return embeddingSpaceForEmbedder(this.embedder, {
      ...(this.embeddingSpaceInfo?.provider ? { provider: this.embeddingSpaceInfo.provider } : {}),
      ...(this.embeddingSpaceInfo?.model ? { model: this.embeddingSpaceInfo.model } : {}),
      ...(this.embeddingSpaceInfo?.transformVersion
        ? { transformVersion: this.embeddingSpaceInfo.transformVersion }
        : {}),
      generation,
    });
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
    const before = this.tierSnapshot();
    const f = super.addShortTerm(content, opts);
    if (f.embedding.length > 0) this.scheduleWrite(f);
    this.reconcileTiers(before, f.fragmentId, false);
    return f;
  }

  override addWorking(content: string, opts: AddOptions = {}): MemoryFragment {
    const before = this.tierSnapshot();
    const f = super.addWorking(content, opts);
    this.scheduleWrite(f);
    this.reconcileTiers(before, f.fragmentId, false);
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
    const before = this.tierSnapshot();
    const f = await super.addShortTermAsync(content, opts);
    await this.persist(f);
    await this.reconcileTiers(before, f.fragmentId, true);
    return f;
  }

  override async addWorkingAsync(content: string, opts: AddOptions = {}): Promise<MemoryFragment> {
    const before = this.tierSnapshot();
    const f = await super.addWorkingAsync(content, opts);
    await this.persist(f);
    await this.reconcileTiers(before, f.fragmentId, true);
    return f;
  }

  override async addLongTermAsync(content: string, opts: AddOptions = {}): Promise<MemoryFragment> {
    const f = await super.addLongTermAsync(content, opts);
    await this.persist(f);
    return f;
  }

  override remove(fragmentId: string): boolean {
    const ok = super.remove(fragmentId);
    if (ok) this.scheduleDelete(fragmentId);
    return ok;
  }

  /** 删除一条 fragment 的持久层记录（canonical tombstone 或 legacy 双删）。 */
  private scheduleDelete(fragmentId: string): void {
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

  private tierSnapshot(): { longTerm: Set<string>; all: Set<string> } {
    return {
      longTerm: new Set(this.longTerm.map((f) => f.fragmentId)),
      all: new Set(this.all.map((f) => f.fragmentId)),
    };
  }

  /**
   * 溢出副作用对账（修 G2 遗留漂移）：短期溢出晋升的 fragment 必须补写新 tier，
   * 工作记忆溢出丢弃的 fragment 必须补删——否则重启后 tier 错位/已丢弃条目复活。
   */
  private async reconcileTiers(
    before: { longTerm: Set<string>; all: Set<string> },
    newFragmentId: string,
    awaitWrites: boolean,
  ): Promise<void> {
    const promoted = this.longTerm.filter(
      (f) => !before.longTerm.has(f.fragmentId) && f.fragmentId !== newFragmentId,
    );
    const now = new Set(this.all.map((f) => f.fragmentId));
    const discarded = [...before.all].filter((id) => !now.has(id));
    for (const f of promoted) {
      if (awaitWrites) await this.persist(f);
      else this.scheduleWrite(f);
    }
    for (const id of discarded) {
      if (awaitWrites && this.canonical) {
        await this.canonical.applyMutation({
          writeKey: `delete:${id}:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`,
          inputHash: id,
          kind: "delete",
          fragmentId: id,
        });
      } else {
        this.scheduleDelete(id);
      }
    }
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
      const record = fragmentToCanonicalRecord(f);
      const inputHash = recordInputHash(record);
      await this.canonical.applyMutation({
        // 写键含记录哈希：同状态重放命中收据（replayed），状态变更是新 mutation。
        writeKey: `upsert:${f.fragmentId}:${inputHash}`,
        inputHash,
        kind: "upsert",
        record,
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

  /**
   * F08 纠正：写入新值并 supersede 旧值（同事务边界内的两次 mutation）。
   * 旧值即刻在所有召回入口不可见，canonical 中保留 superseded 审计态。
   */
  async correctMemory(
    oldFragmentId: string,
    newContent: string,
    opts: AddOptions = {},
  ): Promise<MemoryFragment> {
    const replacement = await this.addLongTermAsync(newContent, opts);
    const old = this.supersedeFragment(oldFragmentId, replacement);
    if (old && this.canonical) {
      await this.canonical.applyMutation({
        writeKey: `supersede:${oldFragmentId}:${replacement.fragmentId}`,
        inputHash: `${oldFragmentId}->${replacement.fragmentId}`,
        kind: "mark_superseded",
        fragmentId: oldFragmentId,
        supersededBy: replacement.fragmentId,
      });
      await this.projectIndex();
    } else if (old) {
      this.scheduleDelete(oldFragmentId);
    }
    return replacement;
  }

  /** 按 canonicalKey 纠正全部当前主张（同 scope 内的旧值一并 supersede）。 */
  async correctByCanonicalKey(
    canonicalKey: string,
    newContent: string,
    opts: AddOptions = {},
  ): Promise<{ replacement: MemoryFragment; superseded: string[] }> {
    const olds = this.findByCanonicalKey(canonicalKey);
    const replacement = await this.addLongTermAsync(newContent, {
      ...opts,
      canonicalKey,
    });
    const superseded: string[] = [];
    for (const old of olds) {
      if (old.fragmentId === replacement.fragmentId) continue;
      this.supersedeFragment(old.fragmentId, replacement);
      superseded.push(old.fragmentId);
      if (this.canonical) {
        await this.canonical.applyMutation({
          writeKey: `supersede:${old.fragmentId}:${replacement.fragmentId}`,
          inputHash: `${old.fragmentId}->${replacement.fragmentId}`,
          kind: "mark_superseded",
          fragmentId: old.fragmentId,
          supersededBy: replacement.fragmentId,
        });
      } else {
        this.scheduleDelete(old.fragmentId);
      }
    }
    if (this.canonical) await this.projectIndex();
    return { replacement, superseded };
  }

  /**
   * F12 有效遗忘：suppression + tombstone + epoch 递增同事务提交（applyForget），
   * 内存即刻移除，向量投影由 delete job 清理。返回新 epoch。
   */
  async forgetFragment(fragmentId: string, principalId: string, reason: string): Promise<number> {
    if (!this.canonical) throw new Error("forgetFragment requires canonical repository");
    const { epoch } = await this.canonical.applyForget({ fragmentId, principalId, reason });
    this.remove(fragmentId);
    await this.awaitPendingWrites();
    await this.projectIndex();
    return epoch;
  }

  /**
   * F12 purge（物理删除基础版）：删 canonical 记录本体与未投影 outbox，
   * 同步清理向量/metadata 投影；suppression barrier（不含正文）保留防回潮。
   * 备份与已导出副本不在物理删除承诺内，报告中如实标注 unsupported。
   */
  async purgeMemory(
    fragmentId: string,
    principalId: string,
  ): Promise<{
    recordDeleted: boolean;
    pendingJobsPurged: number;
    vectorProjectionDeleted: boolean;
    barrierRetained: boolean;
    backups: "unsupported";
  }> {
    if (!this.canonical) throw new Error("purgeMemory requires canonical repository");
    await this.canonical.addSuppression(fragmentId, principalId, "purge");
    const result = await this.canonical.purgeFragment(fragmentId);
    this.remove(fragmentId);
    await this.awaitPendingWrites();
    await this.vectorStore.delete([fragmentId]);
    await this.metadataStore.deleteFragmentMeta([fragmentId]);
    return {
      ...result,
      vectorProjectionDeleted: true,
      barrierRetained: await this.canonical.isSuppressed(fragmentId),
      backups: "unsupported",
    };
  }

  /** F12 授权门：run 的 epoch 落后于当前 control epoch 时拒绝用旧上下文注入。 */
  async assertMemoryEpoch(runEpoch: number | undefined): Promise<void> {
    if (!this.canonical || runEpoch === undefined) return;
    const current = await this.canonical.memoryEpoch();
    if (runEpoch < current) {
      throw new MemoryEpochStaleError(runEpoch, current);
    }
  }

  async memoryEpoch(): Promise<number> {
    return this.canonical ? this.canonical.memoryEpoch() : 0;
  }

  /**
   * F14 记忆状态面：canonical/索引/向量空间/epoch/计数一次读清，
   * 供宿主状态栏与诊断接口展示（各维度独立，不合并成一个模糊的健康值）。
   */
  async memoryStatus(): Promise<{
    canonicalEnabled: boolean;
    activeCount: number;
    tombstonedCount: number;
    quarantinedCount: number;
    supersededCount: number;
    suppressedCount: number;
    memoryEpoch: number;
    index: Awaited<ReturnType<PersistentMemoryStore["indexStatus"]>>;
  }> {
    const index = await this.indexStatus();
    if (!this.canonical) {
      return {
        canonicalEnabled: false,
        activeCount: this.totalCount,
        tombstonedCount: 0,
        quarantinedCount: 0,
        supersededCount: 0,
        suppressedCount: 0,
        memoryEpoch: 0,
        index,
      };
    }
    const all = await this.canonical.listRecords({ includeInactive: true });
    const suppressions = await this.canonical.listSuppressions();
    return {
      canonicalEnabled: true,
      activeCount: all.filter((r) => r.state === "active").length,
      tombstonedCount: all.filter((r) => r.state === "tombstoned").length,
      quarantinedCount: all.filter((r) => r.state === "quarantined").length,
      supersededCount: all.filter((r) => r.state === "superseded").length,
      suppressedCount: suppressions.length,
      memoryEpoch: await this.canonical.memoryEpoch(),
      index,
    };
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

  /** G5.2 审核台数据源：候选结局列表（无 canonical 时空列表，不伪造）。 */
  async listMemoryCandidates(opts?: {
    status?: CandidateOutcomeRecord["status"];
    limit?: number;
  }): Promise<CandidateOutcomeRecord[]> {
    if (!this.canonical) return [];
    return this.canonical.listCandidateOutcomes(opts);
  }

  /**
   * G5.2 人工审核：只有 shadow_only（已校验、未生效）可被审核。
   * approve 由 **host 代码** 执行 mutation（add/supersede/retract），
   * writer/模型无权直接生效；supersede/retract 在 apply 时复查目标
   * revision，漂移则保持 shadow_only 并报冲突。reject 记 dismissed，
   * 与校验 rejected 区分（人工决定 vs 管线拒绝）。
   */
  async reviewMemoryCandidate(
    candidateId: string,
    decision: "approve" | "reject",
    principalId: string,
  ): Promise<{ candidateId: string; status: "applied" | "dismissed"; fragmentId?: string }> {
    if (!this.canonical) {
      throw new CandidateReviewError("canonical_unavailable", "reviewMemoryCandidate requires canonical repository");
    }
    const outcome = await this.canonical.candidateOutcome(candidateId);
    if (!outcome) {
      throw new CandidateReviewError("candidate_not_found", `candidate ${candidateId} not found`);
    }
    if (outcome.status !== "shadow_only") {
      throw new CandidateReviewError(
        "candidate_not_reviewable",
        `candidate ${candidateId} is ${outcome.status}, only shadow_only is reviewable`,
      );
    }
    if (decision === "reject") {
      await this.canonical.upsertCandidateOutcomes([
        { ...outcome, status: "dismissed", updatedAt: Date.now() },
      ]);
      return { candidateId, status: "dismissed" };
    }

    const candidate = JSON.parse(outcome.candidateJson) as MemoryCandidate;
    let fragmentId: string | undefined;
    if (candidate.operation === "add") {
      const frag = await this.addLongTermAsync(candidate.content, {
        kind: candidate.kind,
        ...(candidate.proposedKey ? { canonicalKey: candidate.proposedKey } : {}),
        ...(candidate.proposedValidity?.validUntilMs !== undefined
          ? { validUntil: candidate.proposedValidity.validUntilMs }
          : {}),
        metadata: { viaCandidate: candidateId, sourceRefs: [...candidate.sourceRefs] },
      });
      fragmentId = frag.fragmentId;
    } else if (candidate.operation === "supersede" || candidate.operation === "retract") {
      const target = await this.canonical.getRecord(candidate.targetRecordId ?? "");
      if (!target || target.revision !== candidate.expectedRevision) {
        throw new CandidateReviewError(
          "candidate_revision_conflict",
          `candidate ${candidateId} target revision drifted; re-propose required`,
        );
      }
      if (candidate.operation === "supersede") {
        const frag = await this.correctMemory(candidate.targetRecordId!, candidate.content, {
          kind: candidate.kind,
          ...(candidate.proposedKey ? { canonicalKey: candidate.proposedKey } : {}),
        });
        fragmentId = frag.fragmentId;
      } else {
        await this.forgetFragment(candidate.targetRecordId!, principalId, "candidate-retract-approved");
        fragmentId = candidate.targetRecordId;
      }
    }
    await this.canonical.upsertCandidateOutcomes([
      { ...outcome, status: "applied", updatedAt: Date.now() },
    ]);
    return { candidateId, status: "applied", ...(fragmentId ? { fragmentId } : {}) };
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
