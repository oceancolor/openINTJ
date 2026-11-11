import type { MemoryType } from "@openintj/core";

/**
 * G2 / F05–F06 契约：canonical memory 仓储。
 *
 * SQLite 是权威源：正文、scope、来源、状态、修订都在这里；向量库只是可重建投影。
 * 所有晋升/清除/纠正/删除必须走 applyMutation 的单事务边界：
 * 记录变更 + mutation 收据 + index outbox 在同一事务内提交，禁止内存先变、持久层后补。
 */

export type CanonicalRecordState = "active" | "tombstoned" | "quarantined" | "superseded";

export interface CanonicalMemoryRecord {
  fragmentId: string;
  content: string;
  memoryType: MemoryType;
  importance: number;
  taskTags: string[];
  /** G1 scope 标签（workspace:/task:/conversation: 等），随正文一起权威保存。 */
  scopeTags: string[];
  metadata: Record<string, unknown>;
  summaries: Record<number, string>;
  /** 写入时的 embedding 快照（索引重建/重放靠它，不依赖当时的 embedder）。 */
  embedding: number[];
  source: string;
  state: CanonicalRecordState;
  revision: number;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
  // ---- F08 生命周期 ----
  kind: string;
  assertion: string;
  validUntil?: number;
  canonicalKey?: string;
  supersededBy?: string;
}

export type MutationKind = "upsert" | "delete" | "clear_tier" | "reassign" | "mark_superseded";

export interface MutationRequest {
  /** 幂等写键：同键同 inputHash 重放返回原收据；同键异 hash 是冲突。 */
  writeKey: string;
  inputHash: string;
  kind: MutationKind;
  fragmentId?: string;
  record?: CanonicalMemoryRecord;
  /** clear_tier 用。 */
  tier?: MemoryType;
  /** reassign 用。 */
  newType?: MemoryType;
  /** mark_superseded 用：取代者 fragmentId。 */
  supersededBy?: string;
  /** CAS：期望的当前 revision；不匹配是冲突，零写入。 */
  expectedRevision?: number;
  /** upsert 时的初始状态（迁移隔离用 quarantined）。缺省 active。 */
  initialState?: CanonicalRecordState;
}

export interface MutationReceipt {
  writeKey: string;
  outcome: "applied" | "replayed";
  fragmentId?: string;
  revision?: number;
  appliedAt: number;
}

export class MutationConflictError extends Error {
  readonly code = "MEMORY_MUTATION_CONFLICT";
  constructor(message: string) {
    super(message);
    this.name = "MutationConflictError";
  }
}

export interface IndexJob {
  jobId: number;
  fragmentId: string;
  kind: "upsert" | "delete";
  revision: number;
  generation: number;
  attempts: number;
  lastError?: string;
  createdAt: number;
}

export interface CandidateBatchReceipt {
  receiveKey: string;
  outcome: "applied" | "replayed";
  candidateCount: number;
  receivedAt: number;
}

export class CandidateConflictError extends Error {
  readonly code = "MEMORY_CANDIDATE_CONFLICT";
  constructor(message: string) {
    super(message);
    this.name = "CandidateConflictError";
  }
}

/**
 * 候选批次接收键 = (sourceJournalId, sourceCommitId, consumer, policyVersion, extractionRevision)。
 * G2 只验接收边界（固定 fixture），不接真实 writer；候选只入库待审，绝不直接激活。
 */
export interface CandidateReceiveKey {
  sourceJournalId: string;
  sourceCommitId: string;
  consumer: string;
  policyVersion: string;
  extractionRevision: number;
}

export const candidateReceiveKeyString = (k: CandidateReceiveKey): string =>
  [k.sourceJournalId, k.sourceCommitId, k.consumer, k.policyVersion, String(k.extractionRevision)].join(":");

/**
 * G5.1 影子管线：单个候选的校验结局（与批次接收分开存——批次 ACK 只确认
 * 完整接收，结局供审核台/对照实验读取）。candidateJson 为规范化候选
 * （validated）或原始候选（rejected/conflicted，留作诊断）。
 */
export interface CandidateOutcomeRecord {
  candidateId: string;
  receiveKey: string;
  /** applied/dismissed 为 G5.2 人工审核终态（仅 shadow_only 可被审核）。 */
  status: "shadow_only" | "rejected" | "conflicted" | "applied" | "dismissed";
  reasons: string[];
  candidateJson: string;
  updatedAt: number;
}

/**
 * 影子期轻量可观测：按 policyVersion 聚合的 writer 处理计数。
 * 候选结局只覆盖"提名了什么"；这组计数补上分母（处理了多少轮、
 * 多少轮空转、多少死信），供状态面与激活决策门复盘。replayed 的
 * 幂等重投不重复计数。
 */
export interface WriterPipelineStats {
  policyVersion: string;
  processedTurns: number;
  emptyTurns: number;
  deadLetters: number;
  candidatesNominated: number;
  startedAt: number;
  updatedAt: number;
}

export interface CanonicalMemoryRepository {
  init(): Promise<void>;
  close(): Promise<void>;

  /** 全部记录（可含 tombstone/quarantine，供对账与管理面）。 */
  listRecords(opts?: { includeInactive?: boolean }): Promise<CanonicalMemoryRecord[]>;
  getRecord(fragmentId: string): Promise<CanonicalMemoryRecord | undefined>;

  /** 唯一 mutation 边界：单事务 = 记录变更 + 收据 + outbox。 */
  applyMutation(req: MutationRequest): Promise<MutationReceipt>;

  /** 幂等接收候选批次（只存待审，不激活）。 */
  receiveCandidates(input: {
    key: CandidateReceiveKey;
    inputHash: string;
    candidates: readonly Record<string, unknown>[];
  }): Promise<CandidateBatchReceipt>;
  candidateBatchStatus(
    key: CandidateReceiveKey,
  ): Promise<{ status: string; candidateCount: number } | undefined>;

  /** G5.1：按 candidateId upsert 候选结局（重放同批次不重复计数）。 */
  upsertCandidateOutcomes(records: readonly CandidateOutcomeRecord[]): Promise<void>;
  listCandidateOutcomes(opts?: {
    status?: CandidateOutcomeRecord["status"];
    limit?: number;
  }): Promise<CandidateOutcomeRecord[]>;
  candidateOutcome(candidateId: string): Promise<CandidateOutcomeRecord | undefined>;

  /** 影子管线结算计数（单调累加，startedAt 只在首记时定格）。 */
  recordWriterTurn(stat: {
    policyVersion: string;
    empty: boolean;
    deadLetter: boolean;
    candidates: number;
    at: number;
  }): Promise<void>;
  writerPipelineStats(policyVersion: string): Promise<WriterPipelineStats | undefined>;
  listWriterPipelineStats(): Promise<WriterPipelineStats[]>;

  pendingIndexJobs(limit?: number): Promise<IndexJob[]>;
  markIndexJobApplied(jobId: number): Promise<void>;
  markIndexJobFailed(jobId: number, error: string): Promise<void>;
  indexStats(): Promise<{ pending: number; failed: number; generation: number }>;
  /** 开启新索引代次（空间不兼容/重建时），返回新 generation。 */
  startIndexGeneration(spaceFingerprint: string): Promise<number>;
  indexGenerationFingerprint(): Promise<string | undefined>;

  // ---- F12 P0：control ledger ----
  memoryEpoch(): Promise<number>;
  /** forget 类操作递增 epoch；旧 epoch 下的派生索引/缓存必须失效。 */
  bumpMemoryEpoch(): Promise<number>;
  addSuppression(fragmentId: string, principalId: string, reason: string): Promise<void>;
  isSuppressed(fragmentId: string): Promise<boolean>;
  listSuppressions(): Promise<Array<{ fragmentId: string; principalId: string; reason: string }>>;
  /**
   * F12 原子遗忘：suppression + tombstone + epoch 递增在同一事务提交，
   * 迟到 writer/旧 outbox 不能让被遗忘内容回潮。返回新 epoch。
   */
  applyForget(input: { fragmentId: string; principalId: string; reason: string }): Promise<{ epoch: number }>;
  /**
   * F12 purge（物理删除基础版）：删记录本体与未投影 outbox payload；
   * suppression barrier 保留（不含正文）防回潮。备份/导出副本不在此承诺内。
   */
  purgeFragment(fragmentId: string): Promise<{ recordDeleted: boolean; pendingJobsPurged: number }>;
}

/** 测试用内存实现：与 SQLite 实现共享语义（幂等/CAS/outbox 同事务近似）。 */
export class InMemoryCanonicalMemoryRepository implements CanonicalMemoryRepository {
  private records = new Map<string, CanonicalMemoryRecord>();
  private mutations = new Map<string, { inputHash: string; receipt: MutationReceipt }>();
  private candidates = new Map<string, { inputHash: string; status: string; count: number; receivedAt: number }>();
  private candidateOutcomes = new Map<string, CandidateOutcomeRecord>();
  private writerStats = new Map<string, WriterPipelineStats>();
  private jobs: IndexJob[] = [];
  private nextJobId = 1;
  private generation = 1;
  private generationFingerprint: string | undefined;
  private epoch = 1;
  private suppressions = new Map<string, { fragmentId: string; principalId: string; reason: string }>();

  async init(): Promise<void> {}
  async close(): Promise<void> {}

  async listRecords(opts?: { includeInactive?: boolean }): Promise<CanonicalMemoryRecord[]> {
    const all = [...this.records.values()];
    return opts?.includeInactive ? all : all.filter((r) => r.state === "active");
  }

  async getRecord(fragmentId: string): Promise<CanonicalMemoryRecord | undefined> {
    return this.records.get(fragmentId);
  }

  async applyMutation(req: MutationRequest): Promise<MutationReceipt> {
    const prior = this.mutations.get(req.writeKey);
    if (prior) {
      if (prior.inputHash !== req.inputHash) {
        throw new MutationConflictError(`writeKey ${req.writeKey} already used with different input`);
      }
      return { ...prior.receipt, outcome: "replayed" };
    }
    const now = Date.now();
    let fragmentId = req.fragmentId ?? req.record?.fragmentId;
    let revision: number | undefined;
    const enqueue = (fid: string, kind: "upsert" | "delete", rev: number) => {
      this.jobs.push({
        jobId: this.nextJobId++,
        fragmentId: fid,
        kind,
        revision: rev,
        generation: this.generation,
        attempts: 0,
        createdAt: now,
      });
    };
    if (req.kind === "upsert") {
      if (!req.record) throw new Error("upsert requires record");
      const existing = this.records.get(req.record.fragmentId);
      if (req.expectedRevision !== undefined && (existing?.revision ?? 0) !== req.expectedRevision) {
        throw new MutationConflictError(
          `CAS failed for ${req.record.fragmentId}: expected ${req.expectedRevision}, actual ${existing?.revision ?? 0}`,
        );
      }
      revision = (existing?.revision ?? 0) + 1;
      this.records.set(req.record.fragmentId, {
        ...req.record,
        state: existing?.state ?? req.initialState ?? "active",
        revision,
      });
      enqueue(req.record.fragmentId, "upsert", revision);
      fragmentId = req.record.fragmentId;
    } else if (req.kind === "delete") {
      if (!fragmentId) throw new Error("delete requires fragmentId");
      const existing = this.records.get(fragmentId);
      if (existing) {
        revision = existing.revision + 1;
        this.records.set(fragmentId, { ...existing, state: "tombstoned", revision });
        enqueue(fragmentId, "delete", revision);
      }
    } else if (req.kind === "clear_tier") {
      if (!req.tier) throw new Error("clear_tier requires tier");
      for (const r of [...this.records.values()]) {
        if (r.memoryType === req.tier && r.state === "active") {
          const rev = r.revision + 1;
          this.records.set(r.fragmentId, { ...r, state: "tombstoned", revision: rev });
          enqueue(r.fragmentId, "delete", rev);
        }
      }
    } else if (req.kind === "reassign") {
      if (!fragmentId || !req.newType) throw new Error("reassign requires fragmentId+newType");
      const existing = this.records.get(fragmentId);
      if (existing) {
        revision = existing.revision + 1;
        this.records.set(fragmentId, { ...existing, memoryType: req.newType, revision });
        enqueue(fragmentId, "upsert", revision);
      }
    } else if (req.kind === "mark_superseded") {
      if (!fragmentId || !req.supersededBy) throw new Error("mark_superseded requires fragmentId+supersededBy");
      const existing = this.records.get(fragmentId);
      if (existing) {
        revision = existing.revision + 1;
        this.records.set(fragmentId, {
          ...existing,
          state: "superseded",
          supersededBy: req.supersededBy,
          revision,
        });
        enqueue(fragmentId, "delete", revision);
      }
    }
    const receipt: MutationReceipt = {
      writeKey: req.writeKey,
      outcome: "applied",
      ...(fragmentId ? { fragmentId } : {}),
      ...(revision !== undefined ? { revision } : {}),
      appliedAt: now,
    };
    this.mutations.set(req.writeKey, { inputHash: req.inputHash, receipt });
    return receipt;
  }

  async receiveCandidates(input: {
    key: CandidateReceiveKey;
    inputHash: string;
    candidates: readonly Record<string, unknown>[];
  }): Promise<CandidateBatchReceipt> {
    const keyStr = candidateReceiveKeyString(input.key);
    const prior = this.candidates.get(keyStr);
    if (prior) {
      if (prior.inputHash !== input.inputHash) {
        throw new CandidateConflictError(`candidate batch ${keyStr} redelivered with different input`);
      }
      return { receiveKey: keyStr, outcome: "replayed", candidateCount: prior.count, receivedAt: prior.receivedAt };
    }
    const receivedAt = Date.now();
    this.candidates.set(keyStr, {
      inputHash: input.inputHash,
      status: "received",
      count: input.candidates.length,
      receivedAt,
    });
    return { receiveKey: keyStr, outcome: "applied", candidateCount: input.candidates.length, receivedAt };
  }

  async candidateBatchStatus(
    key: CandidateReceiveKey,
  ): Promise<{ status: string; candidateCount: number } | undefined> {
    const found = this.candidates.get(candidateReceiveKeyString(key));
    return found ? { status: found.status, candidateCount: found.count } : undefined;
  }

  async upsertCandidateOutcomes(records: readonly CandidateOutcomeRecord[]): Promise<void> {
    for (const record of records) this.candidateOutcomes.set(record.candidateId, { ...record });
  }

  async listCandidateOutcomes(opts?: {
    status?: CandidateOutcomeRecord["status"];
    limit?: number;
  }): Promise<CandidateOutcomeRecord[]> {
    const all = [...this.candidateOutcomes.values()].filter(
      (r) => !opts?.status || r.status === opts.status,
    );
    return (opts?.limit ? all.slice(0, opts.limit) : all).map((r) => ({ ...r }));
  }

  async candidateOutcome(candidateId: string): Promise<CandidateOutcomeRecord | undefined> {
    const found = this.candidateOutcomes.get(candidateId);
    return found ? { ...found } : undefined;
  }

  async recordWriterTurn(stat: {
    policyVersion: string;
    empty: boolean;
    deadLetter: boolean;
    candidates: number;
    at: number;
  }): Promise<void> {
    const prior = this.writerStats.get(stat.policyVersion);
    this.writerStats.set(stat.policyVersion, {
      policyVersion: stat.policyVersion,
      processedTurns: (prior?.processedTurns ?? 0) + 1,
      emptyTurns: (prior?.emptyTurns ?? 0) + (stat.empty ? 1 : 0),
      deadLetters: (prior?.deadLetters ?? 0) + (stat.deadLetter ? 1 : 0),
      candidatesNominated: (prior?.candidatesNominated ?? 0) + stat.candidates,
      startedAt: prior?.startedAt ?? stat.at,
      updatedAt: stat.at,
    });
  }

  async writerPipelineStats(policyVersion: string): Promise<WriterPipelineStats | undefined> {
    const found = this.writerStats.get(policyVersion);
    return found ? { ...found } : undefined;
  }

  async listWriterPipelineStats(): Promise<WriterPipelineStats[]> {
    return [...this.writerStats.values()].map((s) => ({ ...s }));
  }

  async pendingIndexJobs(limit = 100): Promise<IndexJob[]> {
    return this.jobs.slice(0, limit).map((j) => ({ ...j }));
  }

  async markIndexJobApplied(jobId: number): Promise<void> {
    this.jobs = this.jobs.filter((j) => j.jobId !== jobId);
  }

  async markIndexJobFailed(jobId: number, error: string): Promise<void> {
    const job = this.jobs.find((j) => j.jobId === jobId);
    if (job) {
      job.attempts += 1;
      job.lastError = error;
    }
  }

  async indexStats(): Promise<{ pending: number; failed: number; generation: number }> {
    return {
      pending: this.jobs.length,
      failed: this.jobs.filter((j) => j.attempts > 0).length,
      generation: this.generation,
    };
  }

  async startIndexGeneration(spaceFingerprint: string): Promise<number> {
    this.generation += 1;
    this.generationFingerprint = spaceFingerprint;
    return this.generation;
  }

  async indexGenerationFingerprint(): Promise<string | undefined> {
    return this.generationFingerprint;
  }

  async memoryEpoch(): Promise<number> {
    return this.epoch;
  }

  async bumpMemoryEpoch(): Promise<number> {
    this.epoch += 1;
    return this.epoch;
  }

  async addSuppression(fragmentId: string, principalId: string, reason: string): Promise<void> {
    this.suppressions.set(fragmentId, { fragmentId, principalId, reason });
  }

  async isSuppressed(fragmentId: string): Promise<boolean> {
    return this.suppressions.has(fragmentId);
  }

  async listSuppressions(): Promise<Array<{ fragmentId: string; principalId: string; reason: string }>> {
    return [...this.suppressions.values()];
  }

  async applyForget(input: {
    fragmentId: string;
    principalId: string;
    reason: string;
  }): Promise<{ epoch: number }> {
    await this.addSuppression(input.fragmentId, input.principalId, input.reason);
    const existing = this.records.get(input.fragmentId);
    if (existing) {
      const revision = existing.revision + 1;
      this.records.set(input.fragmentId, { ...existing, state: "tombstoned", revision });
      this.jobs.push({
        jobId: this.nextJobId++,
        fragmentId: input.fragmentId,
        kind: "delete",
        revision,
        generation: this.generation,
        attempts: 0,
        createdAt: Date.now(),
      });
    }
    this.epoch += 1;
    return { epoch: this.epoch };
  }

  async purgeFragment(
    fragmentId: string,
  ): Promise<{ recordDeleted: boolean; pendingJobsPurged: number }> {
    const recordDeleted = this.records.delete(fragmentId);
    const before = this.jobs.length;
    this.jobs = this.jobs.filter((j) => j.fragmentId !== fragmentId);
    return { recordDeleted, pendingJobsPurged: before - this.jobs.length };
  }
}
