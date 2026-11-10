import {
  type CandidateOutcomeRecord,
  type CandidateReceiveKey,
  type CanonicalMemoryRepository,
  CandidateConflictError,
  candidateReceiveKeyString,
} from "./canonical.js";
import {
  type CandidateValidation,
  type CandidateValidationContext,
  type MemoryWriter,
  type RecordSnapshot,
  type WriterInput,
  validateCandidateBatch,
} from "./writer.js";
import type { MemoryScope } from "./scope.js";

/**
 * host 标准校验上下文：以 canonical 当前状态构造（记录快照全集、
 * user 占位 key、suppression 集合、当前 epoch），证据以本次 input
 * 的 host 签发 span 解析。G5.2 桌面审核台与后续宿主共用。
 */
export const buildCanonicalValidationContext = async (
  repository: CanonicalMemoryRepository,
  input: WriterInput,
  scope: MemoryScope,
): Promise<CandidateValidationContext> => {
  const records = await repository.listRecords({ includeInactive: true });
  const recordsById = new Map<string, RecordSnapshot>(
    records.map((r) => [
      r.fragmentId,
      {
        fragmentId: r.fragmentId,
        revision: r.revision,
        state: r.state,
        kind: r.kind,
        assertion: r.assertion,
        ...(r.canonicalKey ? { canonicalKey: r.canonicalKey } : {}),
        scopeTags: [...r.scopeTags],
      },
    ]),
  );
  const userOwnedKeys = new Set(
    records
      .filter((r) => r.state === "active" && r.assertion === "user" && r.canonicalKey)
      .map((r) => r.canonicalKey!),
  );
  const suppressions = await repository.listSuppressions();
  const suppressed = new Set(suppressions.map((s) => s.fragmentId));
  return {
    scope,
    currentEpoch: await repository.memoryEpoch(),
    allowedKinds: [...input.allowedKinds],
    resolveEvidence: (id) => input.evidence.find((s) => s.evidenceId === id),
    recordsById,
    userOwnedKeys,
    isSuppressed: (fragmentId) => suppressed.has(fragmentId),
  };
};

/**
 * G5.1 / F13 影子管线：committed 轮次 → writer 提名 → G5.0 校验 →
 * validated 候选只以 shadow_only 落盘（经 G2 批次接收通道持久化），
 * 永不 apply、永不建索引投影、永不触碰 Profile/Persona/技能/回答。
 *
 * 可靠性口径：单 worker 串行；writer 异常以**同一 extractionRevision**
 * 有限重试；超预算不重试；死信带明确原因，不无限自动修复生成。
 * ACK（shadow_recorded）只确认"批次完整接收 + 结局已记录"，不代表生效。
 */

export interface WriterShadowJob {
  key: CandidateReceiveKey;
  input: WriterInput;
  /** host 固定 scope（候选无权决定）；缺省为空 scope（仅用户级可见）。 */
  scope?: MemoryScope;
}

export type ShadowDeadLetterReason = "WRITER_FAILED" | "OVER_BUDGET" | "RECEIVE_CONFLICT";

export type ShadowJobResult =
  | {
      status: "shadow_recorded";
      receiveKey: string;
      validated: number;
      rejected: number;
      conflicted: number;
      attempts: number;
    }
  | { status: "replayed"; receiveKey: string }
  | { status: "dead_letter"; receiveKey: string; reason: ShadowDeadLetterReason; attempts: number; error?: string };

export interface ShadowWriterPipelineOptions {
  repository: CanonicalMemoryRepository;
  writer: MemoryWriter;
  /** host 按当前 canonical 状态构造校验上下文（证据解析/记录快照/epoch）。 */
  buildContext: (input: WriterInput, job: WriterShadowJob) => CandidateValidationContext | Promise<CandidateValidationContext>;
  maxCandidatesPerBatch?: number;
  maxEvidenceChars?: number;
  /** 含首次尝试的总尝试次数（同 extractionRevision 重试）。 */
  maxAttempts?: number;
  now?: () => number;
}

const DEFAULT_MAX_CANDIDATES = 20;
const DEFAULT_MAX_EVIDENCE_CHARS = 24_000;
const DEFAULT_MAX_ATTEMPTS = 3;

export class ShadowWriterPipeline {
  private readonly opts: Required<
    Pick<ShadowWriterPipelineOptions, "maxCandidatesPerBatch" | "maxEvidenceChars" | "maxAttempts">
  > &
    ShadowWriterPipelineOptions;
  private chain: Promise<unknown> = Promise.resolve();
  private deadLetters: ShadowJobResult[] = [];

  constructor(options: ShadowWriterPipelineOptions) {
    this.opts = {
      maxCandidatesPerBatch: DEFAULT_MAX_CANDIDATES,
      maxEvidenceChars: DEFAULT_MAX_EVIDENCE_CHARS,
      maxAttempts: DEFAULT_MAX_ATTEMPTS,
      ...options,
    };
  }

  /** 入队即返回该 job 的结算 Promise；内部串行链保证单 worker 语义。 */
  enqueue(job: WriterShadowJob): Promise<ShadowJobResult> {
    const run = this.chain.then(() => this.process(job));
    // 链上单 job 失败不毒化后续 job（process 内部已收口为 ShadowJobResult）。
    this.chain = run.catch(() => undefined);
    return run;
  }

  /** 已死信 job 的结算记录（原因明确，供诊断/人工处理，不自动重投）。 */
  listDeadLetters(): readonly ShadowJobResult[] {
    return [...this.deadLetters];
  }

  private async process(job: WriterShadowJob): Promise<ShadowJobResult> {
    const receiveKey = candidateReceiveKeyString(job.key);
    const { repository, writer } = this.opts;

    const evidenceChars = job.input.evidence.reduce((sum, span) => sum + span.text.length, 0);
    if (evidenceChars > this.opts.maxEvidenceChars || job.input.evidence.length === 0) {
      return this.deadLetter({
        status: "dead_letter",
        receiveKey,
        reason: "OVER_BUDGET",
        attempts: 0,
      });
    }

    let attempts = 0;
    let batch: Awaited<ReturnType<MemoryWriter["propose"]>> | undefined;
    let lastError: unknown;
    while (attempts < this.opts.maxAttempts) {
      attempts += 1;
      try {
        // 重试复用同一 input（同 extractionRevision/同 inputHash），不换版本偷跑。
        batch = await writer.propose(job.input);
        break;
      } catch (err) {
        lastError = err;
      }
    }
    if (!batch) {
      return this.deadLetter({
        status: "dead_letter",
        receiveKey,
        reason: "WRITER_FAILED",
        attempts,
        error: lastError instanceof Error ? lastError.message : String(lastError),
      });
    }
    if (batch.candidates.length > this.opts.maxCandidatesPerBatch) {
      return this.deadLetter({ status: "dead_letter", receiveKey, reason: "OVER_BUDGET", attempts });
    }

    const ctx = await this.opts.buildContext(job.input, job);
    const validations = validateCandidateBatch(batch, job.input, ctx);

    let receipt: Awaited<ReturnType<CanonicalMemoryRepository["receiveCandidates"]>>;
    try {
      receipt = await repository.receiveCandidates({
        key: job.key,
        inputHash: job.input.inputHash,
        candidates: batch.candidates.map((c) =>
          typeof c === "object" && c !== null ? (c as Record<string, unknown>) : { value: c },
        ),
      });
    } catch (err) {
      if (err instanceof CandidateConflictError) {
        return this.deadLetter({ status: "dead_letter", receiveKey, reason: "RECEIVE_CONFLICT", attempts });
      }
      throw err;
    }
    if (receipt.outcome === "replayed") {
      // 同键同 hash 已完整接收过：ACK 幂等，不重复记结局、不重复计数。
      return { status: "replayed", receiveKey };
    }

    const now = this.opts.now ? this.opts.now() : Date.now();
    const outcomes: CandidateOutcomeRecord[] = validations.map((v: CandidateValidation, i: number) => ({
      candidateId: v.candidateId,
      receiveKey,
      status: v.status === "validated" ? "shadow_only" : v.status,
      reasons: [...v.reasons],
      candidateJson: JSON.stringify(v.candidate ?? batch.candidates[i] ?? null),
      updatedAt: now,
    }));
    await repository.upsertCandidateOutcomes(outcomes);

    return {
      status: "shadow_recorded",
      receiveKey,
      validated: outcomes.filter((o) => o.status === "shadow_only").length,
      rejected: outcomes.filter((o) => o.status === "rejected").length,
      conflicted: outcomes.filter((o) => o.status === "conflicted").length,
      attempts,
    };
  }

  private deadLetter(result: ShadowJobResult): ShadowJobResult {
    this.deadLetters.push(result);
    return result;
  }
}
