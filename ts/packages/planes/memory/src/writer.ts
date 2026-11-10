import { createHash } from "node:crypto";
import { z } from "zod";
import type { CanonicalRecordState } from "./canonical.js";
import { type MemoryScope, isFragmentVisibleToScope } from "./scope.js";

/**
 * G5 / F13 契约：独立 writer 候选接口与校验管线。
 *
 * writer 只有"提名权"：它读 host 固定的 committed 证据与 scoped 记录快照，
 * 产出候选；候选不能自带 scope、不能自报身份（candidateId 由 host 按
 * extractionRevision/inputHash/ordinal 生成）、不能直接生效。本文件只做
 * 确定性校验（T26），不调用任何模型、不执行候选内容中的指令。
 */

export const MEMORY_KINDS = [
  "fact",
  "preference",
  "intention",
  "episode",
  "state_reference",
  "procedure",
  "note",
] as const;
export type MemoryKind = (typeof MEMORY_KINDS)[number];

export const CANDIDATE_OPERATIONS = ["add", "supersede", "retract", "noop"] as const;
export type CandidateOperation = (typeof CANDIDATE_OPERATIONS)[number];

/** 候选全生命周期；G5.0 只产出 validated/rejected/conflicted 三态。 */
export type CandidateStatus =
  | "proposed"
  | "validated"
  | "pending_approval"
  | "shadow_only"
  | "applied"
  | "rejected"
  | "conflicted"
  | "superseded"
  | "revoked";

/** host 签发的证据指针：指向已 committed 的轮次文本（span 为字符偏移）。 */
export interface EvidenceRef {
  evidenceId: string;
  journalId: string;
  commitId: string;
  turnId: string;
  spanStart?: number;
  spanEnd?: number;
  /** span 文本（未给 span 时为整段文本）的 sha256 hex。 */
  contentHash: string;
}

/** writer 实际看到的证据：ref + host 解析出的文本。校验时以 host 存储为准。 */
export interface EvidenceSpan extends EvidenceRef {
  text: string;
}

/** host 提供的当前记录快照（writer 只能引用，不能构造）。 */
export interface RecordSnapshot {
  fragmentId: string;
  revision: number;
  state: CanonicalRecordState;
  kind: string;
  assertion: string;
  canonicalKey?: string;
  scopeTags: string[];
}

export interface WriterInput {
  committedTurnRef: string;
  evidence: readonly EvidenceSpan[];
  scopedCurrentRecords: readonly RecordSnapshot[];
  allowedKinds: readonly string[];
  policyVersion: string;
  inputHash: string;
  memoryEpoch: number;
  extractionRevision: number;
}

export interface WriterProposalBatch {
  /** 未受信原始输出：逐个过严格 schema 与校验管线，绝不直接采信。 */
  candidates: readonly unknown[];
}

export interface MemoryWriter {
  readonly writerId: string;
  propose(input: WriterInput): Promise<WriterProposalBatch>;
}

/** 候选 wire 格式：strict —— 未知字段（包括试图自带 scope）一律拒绝。 */
export const MemoryCandidateSchema = z
  .object({
    candidateId: z.string().min(1).max(128).optional(),
    operation: z.enum(CANDIDATE_OPERATIONS),
    kind: z.enum(MEMORY_KINDS),
    content: z.string().min(1).max(2_000),
    sourceRefs: z.array(z.string().min(1).max(128)).max(8),
    proposedKey: z.string().min(1).max(200).optional(),
    targetRecordId: z.string().min(1).max(128).optional(),
    expectedRevision: z.number().int().nonnegative().optional(),
    proposedValidity: z
      .object({
        validFromMs: z.number().int().nonnegative().optional(),
        validUntilMs: z.number().int().nonnegative().optional(),
      })
      .strict()
      .optional(),
    rationale: z.string().max(500),
  })
  .strict();

export type MemoryCandidate = z.infer<typeof MemoryCandidateSchema> & { candidateId: string };

export const hashEvidenceText = (text: string): string =>
  createHash("sha256").update(text, "utf8").digest("hex");

/** host 生成候选身份：模型自报 candidateId 一律被覆盖（T27 协议一致性基础）。 */
export const hostCandidateId = (extractionRevision: number, inputHash: string, ordinal: number): string =>
  `cand_${createHash("sha256")
    .update(`${extractionRevision}:${inputHash}:${ordinal}`, "utf8")
    .digest("hex")
    .slice(0, 24)}`;

export type CandidateRejectionCode =
  | "SCHEMA_INVALID"
  | "STALE_EPOCH"
  | "NO_SOURCE"
  | "EVIDENCE_NOT_FOUND"
  | "EVIDENCE_SPAN_INVALID"
  | "EVIDENCE_HASH_MISMATCH"
  | "KIND_NOT_ALLOWED"
  | "SENSITIVE_CONTENT"
  | "SCOPE_VIOLATION"
  | "USER_VALUE_PROTECTED"
  | "MISSING_TARGET"
  | "TARGET_NOT_FOUND"
  | "TARGET_SUPPRESSED"
  | "MISSING_EXPECTED_REVISION"
  | "VALIDITY_INVALID";

export type CandidateConflictCode = "REVISION_CONFLICT";

export interface CandidateValidation {
  /** host 身份（与模型自报无关）。 */
  candidateId: string;
  status: "validated" | "rejected" | "conflicted";
  reasons: readonly string[];
  /** validated 时的规范化候选（host id 已填入）；内容仅作数据保存，永不执行。 */
  candidate?: MemoryCandidate;
}

export interface CandidateValidationContext {
  /** host 固定的 scope；候选无权扩大（自带 scope 字段会被 strict schema 拒绝）。 */
  scope: MemoryScope;
  currentEpoch: number;
  allowedKinds: readonly string[];
  /** 以 host 存储解析证据：返回 ref 与权威文本；writer 自带文本不作数。 */
  resolveEvidence: (evidenceId: string) => EvidenceSpan | undefined;
  recordsById: ReadonlyMap<string, RecordSnapshot>;
  /** 有 live user 断言占位的 canonicalKey 集合。 */
  userOwnedKeys: ReadonlySet<string>;
  isSuppressed: (fragmentId: string) => boolean;
}

/** 保守的秘密/敏感模式：命中即拒绝，不做"脱敏后通过"（T25 输入脱敏口径）。 */
const SENSITIVE_PATTERNS: readonly RegExp[] = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
  /\bsk-[A-Za-z0-9_-]{16,}/,
  /\bghp_[A-Za-z0-9]{20,}/,
  /\bAKID[A-Za-z0-9]{10,}/,
  /\bxox[baprs]-[A-Za-z0-9-]{10,}/,
  /\b(?:password|passwd|api[_-]?key|secret|access[_-]?token)\s*[:=]\s*\S+/i,
];

const containsSensitive = (content: string): boolean =>
  SENSITIVE_PATTERNS.some((pattern) => pattern.test(content));

const rejected = (candidateId: string, code: CandidateRejectionCode | CandidateConflictCode, conflicted = false): CandidateValidation => ({
  candidateId,
  status: conflicted ? "conflicted" : "rejected",
  reasons: [code],
});

/**
 * 单候选校验（epoch 在批次层校验）。顺序确定：schema → noop 短路 →
 * 来源存在/span/hash → kind 白名单 → 敏感 → 目标与 revision → 用户值优先。
 * 任一硬失败即 rejected 并给出稳定原因码；revision 不匹配是 conflicted
 * （可重读后重提），其余拒绝不可重试。
 */
export const validateCandidate = (
  raw: unknown,
  ordinal: number,
  input: WriterInput,
  ctx: CandidateValidationContext,
): CandidateValidation => {
  const candidateId = hostCandidateId(input.extractionRevision, input.inputHash, ordinal);
  const parsed = MemoryCandidateSchema.safeParse(raw);
  if (!parsed.success) return rejected(candidateId, "SCHEMA_INVALID");
  const c = parsed.data;

  if (c.operation === "noop") {
    return { candidateId, status: "validated", reasons: [], candidate: { ...c, candidateId } };
  }

  if (c.sourceRefs.length === 0) return rejected(candidateId, "NO_SOURCE");
  for (const refId of c.sourceRefs) {
    const span = ctx.resolveEvidence(refId);
    if (!span) return rejected(candidateId, "EVIDENCE_NOT_FOUND");
    const start = span.spanStart ?? 0;
    const end = span.spanEnd ?? span.text.length;
    if (start < 0 || end < start || end > span.text.length) {
      return rejected(candidateId, "EVIDENCE_SPAN_INVALID");
    }
    if (hashEvidenceText(span.text.slice(start, end)) !== span.contentHash) {
      return rejected(candidateId, "EVIDENCE_HASH_MISMATCH");
    }
  }

  if (!ctx.allowedKinds.includes(c.kind)) return rejected(candidateId, "KIND_NOT_ALLOWED");
  if (containsSensitive(c.content)) return rejected(candidateId, "SENSITIVE_CONTENT");

  if (c.proposedValidity?.validFromMs !== undefined && c.proposedValidity?.validUntilMs !== undefined) {
    if (c.proposedValidity.validUntilMs <= c.proposedValidity.validFromMs) {
      return rejected(candidateId, "VALIDITY_INVALID");
    }
  }

  let target: RecordSnapshot | undefined;
  if (c.operation === "supersede" || c.operation === "retract") {
    if (!c.targetRecordId) return rejected(candidateId, "MISSING_TARGET");
    target = ctx.recordsById.get(c.targetRecordId);
    if (!target) return rejected(candidateId, "TARGET_NOT_FOUND");
    if (ctx.isSuppressed(target.fragmentId) || target.state === "tombstoned") {
      return rejected(candidateId, "TARGET_SUPPRESSED");
    }
    if (!isFragmentVisibleToScope({ taskTags: target.scopeTags }, ctx.scope)) {
      return rejected(candidateId, "SCOPE_VIOLATION");
    }
    if (c.expectedRevision === undefined) return rejected(candidateId, "MISSING_EXPECTED_REVISION");
    if (c.expectedRevision !== target.revision) {
      return rejected(candidateId, "REVISION_CONFLICT", true);
    }
  }

  // 用户显式值优先：user 占位的 key / user 断言的目标，不接受 writer 改写。
  if (c.proposedKey && ctx.userOwnedKeys.has(c.proposedKey)) {
    return rejected(candidateId, "USER_VALUE_PROTECTED");
  }
  if (target && target.assertion === "user") {
    return rejected(candidateId, "USER_VALUE_PROTECTED");
  }
  if (target?.canonicalKey && ctx.userOwnedKeys.has(target.canonicalKey) && target.assertion !== "user") {
    return rejected(candidateId, "USER_VALUE_PROTECTED");
  }

  return { candidateId, status: "validated", reasons: [], candidate: { ...c, candidateId } };
};

/**
 * 批次校验：先查 epoch（过期 epoch 整批拒绝，迟到 writer 不生效），
 * 再逐个 validateCandidate。返回与输入等长的逐项结论，顺序一致。
 */
export const validateCandidateBatch = (
  batch: WriterProposalBatch,
  input: WriterInput,
  ctx: CandidateValidationContext,
): CandidateValidation[] => {
  if (input.memoryEpoch !== ctx.currentEpoch) {
    return batch.candidates.map((_, ordinal) => ({
      candidateId: hostCandidateId(input.extractionRevision, input.inputHash, ordinal),
      status: "rejected" as const,
      reasons: ["STALE_EPOCH"],
    }));
  }
  return batch.candidates.map((raw, ordinal) => validateCandidate(raw, ordinal, input, ctx));
};

/**
 * 确定性 fixture writer（T26/T27 协议验证用，不接真实模型）：
 * 只认显式标记 —— "记住：/记住:" → fact，"我偏好/我喜欢/I prefer" → preference。
 * 无标记则零候选。输出不带 candidateId（host 生成），同输入同输出。
 */
export const createFixtureWriter = (): MemoryWriter => ({
  writerId: "fixture",
  async propose(input: WriterInput): Promise<WriterProposalBatch> {
    const candidates: unknown[] = [];
    for (const span of input.evidence) {
      for (const rawLine of span.text.split(/\r?\n/)) {
        const line = rawLine.trim();
        const remember = /^(?:记住[：:]|remember:)\s*(.+)$/i.exec(line);
        if (remember) {
          candidates.push({
            operation: "add",
            kind: "fact",
            content: remember[1]!.trim(),
            sourceRefs: [span.evidenceId],
            rationale: "fixture: explicit remember marker",
          });
          continue;
        }
        const prefer = /(?:我偏好|我喜欢|I prefer)\s*(.+)$/.exec(line);
        if (prefer) {
          candidates.push({
            operation: "add",
            kind: "preference",
            content: prefer[1]!.trim(),
            sourceRefs: [span.evidenceId],
            rationale: "fixture: explicit preference marker",
          });
        }
      }
    }
    return { candidates };
  },
});
