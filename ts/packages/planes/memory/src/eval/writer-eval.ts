import {
  type CandidateValidationContext,
  type MemoryWriter,
  type RecordSnapshot,
  type WriterInput,
  hashEvidenceText,
  validateCandidateBatch,
} from "../writer.js";

/**
 * G5.4 / §10 writer 质量对照：固定样本集 + 固定 reader/校验管线，
 * 只换 writer。指标口径：以 validated 候选为准（过 G5.0 校验才算
 * "会写出去"）——命中（该写的写对）/ 漏记 / 误存（不该写的写了，
 * 或写错内容）/ 正确空转。真实模型臂不在 CI 内跑（需模型服务），
 * 由 env-gated 的 live spec 手动执行并把结果记入 RFC。
 */

export type SampleExpectation =
  | { type: "write"; kind: string; contentIncludes: string }
  | { type: "supersede"; contentIncludes: string }
  | { type: "none" };

export interface WriterSample {
  id: string;
  category: string;
  userText: string;
  currentRecords?: RecordSnapshot[];
  suppressedIds?: string[];
  expect: SampleExpectation;
}

const record = (over: Partial<RecordSnapshot> & { fragmentId: string }): RecordSnapshot => ({
  revision: 1,
  state: "active",
  kind: "fact",
  assertion: "assistant",
  scopeTags: ["workspace:ws1"],
  ...over,
});

export const WRITER_SAMPLES: readonly WriterSample[] = [
  {
    id: "s01-explicit-preference",
    category: "明确长期偏好",
    userText: "我偏好深色主题，所有工具都给我设成深色",
    expect: { type: "write", kind: "preference", contentIncludes: "深色" },
  },
  {
    id: "s02-explicit-remember",
    category: "明确长期偏好",
    userText: "记住：我的项目截止日是每周五下午",
    expect: { type: "write", kind: "fact", contentIncludes: "截止日" },
  },
  {
    id: "s03-one-off",
    category: "仅当次偏好",
    userText: "今天下午三点提醒我开周会，就这一次",
    expect: { type: "none" },
  },
  {
    id: "s04-user-correction",
    category: "用户纠正",
    userText: "截止日改成下周一了，不是周五，记一下新的",
    currentRecords: [
      record({ fragmentId: "frag-deadline", canonicalKey: "project.deadline", revision: 3 }),
    ],
    expect: { type: "supersede", contentIncludes: "下周一" },
  },
  {
    id: "s05-assistant-echo",
    category: "助手回声",
    userText: "嗯，你说的对，就照你说的办吧",
    expect: { type: "none" },
  },
  {
    id: "s06-quoting-others",
    category: "引用他人",
    userText: "我同事老王喜欢用 Vim，天天劝我换过去",
    expect: { type: "none" },
  },
  {
    id: "s07-expired-intention",
    category: "过期意图",
    userText: "上周我本来打算去北京出差，后来取消了，不用管了",
    expect: { type: "none" },
  },
  {
    id: "s08-cross-project",
    category: "项目间冲突",
    userText: "这个项目我们改成每周四发布",
    currentRecords: [
      record({
        fragmentId: "frag-other-project",
        canonicalKey: "project.release-day",
        scopeTags: ["workspace:ws2"],
      }),
    ],
    expect: { type: "write", kind: "fact", contentIncludes: "周四" },
  },
  {
    id: "s09-failure-then-fix",
    category: "失败后改正",
    userText: "刚才说错了，我的联系邮箱是 ben@example.com，之前那个作废",
    currentRecords: [record({ fragmentId: "frag-email", canonicalKey: "user.email", revision: 2 })],
    expect: { type: "supersede", contentIncludes: "ben@example.com" },
  },
  {
    id: "s10-tool-vs-assistant",
    category: "工具返回与助手结论冲突",
    userText: "你刚说部署成功，但我看日志明明是失败的，以后这种事以日志为准",
    expect: { type: "none" },
  },
  {
    id: "s11-revoke-repropose",
    category: "撤销再提案",
    userText: "把之前忘掉的那件事重新记上吧：我现在又开始自己做饭了",
    currentRecords: [
      record({ fragmentId: "frag-cooking", state: "tombstoned", canonicalKey: "user.cooking" }),
    ],
    suppressedIds: ["frag-cooking"],
    expect: { type: "write", kind: "fact", contentIncludes: "做饭" },
  },
  {
    id: "s12-sensitive-secret",
    category: "敏感内容（校验层拦截）",
    userText: "记住：我的服务器密码是 password: hunter2-xyz",
    expect: { type: "none" },
  },
  {
    id: "s13-implicit-fact",
    category: "明确长期偏好",
    userText: "我家在成都，孩子也在这边上学，三年内不会搬家",
    expect: { type: "write", kind: "fact", contentIncludes: "成都" },
  },
  {
    id: "s14-implicit-preference",
    category: "明确长期偏好",
    userText: "以后回答我尽量简短一点，别长篇大论的",
    expect: { type: "write", kind: "preference", contentIncludes: "简短" },
  },
];

export interface SampleVerdict {
  sampleId: string;
  category: string;
  verdict: "hit" | "miss" | "false_write" | "correct_none";
  validatedCount: number;
  rejectedReasons: string[];
}

export interface WriterEvalReport {
  writerId: string;
  verdicts: SampleVerdict[];
  totals: {
    samples: number;
    expectedWrites: number;
    hits: number;
    misses: number;
    falseWrites: number;
    correctNones: number;
    /** 命中 / 应写样本（write+supersede）。 */
    recall: number;
    /** 正确处置（hit+correct_none）/ 全部样本。 */
    accuracy: number;
  };
}

export const sampleInput = (sample: WriterSample): WriterInput => ({
  committedTurnRef: `eval:${sample.id}`,
  evidence: [
    {
      evidenceId: `ev:${sample.id}`,
      journalId: "eval",
      commitId: `eval:${sample.id}`,
      turnId: `eval:${sample.id}`,
      contentHash: hashEvidenceText(sample.userText),
      text: sample.userText,
    },
  ],
  scopedCurrentRecords: sample.currentRecords ?? [],
  allowedKinds: ["fact", "preference", "intention", "episode", "state_reference", "procedure", "note"],
  policyVersion: "eval-v1",
  inputHash: `eval:${sample.id}`,
  memoryEpoch: 1,
  extractionRevision: 1,
});

export const sampleContext = (sample: WriterSample, input: WriterInput): CandidateValidationContext => {
  const records = sample.currentRecords ?? [];
  const suppressed = new Set(sample.suppressedIds ?? []);
  return {
    scope: { workspaceId: "ws1" },
    currentEpoch: 1,
    allowedKinds: [...input.allowedKinds],
    resolveEvidence: (id) => input.evidence.find((s) => s.evidenceId === id),
    recordsById: new Map(records.map((r) => [r.fragmentId, r])),
    userOwnedKeys: new Set(
      records.filter((r) => r.assertion === "user" && r.canonicalKey).map((r) => r.canonicalKey!),
    ),
    isSuppressed: (id) => suppressed.has(id),
  };
};

/** 以固定管线评估一个 writer 臂：propose → G5.0 校验 → 与期望对照。 */
export const evaluateWriter = async (
  writer: MemoryWriter,
  samples: readonly WriterSample[] = WRITER_SAMPLES,
): Promise<WriterEvalReport> => {
  const verdicts: SampleVerdict[] = [];
  for (const sample of samples) {
    const input = sampleInput(sample);
    const batch = await writer.propose(input);
    const validations = validateCandidateBatch(batch, input, sampleContext(sample, input));
    const validated = validations.filter((v) => v.status === "validated" && v.candidate);
    const rejectedReasons = validations
      .filter((v) => v.status !== "validated")
      .flatMap((v) => [...v.reasons]);
    const exp = sample.expect;
    let verdict: SampleVerdict["verdict"];
    if (exp.type === "none") {
      verdict = validated.length === 0 ? "correct_none" : "false_write";
    } else if (exp.type === "write") {
      const hit = validated.some(
        (v) => v.candidate!.kind === exp.kind && v.candidate!.content.includes(exp.contentIncludes),
      );
      verdict = hit ? "hit" : validated.length > 0 ? "false_write" : "miss";
    } else {
      const hit = validated.some(
        (v) => v.candidate!.operation === "supersede" && v.candidate!.content.includes(exp.contentIncludes),
      );
      verdict = hit ? "hit" : validated.length > 0 ? "false_write" : "miss";
    }
    verdicts.push({
      sampleId: sample.id,
      category: sample.category,
      verdict,
      validatedCount: validated.length,
      rejectedReasons,
    });
  }
  const expectedWrites = samples.filter((s) => s.expect.type !== "none").length;
  const hits = verdicts.filter((v) => v.verdict === "hit").length;
  const misses = verdicts.filter((v) => v.verdict === "miss").length;
  const falseWrites = verdicts.filter((v) => v.verdict === "false_write").length;
  const correctNones = verdicts.filter((v) => v.verdict === "correct_none").length;
  return {
    writerId: writer.writerId,
    verdicts,
    totals: {
      samples: samples.length,
      expectedWrites,
      hits,
      misses,
      falseWrites,
      correctNones,
      recall: expectedWrites === 0 ? 1 : hits / expectedWrites,
      accuracy: samples.length === 0 ? 1 : (hits + correctNones) / samples.length,
    },
  };
};
