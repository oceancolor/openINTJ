import { describe, expect, it } from "vitest";
import {
  createFixtureWriter,
  createLlmWriter,
  createRawBaselineWriter,
  evaluateWriter,
  sampleContext,
  sampleInput,
  validateCandidateBatch,
  WRITER_SAMPLES,
} from "../src/index.js";

describe("G5.4 LLM writer 协议适配（stub generate，无网络）", () => {
  it("模型 JSON 输出经提取后走同一校验管线；自带 scope 被拒", async () => {
    const writer = createLlmWriter({
      writerId: "stub-llm",
      generate: async () =>
        '这是结果：\n```json\n[{"operation":"add","kind":"fact","content":"用户住在成都","sourceRefs":["ev:s13-implicit-fact"],"rationale":"长期事实"}]\n```',
    });
    const sample = WRITER_SAMPLES.find((s) => s.id === "s13-implicit-fact")!;
    const input = sampleInput(sample);
    const batch = await writer.propose(input);
    expect(batch.candidates).toHaveLength(1);
    const validations = validateCandidateBatch(batch, input, sampleContext(sample, input));
    expect(validations[0]?.status).toBe("validated");

    const evil = createLlmWriter({
      generate: async () =>
        '[{"operation":"add","kind":"fact","content":"x","sourceRefs":["ev:s13-implicit-fact"],"rationale":"","scope":{"workspaceId":"ws-other"}}]',
    });
    const evilBatch = await evil.propose(input);
    const evilValidations = validateCandidateBatch(evilBatch, input, sampleContext(sample, input));
    expect(evilValidations[0]).toMatchObject({ status: "rejected", reasons: ["SCHEMA_INVALID"] });
  });

  it("非 JSON 输出提取为空批次（不猜测、不修复生成），校验得零候选", async () => {
    const writer = createLlmWriter({ generate: async () => "我觉得应该记住这些（此处省略）" });
    const sample = WRITER_SAMPLES[0]!;
    const batch = await writer.propose(sampleInput(sample));
    expect(batch.candidates).toEqual([]);
  });
});

describe("G5.4 对照评估：原文基线 vs fixture（确定性，§10 首轮下限）", () => {
  it("fixture：显式样本命中、噪声样本正确空转、无标记隐含样本漏记（特征化）", async () => {
    const report = await evaluateWriter(createFixtureWriter());
    const byId = new Map(report.verdicts.map((v) => [v.sampleId, v]));
    expect(byId.get("s01-explicit-preference")?.verdict).toBe("hit");
    expect(byId.get("s02-explicit-remember")?.verdict).toBe("hit");
    expect(byId.get("s03-one-off")?.verdict).toBe("correct_none");
    expect(byId.get("s05-assistant-echo")?.verdict).toBe("correct_none");
    expect(byId.get("s06-quoting-others")?.verdict).toBe("correct_none");
    // s12 候选被校验层以敏感内容拦下（validated=0）→ correct_none。
    expect(byId.get("s12-sensitive-secret")?.verdict).toBe("correct_none");
    expect(byId.get("s12-sensitive-secret")?.rejectedReasons).toContain("SENSITIVE_CONTENT");
    // 无显式标记的隐含事实/偏好与纠正：fixture 漏记（这正是需要 LLM writer 的部分）。
    expect(byId.get("s13-implicit-fact")?.verdict).toBe("miss");
    expect(byId.get("s14-implicit-preference")?.verdict).toBe("miss");
    expect(byId.get("s04-user-correction")?.verdict).toBe("miss");
    expect(report.totals.falseWrites).toBe(0);
  });

  it("原文基线：无差别直存导致误存（复刻 RFC 前行为，作为对照下限）", async () => {
    const report = await evaluateWriter(createRawBaselineWriter());
    expect(report.totals.falseWrites).toBeGreaterThanOrEqual(5);
    expect(report.totals.accuracy).toBeLessThan(0.5);
  });
});
