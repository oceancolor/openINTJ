import { describe, expect, it } from "vitest";
import { SkillLearningRuntime, verifySkillOutcome } from "../src/index.js";

describe("G5.3 技能验证器（工具证据为主 + 用户反馈覆盖）", () => {
  it("证据门：verified 工具成功且无失败 → verified(tool_evidence)", () => {
    expect(
      verifySkillOutcome({
        status: "completed",
        toolOutcomes: [{ toolName: "write_file", businessStatus: "success", evidenceStatus: "verified" }],
      }),
    ).toEqual({ verified: true, source: "tool_evidence", reason: "verified_tool_evidence" });
  });

  it("证据不足逐项未验证：仅 observed / business unknown / 无工具", () => {
    expect(
      verifySkillOutcome({
        status: "completed",
        toolOutcomes: [{ businessStatus: "success", evidenceStatus: "observed" }],
      }).reason,
    ).toBe("no_verified_evidence");
    expect(
      verifySkillOutcome({
        status: "completed",
        toolOutcomes: [{ businessStatus: "unknown", evidenceStatus: "verified" }],
      }).verified,
    ).toBe(false);
    expect(verifySkillOutcome({ status: "completed", toolOutcomes: [] }).verified).toBe(false);
  });

  it("任一 business failure 在场 → 不 verified（哪怕另有 verified 成功）", () => {
    expect(
      verifySkillOutcome({
        status: "completed",
        toolOutcomes: [
          { businessStatus: "success", evidenceStatus: "verified" },
          { businessStatus: "failure", evidenceStatus: "observed" },
        ],
      }),
    ).toMatchObject({ verified: false, reason: "tool_business_failure_present" });
  });

  it("非 completed 不 verified；用户正反馈可独立构成 verified；负反馈一票否决", () => {
    expect(verifySkillOutcome({ status: "failed", toolOutcomes: [], feedback: "positive" })).toMatchObject({
      verified: false,
      reason: "status_not_completed",
    });
    expect(verifySkillOutcome({ status: "completed", toolOutcomes: [], feedback: "positive" })).toEqual({
      verified: true,
      source: "user_feedback",
      reason: "verified_user_feedback",
    });
    expect(
      verifySkillOutcome({
        status: "completed",
        toolOutcomes: [{ businessStatus: "success", evidenceStatus: "verified" }],
        feedback: "negative",
      }),
    ).toMatchObject({ verified: false, reason: "user_negative_feedback" });
  });
});

describe("G5.3 反馈事后追认 (recordFeedbackForSelection)", () => {
  it("未验证 completed + 事后正反馈 → 补升权 + 进 buffer，一次性", () => {
    const rt = new SkillLearningRuntime();
    const id = rt.beginSelection(["s1"])!;
    expect(rt.recordOutcomeForSelection(id, { status: "completed", verified: false, query: "q" })).toBe(true);
    expect(rt.weightFor("s1")).toBe(0);
    expect(rt.bufferedCount()).toBe(0);

    expect(rt.recordFeedbackForSelection(id, "positive")).toBe(true);
    expect(rt.weightFor("s1")).toBe(1);
    expect(rt.bufferedCount()).toBe(1);
    // 重复反馈不再重复计奖。
    expect(rt.recordFeedbackForSelection(id, "positive")).toBe(false);
    expect(rt.recordFeedbackForSelection(id, "negative")).toBe(false);
    expect(rt.weightFor("s1")).toBe(1);
  });

  it("已 verified 生效后再收负反馈 → 一次 -0.5 相抵；未知 selection 返回 false", () => {
    const rt = new SkillLearningRuntime();
    const id = rt.beginSelection(["s1"])!;
    rt.recordOutcomeForSelection(id, { status: "completed", verified: true });
    expect(rt.weightFor("s1")).toBe(1);
    expect(rt.recordFeedbackForSelection(id, "negative")).toBe(true);
    expect(rt.weightFor("s1")).toBeCloseTo(0.5);
    expect(rt.recordFeedbackForSelection("sel-unknown", "positive")).toBe(false);
  });

  it("observe 模式：反馈只记观察不升权", () => {
    const rt = new SkillLearningRuntime({ learnMode: "observe" });
    const id = rt.beginSelection(["s1"])!;
    rt.recordOutcomeForSelection(id, { status: "completed", verified: false });
    expect(rt.recordFeedbackForSelection(id, "positive")).toBe(true);
    expect(rt.weightFor("s1")).toBe(0);
    expect(rt.bufferedCount()).toBe(0);
  });
});
