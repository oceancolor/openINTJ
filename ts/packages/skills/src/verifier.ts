/**
 * G5.3 技能正向学习验证器：给 F10 的 verified 信号一个真实来源。
 *
 * 口径（2026-11-10 用户决策）：**工具证据为主 + 用户反馈可覆盖**。
 * - 工具证据：run completed，且至少一条 ToolOutcome 达到
 *   business=success + evidence=verified，且没有任何 business=failure。
 * - 用户负反馈一票否决（即使证据通过也不升权）；
 *   用户正反馈可在无工具证据时独立构成 verified（纯提示类技能靠它）。
 * - 其余情况一律未验证：completed 只记观察（G3 基础阶段口径不变）。
 *
 * 本模块纯函数、无状态；反馈的事后追认由 learning-runtime 的
 * recordFeedbackForSelection 负责（一次性，防重复计奖）。
 */

export interface ToolEvidenceSummary {
  toolName?: string;
  executionStatus?: string;
  businessStatus?: "success" | "failure" | "unknown";
  evidenceStatus?: "none" | "observed" | "verified";
}

export type FeedbackSignal = "positive" | "negative";

export interface SkillVerification {
  verified: boolean;
  source: "tool_evidence" | "user_feedback" | "none";
  /** 稳定原因码，供日志/诊断（不进用户文案）。 */
  reason:
    | "verified_tool_evidence"
    | "verified_user_feedback"
    | "user_negative_feedback"
    | "status_not_completed"
    | "tool_business_failure_present"
    | "no_verified_evidence";
}

export const verifySkillOutcome = (input: {
  status: string;
  toolOutcomes: readonly ToolEvidenceSummary[];
  feedback?: FeedbackSignal;
}): SkillVerification => {
  if (input.feedback === "negative") {
    return { verified: false, source: "user_feedback", reason: "user_negative_feedback" };
  }
  if (input.status !== "completed") {
    return { verified: false, source: "none", reason: "status_not_completed" };
  }
  const hasFailure = input.toolOutcomes.some((o) => o.businessStatus === "failure");
  if (hasFailure) {
    return { verified: false, source: "none", reason: "tool_business_failure_present" };
  }
  const hasVerifiedSuccess = input.toolOutcomes.some(
    (o) => o.businessStatus === "success" && o.evidenceStatus === "verified",
  );
  if (hasVerifiedSuccess) {
    return { verified: true, source: "tool_evidence", reason: "verified_tool_evidence" };
  }
  if (input.feedback === "positive") {
    return { verified: true, source: "user_feedback", reason: "verified_user_feedback" };
  }
  return { verified: false, source: "none", reason: "no_verified_evidence" };
};
