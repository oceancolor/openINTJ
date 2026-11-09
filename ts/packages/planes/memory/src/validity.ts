import type { MemoryFragment } from "@openintj/core";

/**
 * F08 有效性：validity 与 recency 分离。
 * - superseded（已被纠正取代）与 expired（validUntil 已过）在所有召回入口不可见，
 *   不是只让 shader 不显示。
 * - 未核验内容（inferred/quoted）可作为"带标签的历史陈述"召回，
 *   但不得作为执行权限或成功经验——由调用方按 assertion 决定用途。
 */
export type ValidityState = "valid" | "expired" | "superseded";

export const validityOf = (f: MemoryFragment, nowSec: number = Date.now() / 1000): ValidityState => {
  if (f.supersededBy) return "superseded";
  if (f.validUntil !== undefined && nowSec > f.validUntil) return "expired";
  return "valid";
};

export const isCurrentlyValid = (f: MemoryFragment, nowSec?: number): boolean =>
  validityOf(f, nowSec) === "valid";

/** 是否可作为"已验证事实"使用（执行权限/成功经验）：必须是用户断言且当前有效。 */
export const isVerifiedAssertion = (f: MemoryFragment, nowSec?: number): boolean =>
  isCurrentlyValid(f, nowSec) && f.assertion === "user";
