/**
 * 分类结果 → 路由决策（纯函数）。把「降 token 路由」与装配解耦，便于单测与三端复用。
 */

import { TaskType, type TaskTypeType } from "@openintj/core";
import type { ClassifyResult } from "./reinforcing-classifier.js";

export interface RouteDecision {
  /** 是否走单次 LLM 调用（跳过 ReAct 微循环与工具描述，降 token）。 */
  single: boolean;
  /** 建议检索 topK（简单类更小）。 */
  topK: number;
}

export interface RoutingPolicy {
  /** 视为「简单、可单次」的任务类型。 */
  simpleTypes?: readonly TaskTypeType[];
  /** 永不走 single 路由的复杂任务类型（RFC-006 护栏）。 */
  complexTypes?: readonly TaskTypeType[];
  /** 触发单次路由的最低置信度。 */
  minConfidence?: number;
  /** 简单类 topK。 */
  simpleTopK?: number;
  /** 默认（复杂类）topK。 */
  defaultTopK?: number;
}

/**
 * 高置信 + 简单类 → 单次 LLM 路由（降 token）。
 * 兜底分类（fallback=true）一律不激进路由，避免误判把复杂任务降级。
 */
export const decideRoute = (r: ClassifyResult, policy: RoutingPolicy = {}): RouteDecision => {
  const complexTypes = policy.complexTypes ?? [TaskType.PLANNING, TaskType.ANALYSIS];
  if (complexTypes.includes(r.label)) {
    return { single: false, topK: policy.defaultTopK ?? 6 };
  }
  const simpleTypes = policy.simpleTypes ?? [TaskType.QUICK_RESPONSE, TaskType.GENERAL_CHAT];
  const minConfidence = policy.minConfidence ?? 0.6;
  const single = !r.fallback && r.confidence >= minConfidence && simpleTypes.includes(r.label);
  return {
    single,
    topK: single ? (policy.simpleTopK ?? 3) : (policy.defaultTopK ?? 6),
  };
};

/**
 * 从 run 结果推导强化信号：
 *  - completed → +1（高效完成更值得正反馈）
 *  - failed/timeout → -0.5（标签可能不当）
 *  - 其它（max_iter 等）→ +0.2（弱正）
 */
export const outcomeSignal = (status: string): number => {
  if (status === "completed") return 1;
  if (status === "failed" || status === "timeout") return -0.5;
  return 0.2;
};
