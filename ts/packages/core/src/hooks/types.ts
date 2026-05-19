import type { Command } from "../types/command-event.js";
import type { LODLevelType, ShaderModeType } from "../types/shader.js";
import type { ToolCallResult, ToolDescriptor } from "../types/tool.js";

export type HookCategory = "lifecycle" | "tool" | "event" | "policy";

export interface HookContext<P> {
  readonly eventName: string;
  readonly traceId: string;
  payload: P;
  readonly executedCount: number;
  cancel(): void;
  readonly isCancelled: boolean;
  replace(newPayload: P): void;
  meta: Record<string, unknown>;
}

export type HookHandler<P> = (ctx: HookContext<P>) => void | Promise<void>;

export interface HookRegistration {
  priority?: number;
  once?: boolean;
  tag?: string;
  /**
   * 当事件本身允许短路时，此项为 true 才能在 handler 中调用 ctx.cancel()。
   * 默认为 true（事件级 cancellable 决定是否允许）。
   */
  allowCancel?: boolean;
}

export type Unregister = () => void;

export interface HookInspectResult {
  total: number;
  byEvent: Record<string, number>;
  byTag: Record<string, number>;
  strictMode: boolean;
}

export interface AuditEvent {
  eventId: string;
  timestamp: number;
  action: string;
  actor: string;
  target: string;
  result: "allowed" | "blocked" | "warning";
  details: Record<string, unknown>;
  riskLevel: "low" | "medium" | "high" | "critical";
}

/**
 * 以下接口是钩子事件 payload 中引用的"骨架类型"。
 * 真正实现位于 loop/ 与各 plane 包；此处仅声明结构形状以避免循环依赖。
 * 真正的 PlanGraph、ContextWindow 等由 plane 模块通过 declaration merging 收敛。
 */
export interface PlanGraphLike {
  planId: string;
  totalSteps: number;
}
export interface ContextWindowLike {
  systemPrompt: string;
}
export interface ReactOutputLike {
  finalAnswer: string;
  status: string;
  iterations: number;
}
export interface TrajectoryEntryLike {
  timestamp: number;
}
export interface ReactStopReasonLike {
  kind: string;
  reactIter: number;
}
export interface WorkspaceChangeEventLike {
  path: string;
}

export interface HookEventMap {
  // -------- TAO 宏循环 --------
  "tao.beforeThink": { query: string; iteration: number };
  "tao.afterThink": { plan: PlanGraphLike; iteration: number };
  "tao.beforeAct": {
    plan: PlanGraphLike;
    availableTools: ToolDescriptor[];
    iteration: number;
  };
  "tao.afterAct": { reactOutput: ReactOutputLike; iteration: number };
  "tao.beforeObserve": {
    trajectory: TrajectoryEntryLike[];
    iteration: number;
  };
  "tao.afterObserve": { needsContinue: boolean; iteration: number };

  // -------- ReAct 微循环 --------
  "react.beforeThought": {
    context: ContextWindowLike;
    reactIter: number;
    taoIter: number;
  };
  "react.afterThought": {
    thought: string;
    reactIter: number;
    taoIter: number;
  };
  "react.beforeAction": {
    tool: string;
    params: unknown;
    reactIter: number;
    taoIter: number;
  };
  "react.afterAction": {
    toolResult: ToolCallResult;
    reactIter: number;
    taoIter: number;
  };
  "react.onStopCondition": ReactStopReasonLike;

  // -------- Tool --------
  "tool.beforeCall": {
    tool: string;
    params: unknown;
    toolDescriptor: ToolDescriptor;
  };
  "tool.afterCall": { tool: string; result: ToolCallResult };
  "tool.onError": { tool: string; error: Error; willRetry: boolean };

  // -------- Event (对齐 Python EventType) --------
  "event.MEMORY_LOADED": { count: number; budgetUsage: number };
  "event.CONTEXT_COMPACTED": {
    compactedMessages: number;
    newBudgetUsage: number;
  };
  "event.SHADER_APPLIED": { mode: ShaderModeType; lod: LODLevelType };
  "event.POLICY_BLOCKED": { command: Command; reason: string };
  "event.CIRCUIT_OPENED": { tool: string; failureCount: number };
  "event.LOOP_ITERATION": {
    taoIter: number;
    metrics: Record<string, number>;
  };

  // -------- Policy --------
  "policy.beforeCheck": { command: Command };
  "policy.afterCheck": { command: Command; auditEvent: AuditEvent };
  "policy.onBlock": {
    command: Command;
    auditEvent: AuditEvent;
    reason: string;
  };
}

/** 哪些事件允许 handler 调用 ctx.cancel()。 */
export const CANCELLABLE_EVENTS: ReadonlySet<keyof HookEventMap> = new Set<keyof HookEventMap>([
  "react.beforeAction",
  "tool.beforeCall",
  "policy.beforeCheck",
]);

/** 事件名 → 类别映射，用于审计/分析。 */
export const eventCategory = (event: string): HookCategory => {
  if (event.startsWith("tao.") || event.startsWith("react.")) return "lifecycle";
  if (event.startsWith("tool.")) return "tool";
  if (event.startsWith("event.")) return "event";
  if (event.startsWith("policy.")) return "policy";
  return "lifecycle";
};
