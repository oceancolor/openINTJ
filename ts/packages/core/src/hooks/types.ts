import type { ProductTraitId } from "@openintj/shared";
import type { Command } from "../types/command-event.js";
import type { MemoryFragment } from "../types/memory-fragment.js";
import type { LODLevelType, ShaderModeType } from "../types/shader.js";
import type { ToolCallResult, ToolDescriptor } from "../types/tool.js";

export type HookCategory = "lifecycle" | "tool" | "event" | "policy" | "concurrency";

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
  /**
   * 记忆写入 change-feed：MemoryStore 每次 add/晋升/移除时发出。
   * 消费方（如 session 级 HybridRetriever）据此做增量 upsert/remove，免去全量重建。
   * op=add 新增；op=update 已存在片段元数据变化（如短期溢出晋升为长期）；op=remove 移除。
   */
  "event.MEMORY_WRITTEN": { fragment: MemoryFragment; op: "add" | "update" | "remove" };
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
  /**
   * 技能命中：技能系统为本轮 query 选中并注入了 ≥1 个能力包（opt-in，OPENINTJ_SKILLS）。
   * 消费方据此做可观测（命中 counter）或调试；未命中不发。
   */
  "event.SKILL_SELECTED": {
    skills: { id: string; score: number }[];
    query: string;
  };
  /**
   * 技能提案：自学习闭环从成功轨迹蒸馏出一个候选技能并写入 pending（opt-in，OPENINTJ_SKILLS_LEARN）。
   * 消费方据此做可观测（提案 counter）或提醒用户去审批；只在 `distill()` 产出新提案时发。
   */
  "event.SKILL_PROPOSED": {
    proposalId: string;
    skillId: string;
    evidenceCount: number;
  };

  // -------- Policy --------
  "policy.beforeCheck": { command: Command };
  "policy.afterCheck": { command: Command; auditEvent: AuditEvent };
  "policy.onBlock": {
    command: Command;
    auditEvent: AuditEvent;
    reason: string;
  };

  // -------- 并发 / 多任务 / 多 Agent（RFC-003 方向一/二可观测性）--------
  /** AgentPool/worker 池：单个 job 开始。`active`/`pending` 为发出瞬间的池快照。 */
  "pool.beforeJob": {
    pool: string;
    jobId: string;
    active: number;
    pending: number;
  };
  /** AgentPool/worker 池：单个 job 结束（含成功/失败、耗时与池累计统计）。 */
  "pool.afterJob": {
    pool: string;
    jobId: string;
    success: boolean;
    durationMs: number;
    active: number;
    pending: number;
    completed: number;
    failed: number;
  };
  /** ForkJoin：分叉开始（total = 子任务数）。 */
  "forkjoin.beforeFork": { group: string; total: number };
  /** ForkJoin：合并完成（fulfilled/rejected 子任务数 + 总耗时）。 */
  "forkjoin.afterJoin": {
    group: string;
    total: number;
    fulfilled: number;
    rejected: number;
    durationMs: number;
  };
  /** TaskQueue：任务入队（DAG 依赖数 + 入队即就绪与否）。 */
  "task.enqueue": {
    queue: string;
    taskId: string;
    priority: number;
    depCount: number;
    ready: boolean;
  };
  /** TaskQueue：任务被 worker 取出开始执行（state→running）。 */
  "task.beforeRun": { queue: string; taskId: string; priority: number };
  /** TaskQueue：任务完成/失败（从取出到 complete/fail 的耗时）。 */
  "task.afterRun": {
    queue: string;
    taskId: string;
    success: boolean;
    durationMs: number;
  };
  /** TaskPool（RFC-007）：一次 DAG run 提交。 */
  "taskpool.run.submit": {
    pool: string;
    runId: string;
    planId: string;
    taskCount: number;
  };
  /** TaskPool：run 结束。 */
  "taskpool.run.complete": {
    pool: string;
    runId: string;
    planId: string;
    status: string;
    completed: number;
    failed: number;
    cancelled: number;
    timedOut: number;
  };
  /** TaskPool：依赖满足，task 可调度。 */
  "taskpool.task.ready": {
    pool: string;
    runId: string;
    taskId: string;
    attempt: number;
  };
  /** TaskPool：单 task 开始。 */
  "taskpool.task.start": {
    pool: string;
    runId: string;
    taskId: string;
    action: string;
    attempt: number;
    /** Trace id used by the worker's Tao/ReAct/tool lifecycle. */
    workerTraceId: string;
  };
  /** TaskPool：task 失败后进入有界重试。 */
  "taskpool.task.retry": {
    pool: string;
    runId: string;
    taskId: string;
    attempt: number;
    delayMs: number;
    error: string;
  };
  /** TaskPool：task 触发运行 watchdog。 */
  "taskpool.task.timeout": {
    pool: string;
    runId: string;
    taskId: string;
    attempt: number;
    timeoutMs: number;
  };
  /** TaskPool：task 被显式取消或由依赖级联取消。 */
  "taskpool.task.cancel": {
    pool: string;
    runId: string;
    taskId: string;
    reason: string;
  };
  /** TaskPool：单 task 结束。 */
  "taskpool.task.complete": {
    pool: string;
    runId: string;
    taskId: string;
    success: boolean;
    status: string;
    attempt: number;
    error?: string;
  };
  /** RFC-005：provider 健康探测完成（不包含 endpoint、凭据或响应体）。 */
  "model.provider.probe": {
    channel: "llm" | "embedding";
    provider: string;
    model: string;
    ok: boolean;
    durationMs: number;
    errorCode?: string;
  };
  /** RFC-005：resolution 选定实际 provider。 */
  "model.provider.selected": {
    channel: "llm" | "embedding";
    requestedProvider: string;
    provider: string;
    model: string;
    mode: string;
  };
  /** RFC-005：auto resolution 发生显式可见 fallback。 */
  "model.provider.fallback": {
    channel: "llm" | "embedding";
    from: string;
    to: string;
    errorCode: string;
  };
  /** RFC-005：provider resolution/refresh 失败；message 必须已脱敏。 */
  "model.provider.error": {
    channel: "llm" | "embedding";
    provider: string;
    code: string;
    message: string;
    retriable: boolean;
  };
  /** RFC-005：持久化 embedding 指纹通过校验或首次创建。 */
  "model.embedding.fingerprint.checked": {
    expected: string;
    stored?: string;
    result: "matched" | "created";
  };
  /** RFC-005：持久化 embedding 指纹拒绝；不执行自动清库。 */
  "model.embedding.fingerprint.rejected": {
    expected: string;
    stored?: string;
    code: "EMBEDDING_FINGERPRINT_MISSING" | "EMBEDDING_FINGERPRINT_MISMATCH";
  };
  /** Product Behavior（RFC-006）：本轮 run 注入的行为版本。 */
  "event.PRODUCT_BEHAVIOR": {
    version: string;
    enabled: boolean;
  };
  /**
   * RFC-006 trait signal derived from deterministic framework evidence.
   * It records an observed lifecycle/tool fact, never inferred model intent.
   */
  "event.PRODUCT_TRAIT_SIGNAL": {
    trait: ProductTraitId;
    signal: "plan_decomposed" | "clarification_skill_selected" | "search_before_answer";
    value: number;
    source: "tao.afterThink" | "event.SKILL_SELECTED" | "tool.afterCall";
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
  if (event.startsWith("pool.") || event.startsWith("forkjoin.") || event.startsWith("task.")) {
    return "concurrency";
  }
  if (event.startsWith("taskpool.")) return "concurrency";
  return "lifecycle";
};
