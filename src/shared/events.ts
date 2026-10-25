/**
 * 事件模型 — 封闭词表
 *
 * 依据《系统设计》§3.2 领域模型：
 *   - 事件流为状态的单一事实来源（F5）
 *   - Action / Observation 封闭词表，参照 OpenHands
 *   - EventEnvelope 携带 ULID eventId、任务内 seq 位点、traceId、tenantId、redaction、schemaVersion
 */

/** 动作事件类型（封闭词表 —— 新增必须同步更新事件存储与回放器） */
export type ActionType =
  | 'GoalContractDrafted'
  | 'GoalContractConfirmed' // IC-07 硬闸门
  | 'GoalDecomposed'
  | 'SubAgentSpawned'
  | 'TaskStarted'
  | 'ToolInvoked'
  | 'ApprovalRequested'
  | 'ApprovalDecided'
  | 'CheckpointCreated'
  | 'ContextCompressed'
  | 'CriticScored'
  | 'ReplanProposed'
  | 'InterventionRecorded'
  | 'PolicyTierSwitched'
  | 'WorkspaceCloudProjected'
  | 'WorkspaceCloudRetrieved'
  | 'RunPaused'
  | 'RunResumed'
  | 'RunDelivered'
  | 'RunFailed'

/** 观察事件类型（封闭词表） */
export type ObservationType =
  | 'ToolResult'
  | 'ModelMessage'
  | 'ModelUsage'
  | 'FileChanged'
  | 'CommandOutput'
  | 'ApprovalRejected'
  | 'ErrorObserved'

/** 脱敏级别 —— 事件落盘前必须标注 */
export type Redaction = 'none' | 'secret' | 'path' | 'content'

export interface EventEnvelope<T = unknown> {
  /** ULID，全局唯一，可排序 */
  eventId: string
  /** 任务内单调递增位点，恢复时用于定位重放起点 */
  seq: number
  /** 关联的任务运行 ID */
  runId: string
  /** 链式追踪 ID，贯穿事件流、日志、审计 */
  traceId: string
  /** 租户 ID（单机形态下为固定值，云端多租户时区分） */
  tenantId: string
  /** 来源子 Agent 标识；主控台发起的为 'main' */
  agentId: string
  kind: 'action' | 'observation'
  type: ActionType | ObservationType
  /** 发生时间（epoch ms） */
  at: number
  /** 脱敏级别 */
  redaction: Redaction
  /** 事件负载；redaction !== 'none' 时此处已是脱敏后的内容 */
  payload: T
  /** schema 版本，用于事件回放的向前兼容 */
  schemaVersion: 1
}

/** 事件写入结果 */
export interface AppendResult {
  seq: number
  eventId: string
}

/** 任务运行状态机（《系统设计》§3.2） */
export type TaskRunState =
  | 'CONTRACT_PENDING'
  | 'CONFIRMED'
  | 'RUNNING'
  | 'PAUSED'
  | 'INTERRUPTED'
  | 'CRITIC_REVIEW'
  | 'REPLANNING'
  | 'DELIVERED'
  | 'FAILED'

/** 检查点作用域（恢复粒度） */
export type CheckpointScope =
  | 'milestone-boundary'
  | 'pre-tool-call'
  | 'manual'
  | 'pre-compaction'

export interface Checkpoint {
  checkpointId: string
  runId: string
  /** 指向事件流的位点：恢复时从此 seq 之后重放 */
  atSeq: number
  scope: CheckpointScope
  at: number
  /** 重放步数上限由编排层保证 ≤ 5（V1 / QS-01） */
  label: string
}
