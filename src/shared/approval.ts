/**
 * 权限与审批契约
 *
 * 依据 IC-08（用户 2026-10-20 裁决，方案 C 按风险分级）与《系统设计》§3.2.M10：
 *   - 三档：宽松 / 标准 / 严格，默认「标准」
 *   - 代码枚举 strict / standard / autonomous（autonomous = 中文「宽松」，命名映射见交付说明）
 *   - 只读与工作区内写入**不得阻塞并发**（否则并行子 Agent 会因审批队列退化为串行）
 *   - 每条裁决事件必带 RiskDecisionContext 六要素，缺任一字段拒绝执行（B100003）
 */

/** 审批档位。注意：autonomous 的中文名是「宽松」，不是「免审批」 */
export type ApprovalTier = 'strict' | 'standard' | 'autonomous'

/** 中文显示名 —— UI 一律用这三个词，不得出现「自治」 */
export const TIER_LABELS: Record<ApprovalTier, string> = {
  strict: '严格',
  standard: '标准',
  autonomous: '宽松'
}

export const DEFAULT_TIER: ApprovalTier = 'standard'

/**
 * 命令风险等级
 * L0 只读 / L1 工作区内写入 / L2 工作区外写入 / L3 危险命令
 * UserStory 用 L0~L3，系统设计用 R1~R4，映射：L0↔R1、L1↔R2、L2↔R3、L3↔R4
 * 另有 R5 网络、R0 无法归类（不对应 L 级）
 */
export type RiskClass = 'L0' | 'L1' | 'L2' | 'L3' | 'R5-network' | 'R0-unknown'

/** 裁决结果（四选一） */
export type DecisionOutcome =
  | 'AUTO_APPROVE' // 自动放行
  | 'SESSION_GRANT' // 会话级授权
  | 'PER_CALL_PROMPT' // 逐次审批
  | 'DENY' // 拒绝

/** 裁决依据 */
export type DecisionBasis = 'risk-table' | 'user-grant' | 'user-deny' | 'timeout' | 'policy-deny'

/**
 * 审计风险六要素 —— 缺任一字段即拒绝执行（错误码 B100003）
 */
export interface RiskDecisionContext {
  /** 命令原文（redaction 后） */
  command: string
  riskClass: RiskClass
  tier: ApprovalTier
  outcome: DecisionOutcome
  basis: DecisionBasis
  matchedRuleId: string
}

/** 一条待裁决的审批请求 */
export interface ApprovalRequest {
  requestId: string
  runId: string
  agentId: string
  /** 并行泳道标识 —— 用于保证泳道间并行、泳道内串行 */
  parallelLaneId: string
  command: string
  cwd: string
  riskClass: RiskClass
  /** 是否允许异步提示（AUTO_APPROVE / SESSION_GRANT / DENY 应置 true，零等待） */
  allowAsyncPrompt: boolean
  at: number
}

/** 真值表：档位 × 风险等级 → 处置（IC-08 裁决的代码化） */
export const TIER_TRUTH_TABLE: Record<ApprovalTier, Record<RiskClass, DecisionOutcome>> = {
  // 严格：全部逐次审批
  strict: {
    L0: 'PER_CALL_PROMPT',
    L1: 'PER_CALL_PROMPT',
    L2: 'PER_CALL_PROMPT',
    L3: 'PER_CALL_PROMPT',
    'R5-network': 'PER_CALL_PROMPT',
    'R0-unknown': 'PER_CALL_PROMPT'
  },
  // 标准（默认）：只读自动放行、工作区内写入会话授权一次、高风险逐次
  standard: {
    L0: 'AUTO_APPROVE',
    L1: 'SESSION_GRANT',
    L2: 'PER_CALL_PROMPT',
    L3: 'PER_CALL_PROMPT',
    'R5-network': 'PER_CALL_PROMPT',
    'R0-unknown': 'PER_CALL_PROMPT'
  },
  // 宽松：只读与工作区内写入都自动放行，但 L2/L3 仍逐次（不等于免审批）
  autonomous: {
    L0: 'AUTO_APPROVE',
    L1: 'AUTO_APPROVE',
    L2: 'PER_CALL_PROMPT',
    L3: 'PER_CALL_PROMPT',
    'R5-network': 'PER_CALL_PROMPT',
    'R0-unknown': 'PER_CALL_PROMPT'
  }
}

/**
 * Windows 平台策略：宽松档默认禁用（C1）
 * Windows 无 OS 级隔离兜底，审批是唯一防线；宽松档会把工作区内写入降为自动放行。
 */
export function isTierAllowedOnPlatform(tier: ApprovalTier, platform: NodeJS.Platform): boolean {
  if (platform === 'win32' && tier === 'autonomous') return false
  return true
}

/**
 * 裁决：档位 × 风险等级 → 处置。
 * **必须在 tool-host 侧调用**（裁决权不外包给调用方，见验收报告 D10）。
 */
export function decideOutcome(tier: ApprovalTier, riskClass: RiskClass): DecisionOutcome {
  return TIER_TRUTH_TABLE[tier][riskClass]
}

/** 风险等级权重，用于「取最严」比较（数值越大越危险） */
export const RISK_WEIGHT: Record<RiskClass, number> = {
  L0: 0,
  'R0-unknown': 5, // 无法归类按最严
  L1: 1,
  L2: 2,
  'R5-network': 3,
  L3: 4
}

export function riskWeight(r: RiskClass): number {
  return RISK_WEIGHT[r]
}
