/**
 * 内嵌终端契约（US-15 / F29）
 *
 * 形态决策 —— 这一版终端**不是**「裸 PTY 键盘直通」，而是
 * 「持久 PTY 会话 + 只读输出区 + 命令输入行」。
 *
 * 为什么不做键盘直通：
 *   AC-2 是硬约束 ——「终端不是绕过策略层的通道」。要做到「每条命令执行前裁决」，
 *   就必须知道**一条命令的边界**。键盘直通下，用户敲的每个字节都直接进 pty，
 *   回车也跟着进，pty 立刻执行 —— 此时再想拦已经晚了（命令已经跑了）。
 *   唯一能在执行前拿到完整命令串的办法，是把「提交」这一步拿在自己手里：
 *   输入行里攒出整条命令 → 分级 → 放行才把 `命令 + \r` 写进 pty。
 *
 * 为什么仍然要真 PTY（而不是 child_process）：
 *   1. 会话持久 —— `cd`、`export`、虚拟环境在两条命令之间要保留，这是终端和「跑一条命令」的区别；
 *   2. TTY 语义 —— 有 pty，`ls` 才有颜色、`npm` 才有进度条、程序才会走交互分支；
 *   3. 尺寸语义 —— 有 cols/rows，长输出才会按窗口宽度折行（AC-5）。
 *
 * 代价（必须写清楚，不是缺陷是取舍）：
 *   - 不支持交互式全屏程序（vim / top / ssh 登录后的远端终端）—— 它们的「命令边界」不是换行；
 *   - 不支持需要实时按键的程序（方向键选择菜单之类）；
 *   - 交互式问答可以用：把 `y` 当一条命令提交即可（pty 会当作输入）。
 */

import type { DecisionBasis, RiskClass } from './approval'

/** 会话创建参数 */
export interface TerminalCreateParams {
  /** 工作目录；缺省用工作区根。越出工作区根会被拒（与命令执行同一约束） */
  cwd?: string
  cols: number
  rows: number
}

export interface TerminalSessionInfo {
  sessionId: string
  /** 实际拉起的 shell 可执行文件 */
  shell: string
  cwd: string
  cols: number
  rows: number
  startedAt: number
  /** 该会话绑定的策略版本 —— 档位变了也不会追溯改变已放行的命令 */
  policyProfileVersion: number
}

/**
 * 提交一条命令的结果。
 *
 * 为什么 `need-approval` 要作为**返回值**而不是抛异常：
 * 它不是错误，是正常的中间态。抛异常会让调用方把「等审批」和「被拒绝」混在一起，
 * 而这俩的后续动作完全相反 —— 一个要继续等，一个要明示拒绝原因。
 */
export type TerminalSubmitResult =
  | {
      kind: 'ran'
      command: string
      riskClass: RiskClass
      matchedRuleId: string
      basis: DecisionBasis
    }
  | {
      kind: 'need-approval'
      command: string
      riskClass: RiskClass
      matchedRuleId: string
      reason?: string
    }
  | {
      kind: 'denied'
      command: string
      riskClass: RiskClass
      matchedRuleId: string
      reason: string
    }
  /** 空行 / 纯空白：不裁决、不审计、不执行 */
  | { kind: 'empty' }
  /** 会话不存在（已退出或从未创建） */
  | { kind: 'no-session' }

/** 终端的 runId 约定：终端会话不属于任何 Agent run，但审计需要 runId */
export const TERMINAL_RUN_PREFIX = 'terminal:'

export function terminalRunId(sessionId: string): string {
  return `${TERMINAL_RUN_PREFIX}${sessionId}`
}

/** 终端专属事件通道（tool-host → 主进程 → 渲染层） */
export const TERMINAL_CHANNELS = {
  /** 输出流。高频通道，只推给渲染层，不落事件库 */
  data: 'terminal.data',
  /** 会话结束（shell 退出或被杀） */
  exit: 'terminal.exit'
} as const

export interface TerminalDataPayload {
  sessionId: string
  data: string
}

export interface TerminalExitPayload {
  sessionId: string
  exitCode: number
  signal?: number
}

/**
 * 终端输出**不做脱敏**，这是刻意的。
 *
 * 与 D8 脱敏管线的边界：脱敏管的是「落盘」与「进事件流」—— 那两处一旦漏了，
 * 密钥会永久留在磁盘上或被发到别处。终端输出是**用户自己屏幕上的回显**：
 * 他在终端里 `cat .env` 是他自己的选择，把回显遮掉只会让他以为文件是空的。
 *
 * 所以：终端流不进事件库（见 TerminalHost），也就不经过 emitEvent 那个唯一口子。
 * 若将来要把终端输出落盘或喂给模型，必须先过脱敏 —— 那时这条注释就是提醒。
 */
export const TERMINAL_OUTPUT_IS_NOT_REDACTED = true
