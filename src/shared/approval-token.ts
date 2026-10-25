/**
 * 审批票据 —— 裁决权的凭据化
 *
 * 背景（首版验收报告 D10）：
 *   首版 `ToolExecParams.risk.outcome` 由调用方（agent-host）填写，tool-host 只做白名单校验。
 *   一旦 LLM 接线，agent-host 跑的是模型产出的动作 —— 只要它自称 `AUTO_APPROVE`，闸门就废了。
 *
 * 修正：裁决权完全收回 tool-host（自己查真值表 + 会话授权）。
 *   「用户已批准」这件事不能靠调用方声明，只能靠**主进程签发的票据**证明。
 *
 * 信任边界：
 *   - 密钥只在启动时由主进程经 **tool-host 专用端口**下发，agent-host 拿不到
 *   - 票据绑定 runId + 命令哈希 + 过期时间 + 使用范围（单次 / 本任务内）
 *   - 票据单次使用即作废，重放无效
 */

import { createHash, createHmac, randomUUID, timingSafeEqual } from 'node:crypto'

/** 票据使用范围：单次 / 本任务内同级复用 */
export type TicketScope = 'once' | 'session'

export interface ApprovalTicket {
  ticketId: string
  /** 绑定任务 —— 票据不能跨任务挪用 */
  runId: string
  /** 绑定命令原文哈希 —— 批的是这条命令，不是「以后随便什么命令」 */
  commandHash: string
  scope: TicketScope
  /** 签发时确定的该命令风险等级；tool-host 校验时必须一致 */
  riskClass: string
  expiresAt: number
  /** HMAC-SHA256(secret, ticketId|runId|commandHash|scope|riskClass|expiresAt) */
  signature: string
}

/** 票据有效期：批一次命令不该长期有效 */
export const TICKET_TTL_MS = 5 * 60 * 1000

export function hashCommand(command: string): string {
  return createHash('sha256').update(command.trim(), 'utf8').digest('hex')
}

function payloadOf(
  t: Pick<ApprovalTicket, 'ticketId' | 'runId' | 'commandHash' | 'scope' | 'riskClass' | 'expiresAt'>
): string {
  return `${t.ticketId}|${t.runId}|${t.commandHash}|${t.scope}|${t.riskClass}|${t.expiresAt}`
}

function sign(secret: string, payload: string): string {
  return createHmac('sha256', secret).update(payload, 'utf8').digest('hex')
}

/** 由主进程签发（唯一持密钥方） */
export function issueTicket(
  secret: string,
  input: { runId: string; command: string; riskClass: string; scope: TicketScope; ttlMs?: number; now?: number }
): ApprovalTicket {
  const now = input.now ?? Date.now()
  const t = {
    ticketId: randomUUID(),
    runId: input.runId,
    commandHash: hashCommand(input.command),
    scope: input.scope,
    riskClass: input.riskClass,
    expiresAt: now + (input.ttlMs ?? TICKET_TTL_MS)
  }
  return { ...t, signature: sign(secret, payloadOf(t)) }
}

export interface VerifyResult {
  ok: boolean
  reason?: string
}

/** 由 tool-host 校验（持密钥另一副本，agent-host 无密钥） */
export function verifyTicket(
  secret: string,
  ticket: ApprovalTicket,
  expect: { runId: string; command: string; riskClass: string; now?: number }
): VerifyResult {
  if (!ticket || typeof ticket.signature !== 'string') return { ok: false, reason: '票据缺失或结构非法' }
  if (ticket.runId !== expect.runId) return { ok: false, reason: '票据与任务不匹配（防跨任务挪用）' }
  if (ticket.commandHash !== hashCommand(expect.command))
    return { ok: false, reason: '票据与命令不匹配（防改口令重放）' }
  if (ticket.riskClass !== expect.riskClass) return { ok: false, reason: '票据风险等级与实际判定不符' }

  const now = expect.now ?? Date.now()
  if (ticket.expiresAt <= now) return { ok: false, reason: '票据已过期' }

  const expected = sign(secret, payloadOf(ticket))
  const a = Buffer.from(ticket.signature, 'hex')
  const b = Buffer.from(expected, 'hex')
  if (a.length !== b.length || !timingSafeEqual(a, b)) return { ok: false, reason: '签名校验失败（伪造或密钥不符）' }

  return { ok: true }
}
