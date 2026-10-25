/**
 * 审批审计落库（P-04 自己写）
 *
 * 为什么由执行方写而不是调用方写：
 *   审计的意义是「即使 Agent 想赖账，记录也在」。如果让 agent-host 写，
 *   它可以选择不写、或写一份对不上的 —— 那审计就成了自我申报。
 *
 * 失败策略：审计写不进去就不执行。
 *   这比「先跑了再说」严，但正是《系统设计》§3.2.M10 的意思：
 *   六要素缺一不可（B100003），缺失即拒绝执行。
 */

import Database from 'better-sqlite3'
import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import type { RiskDecisionContext } from '../shared/approval'
import { redactText } from '../shared/redaction'
import { applyMigrations, SCHEMA_SQL } from '../shared/schema'
import { ulid } from '../shared/ulid'

export class AuditLog {
  private readonly db: Database.Database

  constructor(dbPath: string) {
    mkdirSync(dirname(dbPath), { recursive: true })
    this.db = new Database(dbPath)
    this.db.pragma('journal_mode = WAL')
    this.db.pragma('synchronous = NORMAL')
    this.db.exec(SCHEMA_SQL)
    applyMigrations(this.db)
  }

  /** 记录一条裁决。executed=false 表示被拦下（拦下同样要留痕） */
  record(runId: string, ctx: RiskDecisionContext, executed: boolean): string {
    // 六要素必须齐 —— 缺任一项宁可抛错，也不写一条查不清的审计
    const missing = (['command', 'riskClass', 'tier', 'outcome', 'basis', 'matchedRuleId'] as const).filter(
      (k) => ctx[k] === undefined || ctx[k] === null || ctx[k] === ''
    )
    if (missing.length > 0) {
      throw new Error(`拒绝执行：审计字段缺失 ${missing.join(',')}（B100003）`)
    }

    const eventId = ulid()
    // 命令本身也过一遍密钥脱敏。
    //
    // 为什么审计也要脱：`curl -H "Authorization: Bearer sk-..."` 这种命令很常见，
    // 把 key 原样存进库里是纯风险，而审计关心的是「执行了什么类型的命令、在哪、风险多高」，
    // 不关心「用的哪把 key」。脱敏只替换密钥字面量，命令结构完整保留，仍然可读可核验。
    const command = redactText(ctx.command, 'secret')
    this.db
      .prepare(
        `INSERT INTO t_approval_audit
           (event_id, run_id, command, risk_class, tier, outcome, basis, matched_rule_id, executed, at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        eventId,
        runId,
        command,
        ctx.riskClass,
        ctx.tier,
        ctx.outcome,
        ctx.basis,
        ctx.matchedRuleId,
        executed ? 1 : 0,
        Date.now()
      )
    return eventId
  }

  close(): void {
    this.db.close()
  }

  /**
   * 读最近若干条审计（倒序）。
   *
   * 存在意义不是「调试用」：审计如果不能被读出来，它就只是写给自己的日记。
   * 有了这条读取路径，探针才能从**库文件**里核实「那条被拒的调用真的入库了」，
   * 而不是只相信 API 返回了一个 ok。
   */
  recent(runId: string, limit = 20): AuditRow[] {
    const rows = this.db
      .prepare(
        `SELECT event_id, run_id, command, risk_class, tier, outcome, basis, matched_rule_id, executed, at
           FROM t_approval_audit WHERE run_id = ? ORDER BY at DESC LIMIT ?`
      )
      .all(runId, limit) as Record<string, unknown>[]

    return rows.map((r) => ({
      eventId: String(r['event_id']),
      runId: String(r['run_id']),
      command: String(r['command']),
      riskClass: String(r['risk_class']) as AuditRow['riskClass'],
      tier: String(r['tier']) as AuditRow['tier'],
      outcome: String(r['outcome']) as AuditRow['outcome'],
      basis: String(r['basis']) as AuditRow['basis'],
      matchedRuleId: String(r['matched_rule_id']),
      executed: Number(r['executed']) === 1,
      at: Number(r['at'])
    }))
  }
}

export interface AuditRow {
  eventId: string
  runId: string
  command: string
  riskClass: RiskDecisionContext['riskClass']
  tier: RiskDecisionContext['tier']
  outcome: RiskDecisionContext['outcome']
  basis: RiskDecisionContext['basis']
  matchedRuleId: string
  executed: boolean
  at: number
}
