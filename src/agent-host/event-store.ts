/**
 * 事件存储（SQLite 实现）
 *
 * 依据《系统设计》§4：
 *   - 事件流是状态的单一事实来源（F5）
 *   - 全量按序落盘，seq 单调递增，作为恢复位点
 *   - 检查点在里程碑边界与关键节点强制生成（F6）
 *   - 审批裁决必须留痕（RiskDecisionContext 六要素，缺一不可，错误码 B100003）
 *
 * 与首版的差别：首版是内存 Map，进程重启即丢；本版落 SQLite，恢复与审计才有意义。
 *
 * 失败策略：**打开或建表失败即 fail-stop**。
 *   事件流是单一事实来源，落不了盘就不该继续跑 —— 悄悄退化成内存存储，
 *   等于让「审计可查」这件事在没有提示的情况下失效。
 *   仅当显式设置 `OPENINTJ_EVENT_STORE=memory` 时才使用内存实现（供无原生模块的环境用）。
 */

import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import Database from 'better-sqlite3'
import type {
  ActionType,
  AppendResult,
  Checkpoint,
  CheckpointScope,
  EventEnvelope,
  ObservationType,
  Redaction
} from '../shared/events'
import type { RiskDecisionContext } from '../shared/approval'
import { applyMigrations, SCHEMA_SQL } from '../shared/schema'
import { ulid } from '../shared/ulid'

export interface AppendInput {
  runId: string
  traceId: string
  agentId: string
  kind: 'action' | 'observation'
  type: ActionType | ObservationType
  payload: unknown
  redaction?: Redaction
}

/** 存储接口 —— 编排层只依赖这个，不关心底层是 SQLite 还是内存 */
export interface EventStoreLike {
  append(input: AppendInput): AppendResult
  list(runId: string, fromSeq?: number): EventEnvelope[]
  createCheckpoint(runId: string, scope: CheckpointScope, label: string): Checkpoint
  listCheckpoints(runId: string): Checkpoint[]
  appendApprovalAudit(eventId: string, runId: string, ctx: RiskDecisionContext, executed: boolean): void
  size(runId: string): number
  /** 读一项应用设置；不存在返回 null */
  getSetting(key: string): string | null
  /** 写一项应用设置（upsert） */
  setSetting(key: string, value: string): void
  close(): void
}

/* ------------------------------------------------------------------ *
 * SQLite 实现
 * ------------------------------------------------------------------ */

interface EventRow {
  event_id: string
  run_id: string
  seq: number
  trace_id: string
  tenant_id: string
  agent_id: string
  kind: string
  type: string
  at: number
  redaction: string
  payload: string
  schema_version: number
}

interface CheckpointRow {
  checkpoint_id: string
  run_id: string
  at_seq: number
  scope: string
  at: number
  label: string
}


export class SqliteEventStore implements EventStoreLike {
  private readonly db: Database.Database
  private readonly tenantId: string

  constructor(dbPath: string, tenantId = 'local') {
    mkdirSync(dirname(dbPath), { recursive: true })
    this.db = new Database(dbPath)
    // WAL：读写不互相阻塞；NORMAL 同步在崩溃时可能丢最后几条，但不会损坏库
    this.db.pragma('journal_mode = WAL')
    this.db.pragma('synchronous = NORMAL')
    this.db.pragma('foreign_keys = ON')
    this.db.exec(SCHEMA_SQL)
    applyMigrations(this.db)
    this.tenantId = tenantId
  }

  append(input: AppendInput): AppendResult {
    const now = Date.now()
    const eventId = ulid(now)

    const insert = this.db.transaction((): number => {
      const row = this.db
        .prepare('SELECT COALESCE(MAX(seq), 0) AS maxSeq FROM t_event WHERE run_id = ?')
        .get(input.runId) as { maxSeq: number }
      const seq = row.maxSeq + 1
      this.db
        .prepare(
          `INSERT INTO t_event
             (event_id, run_id, seq, trace_id, tenant_id, agent_id, kind, type, at, redaction, payload, schema_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          eventId,
          input.runId,
          seq,
          input.traceId,
          this.tenantId,
          input.agentId,
          input.kind,
          input.type,
          now,
          input.redaction ?? 'none',
          JSON.stringify(input.payload ?? null),
          1
        )
      return seq
    })

    return { seq: insert(), eventId }
  }

  list(runId: string, fromSeq = 0): EventEnvelope[] {
    const rows = this.db
      .prepare('SELECT * FROM t_event WHERE run_id = ? AND seq > ? ORDER BY seq ASC')
      .all(runId, fromSeq) as EventRow[]
    return rows.map(toEnvelope)
  }

  createCheckpoint(runId: string, scope: CheckpointScope, label: string): Checkpoint {
    const cp: Checkpoint = {
      checkpointId: ulid(),
      runId,
      atSeq: this.size(runId),
      scope,
      at: Date.now(),
      label
    }
    this.db
      .prepare(
        'INSERT INTO t_checkpoint (checkpoint_id, run_id, at_seq, scope, at, label) VALUES (?, ?, ?, ?, ?, ?)'
      )
      .run(cp.checkpointId, cp.runId, cp.atSeq, cp.scope, cp.at, cp.label)
    return cp
  }

  listCheckpoints(runId: string): Checkpoint[] {
    const rows = this.db
      .prepare('SELECT * FROM t_checkpoint WHERE run_id = ? ORDER BY at_seq ASC, at ASC')
      .all(runId) as CheckpointRow[]
    return rows.map((r) => ({
      checkpointId: r.checkpoint_id,
      runId: r.run_id,
      atSeq: r.at_seq,
      scope: r.scope as CheckpointScope,
      at: r.at,
      label: r.label
    }))
  }

  appendApprovalAudit(eventId: string, runId: string, ctx: RiskDecisionContext, executed: boolean): void {
    this.db
      .prepare(
        `INSERT INTO t_approval_audit
           (event_id, run_id, command, risk_class, tier, outcome, basis, matched_rule_id, executed, at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        eventId,
        runId,
        ctx.command,
        ctx.riskClass,
        ctx.tier,
        ctx.outcome,
        ctx.basis,
        ctx.matchedRuleId,
        executed ? 1 : 0,
        Date.now()
      )
  }

  size(runId: string): number {
    const row = this.db.prepare('SELECT COUNT(*) AS c FROM t_event WHERE run_id = ?').get(runId) as {
      c: number
    }
    return row.c
  }

  getSetting(key: string): string | null {
    const row = this.db.prepare('SELECT v FROM t_app_setting WHERE k = ?').get(key) as
      | { v: string }
      | undefined
    return row?.v ?? null
  }

  setSetting(key: string, value: string): void {
    this.db
      .prepare(
        `INSERT INTO t_app_setting (k, v, updated_at) VALUES (?, ?, ?)
           ON CONFLICT(k) DO UPDATE SET v = excluded.v, updated_at = excluded.updated_at`
      )
      .run(key, value, Date.now())
  }

  close(): void {
    this.db.close()
  }
}

function toEnvelope(r: EventRow): EventEnvelope {
  return {
    eventId: r.event_id,
    seq: r.seq,
    runId: r.run_id,
    traceId: r.trace_id,
    tenantId: r.tenant_id,
    agentId: r.agent_id,
    kind: r.kind as EventEnvelope['kind'],
    type: r.type as EventEnvelope['type'],
    at: r.at,
    redaction: r.redaction as Redaction,
    payload: safeParse(r.payload),
    schemaVersion: 1
  }
}

function safeParse(s: string): unknown {
  try {
    return JSON.parse(s)
  } catch {
    // 落盘内容损坏时不要抛断读取 —— 原样返回并交由上层判断
    return { __unparsable: s }
  }
}

/* ------------------------------------------------------------------ *
 * 内存实现（仅供无原生模块的环境；显式开启，绝不静默降级）
 * ------------------------------------------------------------------ */

export class MemoryEventStore implements EventStoreLike {
  private readonly byRun = new Map<string, EventEnvelope[]>()
  private readonly checkpoints = new Map<string, Checkpoint[]>()
  private readonly tenantId: string

  constructor(tenantId = 'local') {
    this.tenantId = tenantId
  }

  append(input: AppendInput): AppendResult {
    const list = this.byRun.get(input.runId) ?? []
    const now = Date.now()
    const env: EventEnvelope = {
      eventId: ulid(now),
      seq: list.length + 1,
      runId: input.runId,
      traceId: input.traceId,
      tenantId: this.tenantId,
      agentId: input.agentId,
      kind: input.kind,
      type: input.type,
      at: now,
      redaction: input.redaction ?? 'none',
      payload: input.payload,
      schemaVersion: 1
    }
    list.push(env)
    this.byRun.set(input.runId, list)
    return { seq: env.seq, eventId: env.eventId }
  }

  list(runId: string, fromSeq = 0): EventEnvelope[] {
    return (this.byRun.get(runId) ?? []).filter((e) => e.seq > fromSeq)
  }

  createCheckpoint(runId: string, scope: CheckpointScope, label: string): Checkpoint {
    const cp: Checkpoint = {
      checkpointId: ulid(),
      runId,
      atSeq: (this.byRun.get(runId) ?? []).length,
      scope,
      at: Date.now(),
      label
    }
    const arr = this.checkpoints.get(runId) ?? []
    arr.push(cp)
    this.checkpoints.set(runId, arr)
    return cp
  }

  listCheckpoints(runId: string): Checkpoint[] {
    return [...(this.checkpoints.get(runId) ?? [])]
  }

  appendApprovalAudit(
    _eventId: string,
    _runId: string,
    _ctx: RiskDecisionContext,
    _executed: boolean
  ): void {
    // 内存实现不保留审计表：调用方需知悉这一点（见 openStore 的日志）
  }

  size(runId: string): number {
    return (this.byRun.get(runId) ?? []).length
  }

  getSetting(_key: string): string | null {
    // 内存实现不持久化 —— 返回 null 会让上层拿默认值，语义上是「没存过」，不是「存了空」
    return null
  }

  setSetting(_key: string, _value: string): void {
    // 同上：这个实现存在的意义就是「跑得起来但不落盘」，不该假装存住了
  }

  close(): void {
    this.byRun.clear()
    this.checkpoints.clear()
  }
}

/* ------------------------------------------------------------------ *
 * 工厂
 * ------------------------------------------------------------------ */

/**
 * 打开事件存储。
 * @param dataDir 数据目录（P-01 经 OPENINTJ_DATA_DIR 下发）
 * @throws 除显式内存模式外，打开失败直接抛出 —— 由调用方 fail-stop
 */
export function openStore(dataDir: string): EventStoreLike {
  if (process.env['OPENINTJ_EVENT_STORE'] === 'memory') {
    console.warn('[event-store] 使用内存实现（OPENINTJ_EVENT_STORE=memory）：重启即丢，仅限调试')
    return new MemoryEventStore()
  }
  const dbPath = `${dataDir}/openintj.db`
  const store = new SqliteEventStore(dbPath)
  console.log(`[event-store] SQLite 已就绪：${dbPath}`)
  return store
}
