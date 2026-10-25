/**
 * 数据库表结构 —— 单一事实来源
 *
 * 两处会用到同一张库：
 *   - P-03 agent-host：事件流（t_event）、检查点（t_checkpoint）
 *   - P-04 tool-host：审批审计（t_approval_audit）—— 由执行方自己写，调用方绕不过
 *
 * 因此 DDL 必须共享，不能各写一份各的，否则表结构会在两个进程里漂移。
 * 全部 IF NOT EXISTS：任一进程先启动都能建表，后启动的不会覆盖。
 */

import type Database from 'better-sqlite3'

export const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS t_event (
  event_id       TEXT PRIMARY KEY,
  run_id         TEXT    NOT NULL,
  seq            INTEGER NOT NULL,
  trace_id       TEXT    NOT NULL,
  tenant_id      TEXT    NOT NULL,
  agent_id       TEXT    NOT NULL,
  kind           TEXT    NOT NULL,
  type           TEXT    NOT NULL,
  at             INTEGER NOT NULL,
  redaction      TEXT    NOT NULL,
  payload        TEXT    NOT NULL,
  schema_version INTEGER NOT NULL,
  UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_event_run_seq ON t_event (run_id, seq);

CREATE TABLE IF NOT EXISTS t_checkpoint (
  checkpoint_id TEXT PRIMARY KEY,
  run_id        TEXT    NOT NULL,
  at_seq        INTEGER NOT NULL,
  scope         TEXT    NOT NULL,
  at            INTEGER NOT NULL,
  label         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_checkpoint_run ON t_checkpoint (run_id, at_seq);

CREATE TABLE IF NOT EXISTS t_approval_audit (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id        TEXT    NOT NULL,
  run_id          TEXT    NOT NULL,
  command         TEXT    NOT NULL,
  risk_class      TEXT    NOT NULL,
  tier            TEXT    NOT NULL,
  outcome         TEXT    NOT NULL,
  basis           TEXT    NOT NULL,
  matched_rule_id TEXT    NOT NULL,
  executed        INTEGER NOT NULL,
  at              INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON t_approval_audit (run_id, at);

CREATE TABLE IF NOT EXISTS t_schema_meta (
  k TEXT PRIMARY KEY,
  v TEXT NOT NULL
);

-- 应用级设置（D9）。首版只有审批档位，但做成 key-value 是为了不再为每加一项设置动表结构。
--
-- 为什么不复用 t_schema_meta：那张表是「库结构自己的元数据」，
-- 混进业务配置之后，「升级库」和「改用户设置」就共用了同一条写入路径，
-- 一旦谁写坏了，你分不清是迁移出问题还是用户配错了。
CREATE TABLE IF NOT EXISTS t_app_setting (
  k          TEXT PRIMARY KEY,
  v          TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
`

/** 当前表结构版本。新增列 / 新表时递增，并在 applyMigrations 里补一条迁移 */
export const SCHEMA_VERSION = 3

/** 设置的键名 —— 单一事实来源，避免各进程各写一套字符串 */
export const SETTING_KEYS = {
  /** 审批档位：'strict' | 'standard' | 'autonomous' */
  approvalTier: 'approval.tier'
} as const

/**
 * 迁移。
 *
 * 为什么必须有：`CREATE TABLE IF NOT EXISTS` 对已存在的表**不会补列**。
 * 开发期改一次表结构，旧库就会在运行时报 “no column named xxx”，
 * 而这时最容易被误判成代码 bug —— 实际是库没升级。
 */
export function applyMigrations(db: Database.Database): void {
  // v1 → v2：审批审计增加 executed 列（区分「放行执行」与「拦下未执行」）
  const cols = db.prepare('PRAGMA table_info(t_approval_audit)').all() as Array<{ name: string }>
  if (cols.length > 0 && !cols.some((c) => c.name === 'executed')) {
    db.exec('ALTER TABLE t_approval_audit ADD COLUMN executed INTEGER NOT NULL DEFAULT 0')
  }

  // v2 → v3：新增 t_app_setting（档位持久化）。
  // 不需要在这里写 CREATE TABLE：两个 store 都是「先 exec(SCHEMA_SQL) 再 applyMigrations」，
  // 而 SCHEMA_SQL 全是 IF NOT EXISTS，所以老库下次打开时新表会自动出现。
  // 只有「给已存在的表补列」才必须显式迁移（见上面的 executed 列）。

  db.prepare('INSERT INTO t_schema_meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v').run(
    'version',
    String(SCHEMA_VERSION)
  )
}
