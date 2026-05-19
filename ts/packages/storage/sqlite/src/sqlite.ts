import { z } from "zod";
import {
  type AuditRow,
  AuditRowSchema,
  type FragmentMeta,
  FragmentMetaSchema,
  type MetadataStore,
  type SessionRow,
  SessionRowSchema,
} from "./types.js";

export const SqliteMetadataConfigSchema = z.object({
  /** 数据库文件路径；":memory:" 用内存数据库测试。 */
  dbPath: z.string(),
  /** 是否启用 WAL（建议开启用于本地）。 */
  wal: z.boolean().default(true),
});
export type SqliteMetadataConfig = z.infer<typeof SqliteMetadataConfigSchema>;

interface BetterSqliteStmt {
  run(...params: unknown[]): { changes: number; lastInsertRowid: number };
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
}
interface BetterSqliteDB {
  exec(sql: string): void;
  prepare(sql: string): BetterSqliteStmt;
  pragma(name: string): unknown;
  transaction<T>(fn: (...args: unknown[]) => T): (...args: unknown[]) => T;
  close(): void;
}
type BetterSqliteCtor = new (filename: string, opts?: Record<string, unknown>) => BetterSqliteDB;

const TARGET_VERSION = 1;

const MIGRATIONS: Record<number, string> = {
  1: `
    CREATE TABLE IF NOT EXISTS schema_version (
      version INTEGER PRIMARY KEY
    );
    CREATE TABLE IF NOT EXISTS fragments_meta (
      fragmentId      TEXT PRIMARY KEY,
      memoryType      TEXT NOT NULL,
      importance      REAL NOT NULL,
      contentHash     TEXT NOT NULL,
      taskTagsCsv     TEXT NOT NULL DEFAULT '',
      metadataJson    TEXT NOT NULL DEFAULT '{}',
      summariesJson   TEXT NOT NULL DEFAULT '{}',
      timestamp       REAL NOT NULL,
      accessCount     INTEGER NOT NULL DEFAULT 0,
      lastAccessed    REAL NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_frag_memory_type ON fragments_meta(memoryType);
    CREATE INDEX IF NOT EXISTS idx_frag_timestamp ON fragments_meta(timestamp);

    CREATE TABLE IF NOT EXISTS audit_events (
      eventId      TEXT PRIMARY KEY,
      eventType    TEXT NOT NULL,
      command      TEXT,
      riskLevel    TEXT,
      approved     INTEGER,
      reason       TEXT,
      metadataJson TEXT NOT NULL DEFAULT '{}',
      timestamp    REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_events(eventType);
    CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_events(timestamp);

    CREATE TABLE IF NOT EXISTS sessions (
      sessionId     TEXT PRIMARY KEY,
      startedAt     REAL NOT NULL,
      lastActiveAt  REAL NOT NULL,
      metadataJson  TEXT NOT NULL DEFAULT '{}'
    );
  `,
};

/**
 * SqliteMetadataStore —— 基于 better-sqlite3 的本地元数据存储。
 *
 * 特性：
 * - 同步底层（better-sqlite3 全部 API 同步），但接口暴露 async 以兼容 MetadataStore
 * - WAL 默认开启
 * - 重入式 migrate（schema_version 版本号自管理）
 * - peer dep（让 CI 在没装 better-sqlite3 时也能编译/类型通过）
 */
export class SqliteMetadataStore implements MetadataStore {
  readonly name: string;
  readonly config: SqliteMetadataConfig;
  private db?: BetterSqliteDB;

  constructor(config: SqliteMetadataConfig) {
    this.config = SqliteMetadataConfigSchema.parse(config);
    this.name = `sqlite:${this.config.dbPath}`;
  }

  async init(): Promise<void> {
    const moduleName = "better-sqlite3";
    const mod = (await import(moduleName).catch((e) => {
      throw new Error(
        `SqliteMetadataStore: failed to load better-sqlite3 (peer dep). Install: pnpm add better-sqlite3. Cause: ${(e as Error).message}`,
      );
    })) as { default?: BetterSqliteCtor } & BetterSqliteCtor;
    const Ctor = (mod.default ?? mod) as BetterSqliteCtor;
    this.db = new Ctor(this.config.dbPath);
    if (this.config.wal && this.config.dbPath !== ":memory:") {
      this.db.pragma("journal_mode = WAL");
    }
    this.db.pragma("foreign_keys = ON");
  }

  async migrate(): Promise<{ from: number; to: number }> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    this.db.exec(MIGRATIONS[1] ?? "");
    const row = this.db.prepare("SELECT version FROM schema_version").get() as
      | { version: number }
      | undefined;
    const currentVersion = row?.version ?? 0;
    if (currentVersion === TARGET_VERSION) {
      return { from: currentVersion, to: currentVersion };
    }
    // 此处只有 v1，未来加 v2/v3 时按版本号顺序执行
    for (let v = currentVersion + 1; v <= TARGET_VERSION; v++) {
      const sql = MIGRATIONS[v];
      if (sql) this.db.exec(sql);
    }
    this.db.exec("DELETE FROM schema_version");
    this.db.prepare("INSERT INTO schema_version (version) VALUES (?)").run(TARGET_VERSION);
    return { from: currentVersion, to: TARGET_VERSION };
  }

  async putFragmentMeta(rows: readonly FragmentMeta[]): Promise<void> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    if (rows.length === 0) return;
    const stmt = this.db.prepare(`
      INSERT INTO fragments_meta
        (fragmentId, memoryType, importance, contentHash, taskTagsCsv, metadataJson, summariesJson, timestamp, accessCount, lastAccessed)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(fragmentId) DO UPDATE SET
        memoryType = excluded.memoryType,
        importance = excluded.importance,
        contentHash = excluded.contentHash,
        taskTagsCsv = excluded.taskTagsCsv,
        metadataJson = excluded.metadataJson,
        summariesJson = excluded.summariesJson,
        timestamp = excluded.timestamp,
        accessCount = excluded.accessCount,
        lastAccessed = excluded.lastAccessed
    `);
    const tx = this.db.transaction((items: unknown) => {
      for (const r of items as FragmentMeta[]) {
        stmt.run(
          r.fragmentId,
          r.memoryType,
          r.importance,
          r.contentHash,
          r.taskTagsCsv,
          r.metadataJson,
          r.summariesJson,
          r.timestamp,
          r.accessCount,
          r.lastAccessed,
        );
      }
    });
    tx(rows.map((r) => FragmentMetaSchema.parse(r)));
  }

  async getFragmentMeta(fragmentId: string): Promise<FragmentMeta | undefined> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    const row = this.db
      .prepare("SELECT * FROM fragments_meta WHERE fragmentId = ?")
      .get(fragmentId);
    return row ? FragmentMetaSchema.parse(row) : undefined;
  }

  async listFragmentMeta(
    opts: {
      memoryType?: "short_term" | "working" | "long_term";
      limit?: number;
    } = {},
  ): Promise<FragmentMeta[]> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    const where = opts.memoryType ? "WHERE memoryType = ?" : "";
    const limit = opts.limit !== undefined ? `LIMIT ${opts.limit}` : "";
    const sql = `SELECT * FROM fragments_meta ${where} ORDER BY timestamp DESC ${limit}`;
    const stmt = this.db.prepare(sql);
    const rows = (opts.memoryType ? stmt.all(opts.memoryType) : stmt.all()) as unknown[];
    return rows.map((r) => FragmentMetaSchema.parse(r));
  }

  async deleteFragmentMeta(fragmentIds: readonly string[]): Promise<number> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    if (fragmentIds.length === 0) return 0;
    const placeholders = fragmentIds.map(() => "?").join(",");
    const r = this.db
      .prepare(`DELETE FROM fragments_meta WHERE fragmentId IN (${placeholders})`)
      .run(...fragmentIds);
    return r.changes;
  }

  async recordAudit(row: AuditRow): Promise<void> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    const v = AuditRowSchema.parse(row);
    this.db
      .prepare(
        `INSERT INTO audit_events
          (eventId, eventType, command, riskLevel, approved, reason, metadataJson, timestamp)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        v.eventId,
        v.eventType,
        v.command,
        v.riskLevel,
        v.approved,
        v.reason,
        v.metadataJson,
        v.timestamp,
      );
  }

  async queryAudit(
    opts: { eventType?: string; since?: number; limit?: number } = {},
  ): Promise<AuditRow[]> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    const where: string[] = [];
    const args: unknown[] = [];
    if (opts.eventType) {
      where.push("eventType = ?");
      args.push(opts.eventType);
    }
    if (opts.since !== undefined) {
      where.push("timestamp >= ?");
      args.push(opts.since);
    }
    const whereSql = where.length ? `WHERE ${where.join(" AND ")}` : "";
    const limit = opts.limit !== undefined ? `LIMIT ${opts.limit}` : "";
    const sql = `SELECT * FROM audit_events ${whereSql} ORDER BY timestamp DESC ${limit}`;
    const rows = this.db.prepare(sql).all(...args) as unknown[];
    return rows.map((r) => AuditRowSchema.parse(r));
  }

  async pruneAudit(beforeTimestamp: number): Promise<number> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    const r = this.db.prepare("DELETE FROM audit_events WHERE timestamp < ?").run(beforeTimestamp);
    return r.changes;
  }

  async putSession(row: SessionRow): Promise<void> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    const v = SessionRowSchema.parse(row);
    this.db
      .prepare(
        `INSERT INTO sessions (sessionId, startedAt, lastActiveAt, metadataJson)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(sessionId) DO UPDATE SET
           lastActiveAt = excluded.lastActiveAt,
           metadataJson = excluded.metadataJson`,
      )
      .run(v.sessionId, v.startedAt, v.lastActiveAt, v.metadataJson);
  }

  async getSession(sessionId: string): Promise<SessionRow | undefined> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    const r = this.db.prepare("SELECT * FROM sessions WHERE sessionId = ?").get(sessionId);
    return r ? SessionRowSchema.parse(r) : undefined;
  }

  async touchSession(sessionId: string, ts: number): Promise<void> {
    if (!this.db) throw new Error("SqliteMetadataStore not initialized");
    this.db.prepare("UPDATE sessions SET lastActiveAt = ? WHERE sessionId = ?").run(ts, sessionId);
  }

  async close(): Promise<void> {
    this.db?.close();
    delete this.db;
  }
}
