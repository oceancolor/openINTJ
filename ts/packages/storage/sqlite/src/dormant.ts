import type {
  DormantPersistenceAdapter,
  DormantSnapshot,
  InternalizationProposal,
  PassiveEvent,
  PersonaConfig,
} from "@openintj/dormant";
import {
  InternalizationProposalSchema,
  PassiveEventSchema,
  PersonaConfigSchema,
} from "@openintj/dormant";
import { z } from "zod";

export const SqliteDormantConfigSchema = z.object({
  /** dormant 库文件路径；":memory:" 走纯内存。 */
  dbPath: z.string(),
  /** WAL 模式（建议本地开启）。 */
  wal: z.boolean().default(true),
});
/** 内部规范化后的 config（wal 必填）。 */
export type SqliteDormantConfig = z.infer<typeof SqliteDormantConfigSchema>;
/** 装配点入参（wal 可选，进 ctor 后 zod 兜底为 true）。 */
export type SqliteDormantConfigInput = z.input<typeof SqliteDormantConfigSchema>;

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
    CREATE TABLE IF NOT EXISTS dormant_schema_version (
      version INTEGER PRIMARY KEY
    );

    CREATE TABLE IF NOT EXISTS dormant_events (
      eventId       TEXT PRIMARY KEY,
      ts            INTEGER NOT NULL,
      source        TEXT NOT NULL,
      text          TEXT NOT NULL,
      metadataJson  TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_devt_ts ON dormant_events(ts);
    CREATE INDEX IF NOT EXISTS idx_devt_source ON dormant_events(source);

    CREATE TABLE IF NOT EXISTS dormant_proposals (
      proposalId    TEXT PRIMARY KEY,
      patternJson   TEXT NOT NULL,
      targetField   TEXT NOT NULL,
      valueJson     TEXT NOT NULL,
      status        TEXT NOT NULL,
      ts            INTEGER NOT NULL,
      decidedAt     INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_dprop_status ON dormant_proposals(status);
    CREATE INDEX IF NOT EXISTS idx_dprop_ts ON dormant_proposals(ts);

    CREATE TABLE IF NOT EXISTS dormant_persona (
      id     INTEGER PRIMARY KEY CHECK (id = 1),
      json   TEXT NOT NULL
    );
  `,
};

interface EventRow {
  eventId: string;
  ts: number;
  source: string;
  text: string;
  metadataJson: string;
}

interface ProposalRow {
  proposalId: string;
  patternJson: string;
  targetField: string;
  valueJson: string;
  status: string;
  ts: number;
  decidedAt: number | null;
}

interface PersonaRow {
  id: number;
  json: string;
}

/**
 * SqliteDormantStore —— Dormant 子系统的 SQLite 持久化适配器。
 *
 * 与 {@link import("./sqlite.js").SqliteMetadataStore} 一样走 better-sqlite3 peer dep 路径，
 * 但**使用独立的数据库文件**（默认 `dormant.sqlite`）以避免与元数据/审计表耦合。
 *
 * 热路径（recordEvent / upsertProposal / savePersona / clearAll）走 better-sqlite3 同步 API，
 * 不抛错（写入失败仅 console.error，避免污染 agent 主循环）。
 */
export class SqliteDormantStore implements DormantPersistenceAdapter {
  readonly name: string;
  readonly config: SqliteDormantConfig;
  private db?: BetterSqliteDB;
  private stmtInsertEvent?: BetterSqliteStmt;
  private stmtUpsertProposal?: BetterSqliteStmt;
  private stmtUpsertPersona?: BetterSqliteStmt;
  private stmtSelectEvents?: BetterSqliteStmt;
  private stmtSelectProposals?: BetterSqliteStmt;
  private stmtSelectPersona?: BetterSqliteStmt;
  private stmtPruneOlderThan?: BetterSqliteStmt;
  private stmtPruneToMax?: BetterSqliteStmt;

  constructor(config: SqliteDormantConfigInput) {
    this.config = SqliteDormantConfigSchema.parse(config);
    this.name = `sqlite-dormant:${this.config.dbPath}`;
  }

  /**
   * 必须在使用前 await。
   *  1. dynamic import better-sqlite3（CI 无该 peer 时给出明确错误）
   *  2. 开 WAL（":memory:" 跳过）
   *  3. 执行 migration 至 TARGET_VERSION
   *  4. prepare 所有热路径 statements
   */
  async init(): Promise<void> {
    const moduleName = "better-sqlite3";
    const mod = (await import(moduleName).catch((e) => {
      throw new Error(
        `SqliteDormantStore: failed to load better-sqlite3 (peer dep). Install: pnpm add better-sqlite3. Cause: ${(e as Error).message}`,
      );
    })) as { default?: BetterSqliteCtor } & BetterSqliteCtor;
    const Ctor = (mod.default ?? mod) as BetterSqliteCtor;
    this.db = new Ctor(this.config.dbPath);
    if (this.config.wal && this.config.dbPath !== ":memory:") {
      this.db.pragma("journal_mode = WAL");
    }
    this.db.pragma("foreign_keys = ON");
    await this.migrate();
    this.prepareStatements();
  }

  private async migrate(): Promise<{ from: number; to: number }> {
    if (!this.db) throw new Error("SqliteDormantStore not initialized");
    this.db.exec(MIGRATIONS[1] ?? "");
    const row = this.db.prepare("SELECT version FROM dormant_schema_version").get() as
      | { version: number }
      | undefined;
    const currentVersion = row?.version ?? 0;
    if (currentVersion === TARGET_VERSION) {
      return { from: currentVersion, to: currentVersion };
    }
    for (let v = currentVersion + 1; v <= TARGET_VERSION; v++) {
      const sql = MIGRATIONS[v];
      if (sql) this.db.exec(sql);
    }
    this.db.exec("DELETE FROM dormant_schema_version");
    this.db.prepare("INSERT INTO dormant_schema_version (version) VALUES (?)").run(TARGET_VERSION);
    return { from: currentVersion, to: TARGET_VERSION };
  }

  private prepareStatements(): void {
    if (!this.db) throw new Error("SqliteDormantStore not initialized");
    this.stmtInsertEvent = this.db.prepare(
      `INSERT INTO dormant_events (eventId, ts, source, text, metadataJson)
       VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(eventId) DO UPDATE SET
         ts = excluded.ts,
         source = excluded.source,
         text = excluded.text,
         metadataJson = excluded.metadataJson`,
    );
    this.stmtUpsertProposal = this.db.prepare(
      `INSERT INTO dormant_proposals
         (proposalId, patternJson, targetField, valueJson, status, ts, decidedAt)
       VALUES (?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(proposalId) DO UPDATE SET
         patternJson = excluded.patternJson,
         targetField = excluded.targetField,
         valueJson = excluded.valueJson,
         status = excluded.status,
         ts = excluded.ts,
         decidedAt = excluded.decidedAt`,
    );
    this.stmtUpsertPersona = this.db.prepare(
      `INSERT INTO dormant_persona (id, json) VALUES (1, ?)
       ON CONFLICT(id) DO UPDATE SET json = excluded.json`,
    );
    this.stmtSelectEvents = this.db.prepare(
      "SELECT eventId, ts, source, text, metadataJson FROM dormant_events ORDER BY ts ASC",
    );
    this.stmtSelectProposals = this.db.prepare(
      "SELECT proposalId, patternJson, targetField, valueJson, status, ts, decidedAt FROM dormant_proposals ORDER BY ts ASC",
    );
    this.stmtSelectPersona = this.db.prepare("SELECT id, json FROM dormant_persona WHERE id = 1");
    this.stmtPruneOlderThan = this.db.prepare("DELETE FROM dormant_events WHERE ts < ?");
    this.stmtPruneToMax = this.db.prepare(
      `DELETE FROM dormant_events
       WHERE eventId NOT IN (
         SELECT eventId FROM dormant_events ORDER BY ts DESC, eventId DESC LIMIT ?
       )`,
    );
  }

  async loadAll(): Promise<DormantSnapshot> {
    if (
      !this.db ||
      !this.stmtSelectEvents ||
      !this.stmtSelectProposals ||
      !this.stmtSelectPersona
    ) {
      throw new Error("SqliteDormantStore not initialized");
    }
    const eventRows = this.stmtSelectEvents.all() as EventRow[];
    const events: PassiveEvent[] = eventRows.map((r) =>
      PassiveEventSchema.parse({
        eventId: r.eventId,
        ts: r.ts,
        source: r.source,
        text: r.text,
        metadata: safeJsonParse(r.metadataJson, {}),
      }),
    );

    const proposalRows = this.stmtSelectProposals.all() as ProposalRow[];
    const proposals: InternalizationProposal[] = proposalRows.map((r) =>
      InternalizationProposalSchema.parse({
        proposalId: r.proposalId,
        pattern: safeJsonParse(r.patternJson, {}),
        targetField: r.targetField,
        value: safeJsonParse(r.valueJson, null),
        status: r.status,
        ts: r.ts,
        ...(r.decidedAt !== null ? { decidedAt: r.decidedAt } : {}),
      }),
    );

    const personaRow = this.stmtSelectPersona.get() as PersonaRow | undefined;
    const persona: PersonaConfig | undefined = personaRow
      ? PersonaConfigSchema.parse(safeJsonParse(personaRow.json, {}))
      : undefined;

    return { events, proposals, persona };
  }

  recordEvent(event: PassiveEvent): void {
    if (!this.stmtInsertEvent) return;
    try {
      this.stmtInsertEvent.run(
        event.eventId,
        event.ts,
        event.source,
        event.text,
        JSON.stringify(event.metadata ?? {}),
      );
    } catch (e) {
      console.error("[SqliteDormantStore] recordEvent failed:", (e as Error).message);
    }
  }

  upsertProposal(proposal: InternalizationProposal): void {
    if (!this.stmtUpsertProposal) return;
    try {
      this.stmtUpsertProposal.run(
        proposal.proposalId,
        JSON.stringify(proposal.pattern),
        proposal.targetField,
        JSON.stringify(proposal.value ?? null),
        proposal.status,
        proposal.ts,
        proposal.decidedAt ?? null,
      );
    } catch (e) {
      console.error("[SqliteDormantStore] upsertProposal failed:", (e as Error).message);
    }
  }

  savePersona(persona: PersonaConfig): void {
    if (!this.stmtUpsertPersona) return;
    try {
      this.stmtUpsertPersona.run(JSON.stringify(persona));
    } catch (e) {
      console.error("[SqliteDormantStore] savePersona failed:", (e as Error).message);
    }
  }

  clearAll(): void {
    if (!this.db) return;
    try {
      this.db.exec(
        "DELETE FROM dormant_events; DELETE FROM dormant_proposals; DELETE FROM dormant_persona;",
      );
    } catch (e) {
      console.error("[SqliteDormantStore] clearAll failed:", (e as Error).message);
    }
  }

  pruneEvents(olderThanTs: number): number {
    if (!this.stmtPruneOlderThan) return 0;
    try {
      return this.stmtPruneOlderThan.run(olderThanTs).changes;
    } catch (e) {
      console.error("[SqliteDormantStore] pruneEvents failed:", (e as Error).message);
      return 0;
    }
  }

  pruneEventsToMax(maxRows: number): number {
    if (!this.stmtPruneToMax) return 0;
    const limit = Math.max(0, Math.floor(maxRows));
    try {
      return this.stmtPruneToMax.run(limit).changes;
    } catch (e) {
      console.error("[SqliteDormantStore] pruneEventsToMax failed:", (e as Error).message);
      return 0;
    }
  }

  async close(): Promise<void> {
    this.db?.close();
    delete this.db;
    delete this.stmtInsertEvent;
    delete this.stmtUpsertProposal;
    delete this.stmtUpsertPersona;
    delete this.stmtSelectEvents;
    delete this.stmtSelectProposals;
    delete this.stmtSelectPersona;
    delete this.stmtPruneOlderThan;
    delete this.stmtPruneToMax;
  }
}

const safeJsonParse = <T>(raw: string, fallback: T): T => {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

/**
 * 工厂方法 —— 一次性 new + init，方便装配点直接 await。
 */
export const createSqliteDormantStore = async (
  config: SqliteDormantConfigInput,
): Promise<SqliteDormantStore> => {
  const s = new SqliteDormantStore(config);
  await s.init();
  return s;
};
