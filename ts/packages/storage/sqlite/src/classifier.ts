import type { ClassifierState, ClassifierStore, Exemplar } from "@openintj/classifier";
import { z } from "zod";

/**
 * SqliteClassifierStore —— ReinforcingClassifier 状态的 SQLite 持久化适配器。
 *
 * 设计同 {@link import("./dormant.js").SqliteDormantStore}：走 better-sqlite3 peer dep、
 * 独立库文件（默认 classifier.sqlite）、热路径同步不抛错。
 *
 * 存储模型：exemplar 一行一条（vector 存 JSON 数组），save() 用事务全量覆盖
 * （exemplar 量级有上限 maxExemplars，全量重写简单且足够快）。
 */

export const SqliteClassifierConfigSchema = z.object({
  dbPath: z.string(),
  wal: z.boolean().default(true),
});
export type SqliteClassifierConfig = z.infer<typeof SqliteClassifierConfigSchema>;
export type SqliteClassifierConfigInput = z.input<typeof SqliteClassifierConfigSchema>;

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
    CREATE TABLE IF NOT EXISTS classifier_schema_version (
      version INTEGER PRIMARY KEY
    );
    CREATE TABLE IF NOT EXISTS classifier_exemplars (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      label       TEXT NOT NULL,
      weight      REAL NOT NULL,
      lastUsed    REAL NOT NULL,
      vectorJson  TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_clf_label ON classifier_exemplars(label);
  `,
};

interface ExemplarRow {
  label: string;
  weight: number;
  lastUsed: number;
  vectorJson: string;
}

export class SqliteClassifierStore implements ClassifierStore {
  readonly name: string;
  readonly config: SqliteClassifierConfig;
  private db?: BetterSqliteDB;
  private stmtInsert?: BetterSqliteStmt;
  private stmtSelectAll?: BetterSqliteStmt;
  private stmtDeleteAll?: BetterSqliteStmt;

  constructor(config: SqliteClassifierConfigInput) {
    this.config = SqliteClassifierConfigSchema.parse(config);
    this.name = `sqlite-classifier:${this.config.dbPath}`;
  }

  async init(): Promise<void> {
    const moduleName = "better-sqlite3";
    const mod = (await import(moduleName).catch((e) => {
      throw new Error(
        `SqliteClassifierStore: failed to load better-sqlite3 (peer dep). Install: pnpm add better-sqlite3. Cause: ${(e as Error).message}`,
      );
    })) as { default?: BetterSqliteCtor } & BetterSqliteCtor;
    const Ctor = (mod.default ?? mod) as BetterSqliteCtor;
    this.db = new Ctor(this.config.dbPath);
    if (this.config.wal && this.config.dbPath !== ":memory:") {
      this.db.pragma("journal_mode = WAL");
    }
    await this.migrate();
    this.prepareStatements();
  }

  private async migrate(): Promise<void> {
    if (!this.db) throw new Error("SqliteClassifierStore not initialized");
    this.db.exec(MIGRATIONS[1] ?? "");
    const row = this.db.prepare("SELECT version FROM classifier_schema_version").get() as
      | { version: number }
      | undefined;
    const current = row?.version ?? 0;
    if (current === TARGET_VERSION) return;
    for (let v = current + 1; v <= TARGET_VERSION; v++) {
      const sql = MIGRATIONS[v];
      if (sql) this.db.exec(sql);
    }
    this.db.exec("DELETE FROM classifier_schema_version");
    this.db
      .prepare("INSERT INTO classifier_schema_version (version) VALUES (?)")
      .run(TARGET_VERSION);
  }

  private prepareStatements(): void {
    if (!this.db) throw new Error("SqliteClassifierStore not initialized");
    this.stmtInsert = this.db.prepare(
      "INSERT INTO classifier_exemplars (label, weight, lastUsed, vectorJson) VALUES (?, ?, ?, ?)",
    );
    this.stmtSelectAll = this.db.prepare(
      "SELECT label, weight, lastUsed, vectorJson FROM classifier_exemplars",
    );
    this.stmtDeleteAll = this.db.prepare("DELETE FROM classifier_exemplars");
  }

  async load(): Promise<ClassifierState | undefined> {
    if (!this.stmtSelectAll) throw new Error("SqliteClassifierStore not initialized");
    const rows = this.stmtSelectAll.all() as ExemplarRow[];
    if (rows.length === 0) return undefined;
    const exemplars: Exemplar[] = [];
    for (const r of rows) {
      const vector = safeJsonParse<number[]>(r.vectorJson, []);
      if (vector.length === 0) continue;
      exemplars.push({
        vector,
        label: r.label as Exemplar["label"],
        weight: r.weight,
        lastUsed: r.lastUsed,
      });
    }
    return { exemplars };
  }

  save(state: ClassifierState): void {
    if (!this.db || !this.stmtDeleteAll || !this.stmtInsert) return;
    try {
      const tx = this.db.transaction(() => {
        this.stmtDeleteAll!.run();
        for (const e of state.exemplars) {
          this.stmtInsert!.run(e.label, e.weight, e.lastUsed, JSON.stringify(e.vector));
        }
      });
      tx();
    } catch (e) {
      console.error("[SqliteClassifierStore] save failed:", (e as Error).message);
    }
  }

  clear(): void {
    if (!this.stmtDeleteAll) return;
    try {
      this.stmtDeleteAll.run();
    } catch (e) {
      console.error("[SqliteClassifierStore] clear failed:", (e as Error).message);
    }
  }

  async close(): Promise<void> {
    this.db?.close();
    delete this.db;
    delete this.stmtInsert;
    delete this.stmtSelectAll;
    delete this.stmtDeleteAll;
  }
}

const safeJsonParse = <T>(raw: string, fallback: T): T => {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

export const createSqliteClassifierStore = async (
  config: SqliteClassifierConfigInput,
): Promise<SqliteClassifierStore> => {
  const s = new SqliteClassifierStore(config);
  await s.init();
  return s;
};
