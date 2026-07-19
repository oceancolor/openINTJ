import type { StoredTaskRun, TaskStore } from "@openintj/taskpool";

interface Statement {
  run(...params: unknown[]): unknown;
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
}
interface Database {
  exec(sql: string): void;
  prepare(sql: string): Statement;
  pragma(value: string): unknown;
  close(): void;
}
type DatabaseCtor = new (filename: string) => Database;

/**
 * SQLite TaskStore keeps one canonical JSON snapshot per run. Snapshot writes
 * are atomic SQLite statements and avoid coupling taskpool to a DB package.
 */
export class SqliteTaskStore implements TaskStore {
  private db?: Database;

  constructor(
    readonly dbPath: string,
    private readonly wal = true,
  ) {}

  async init(): Promise<void> {
    const moduleName = "better-sqlite3";
    const mod = (await import(moduleName).catch((error) => {
      throw new Error(`SqliteTaskStore: better-sqlite3 is required: ${(error as Error).message}`);
    })) as { default?: DatabaseCtor } & DatabaseCtor;
    const Ctor = (mod.default ?? mod) as DatabaseCtor;
    this.db = new Ctor(this.dbPath);
    if (this.wal && this.dbPath !== ":memory:") this.db.pragma("journal_mode = WAL");
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS task_runs (
        run_id TEXT PRIMARY KEY,
        plan_id TEXT NOT NULL,
        status TEXT NOT NULL,
        snapshot_json TEXT NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_task_runs_status_updated
        ON task_runs(status, updated_at);
    `);
  }

  async saveRun(run: StoredTaskRun): Promise<void> {
    const db = this.requireDb();
    db.prepare(`
      INSERT INTO task_runs
        (run_id, plan_id, status, snapshot_json, created_at, updated_at)
      VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(run_id) DO UPDATE SET
        plan_id = excluded.plan_id,
        status = excluded.status,
        snapshot_json = excluded.snapshot_json,
        updated_at = excluded.updated_at
    `).run(run.runId, run.planId, run.status, JSON.stringify(run), run.createdAt, run.updatedAt);
  }

  async loadRun(runId: string): Promise<StoredTaskRun | undefined> {
    const row = this.requireDb()
      .prepare("SELECT snapshot_json FROM task_runs WHERE run_id = ?")
      .get(runId) as { snapshot_json: string } | undefined;
    return row ? (JSON.parse(row.snapshot_json) as StoredTaskRun) : undefined;
  }

  async listIncompleteRuns(): Promise<readonly StoredTaskRun[]> {
    const rows = this.requireDb()
      .prepare(
        "SELECT snapshot_json FROM task_runs WHERE status = 'running' ORDER BY created_at, run_id",
      )
      .all() as Array<{ snapshot_json: string }>;
    return rows.map((row) => JSON.parse(row.snapshot_json) as StoredTaskRun);
  }

  async close(): Promise<void> {
    this.db?.close();
    delete this.db;
  }

  private requireDb(): Database {
    if (!this.db) throw new Error("SqliteTaskStore not initialized");
    return this.db;
  }
}
