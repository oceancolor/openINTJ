/**
 * SqliteCanonicalMemoryStore —— G2 canonical memory 权威源（better-sqlite3）。
 *
 * 与 plane-memory 的 CanonicalMemoryRepository 契约结构兼容（storage 层不反向依赖 plane）。
 * 单事务边界：记录变更 + mutation 收据 + index outbox 一起提交或一起回滚。
 */

export type CanonicalRecordState = "active" | "tombstoned" | "quarantined" | "superseded";
export type CanonicalMemoryType = "short_term" | "working" | "long_term";

export interface CanonicalMemoryRecordRow {
  fragmentId: string;
  content: string;
  memoryType: CanonicalMemoryType;
  importance: number;
  taskTags: string[];
  scopeTags: string[];
  metadata: Record<string, unknown>;
  summaries: Record<number, string>;
  embedding: number[];
  source: string;
  state: CanonicalRecordState;
  revision: number;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
  kind: string;
  assertion: string;
  validUntil?: number;
  canonicalKey?: string;
  supersededBy?: string;
}

export interface CanonicalMutationRequest {
  writeKey: string;
  inputHash: string;
  kind: "upsert" | "delete" | "clear_tier" | "reassign" | "mark_superseded";
  fragmentId?: string;
  record?: CanonicalMemoryRecordRow;
  tier?: CanonicalMemoryType;
  newType?: CanonicalMemoryType;
  expectedRevision?: number;
  initialState?: CanonicalRecordState;
  supersededBy?: string;
}

export interface CanonicalMutationReceipt {
  writeKey: string;
  outcome: "applied" | "replayed";
  fragmentId?: string;
  revision?: number;
  appliedAt: number;
}

export interface CanonicalIndexJob {
  jobId: number;
  fragmentId: string;
  kind: "upsert" | "delete";
  revision: number;
  generation: number;
  attempts: number;
  lastError?: string;
  createdAt: number;
}

export class CanonicalMutationConflictError extends Error {
  readonly code = "MEMORY_MUTATION_CONFLICT";
}

export class CanonicalCandidateConflictError extends Error {
  readonly code = "MEMORY_CANDIDATE_CONFLICT";
}

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

const SCHEMA = `
CREATE TABLE IF NOT EXISTS memory_records (
  fragmentId TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  memoryType TEXT NOT NULL,
  importance REAL NOT NULL,
  taskTagsJson TEXT NOT NULL DEFAULT '[]',
  scopeTagsJson TEXT NOT NULL DEFAULT '[]',
  metadataJson TEXT NOT NULL DEFAULT '{}',
  summariesJson TEXT NOT NULL DEFAULT '{}',
  embeddingJson TEXT NOT NULL DEFAULT '[]',
  source TEXT NOT NULL DEFAULT 'runtime',
  state TEXT NOT NULL DEFAULT 'active',
  revision INTEGER NOT NULL DEFAULT 1,
  timestamp REAL NOT NULL,
  accessCount INTEGER NOT NULL DEFAULT 0,
  lastAccessed REAL NOT NULL DEFAULT 0,
  kind TEXT NOT NULL DEFAULT 'note',
  assertion TEXT NOT NULL DEFAULT 'assistant',
  validUntil REAL,
  canonicalKey TEXT,
  supersededBy TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_state ON memory_records(state);
CREATE INDEX IF NOT EXISTS idx_mem_type ON memory_records(memoryType);

CREATE TABLE IF NOT EXISTS memory_mutations (
  writeKey TEXT PRIMARY KEY,
  inputHash TEXT NOT NULL,
  kind TEXT NOT NULL,
  fragmentId TEXT,
  resultRevision INTEGER,
  appliedAt REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_candidates (
  receiveKey TEXT PRIMARY KEY,
  inputHash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'received',
  candidateCount INTEGER NOT NULL DEFAULT 0,
  payloadJson TEXT NOT NULL DEFAULT '[]',
  receivedAt REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_index_outbox (
  jobId INTEGER PRIMARY KEY AUTOINCREMENT,
  fragmentId TEXT NOT NULL,
  kind TEXT NOT NULL,
  revision INTEGER NOT NULL,
  generation INTEGER NOT NULL DEFAULT 1,
  attempts INTEGER NOT NULL DEFAULT 0,
  lastError TEXT,
  createdAt REAL NOT NULL,
  appliedAt REAL
);
CREATE INDEX IF NOT EXISTS idx_outbox_pending ON memory_index_outbox(appliedAt, jobId);

CREATE TABLE IF NOT EXISTS memory_index_generations (
  generation INTEGER PRIMARY KEY,
  spaceFingerprint TEXT NOT NULL,
  createdAt REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_control (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  epoch INTEGER NOT NULL DEFAULT 1,
  updatedAt REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_suppressions (
  fragmentId TEXT PRIMARY KEY,
  principalId TEXT NOT NULL,
  reason TEXT NOT NULL,
  createdAt REAL NOT NULL
);
`;

interface RecordRowDb {
  fragmentId: string;
  content: string;
  memoryType: string;
  importance: number;
  taskTagsJson: string;
  scopeTagsJson: string;
  metadataJson: string;
  summariesJson: string;
  embeddingJson: string;
  source: string;
  state: string;
  revision: number;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
  kind: string | null;
  assertion: string | null;
  validUntil: number | null;
  canonicalKey: string | null;
  supersededBy: string | null;
}

const rowToRecord = (r: RecordRowDb): CanonicalMemoryRecordRow => ({
  fragmentId: r.fragmentId,
  content: r.content,
  memoryType: r.memoryType as CanonicalMemoryType,
  importance: r.importance,
  taskTags: JSON.parse(r.taskTagsJson || "[]") as string[],
  scopeTags: JSON.parse(r.scopeTagsJson || "[]") as string[],
  metadata: JSON.parse(r.metadataJson || "{}") as Record<string, unknown>,
  summaries: Object.fromEntries(
    Object.entries(JSON.parse(r.summariesJson || "{}") as Record<string, string>).map(([k, v]) => [
      Number(k),
      v,
    ]),
  ),
  embedding: JSON.parse(r.embeddingJson || "[]") as number[],
  source: r.source,
  state: r.state as CanonicalRecordState,
  revision: r.revision,
  timestamp: r.timestamp,
  accessCount: r.accessCount,
  lastAccessed: r.lastAccessed,
  kind: r.kind ?? "note",
  assertion: r.assertion ?? "assistant",
  ...(r.validUntil !== null && r.validUntil !== undefined ? { validUntil: r.validUntil } : {}),
  ...(r.canonicalKey ? { canonicalKey: r.canonicalKey } : {}),
  ...(r.supersededBy ? { supersededBy: r.supersededBy } : {}),
});

export class SqliteCanonicalMemoryStore {
  readonly name: string;
  private db?: BetterSqliteDB;

  constructor(private readonly dbPath: string) {
    this.name = `sqlite-canonical:${dbPath}`;
  }

  async init(): Promise<void> {
    if (this.db) return;
    const moduleName = "better-sqlite3";
    const mod = (await import(moduleName).catch((e) => {
      throw new Error(
        `SqliteCanonicalMemoryStore: failed to load better-sqlite3 (peer dep). Cause: ${(e as Error).message}`,
      );
    })) as { default?: BetterSqliteCtor } & BetterSqliteCtor;
    const Ctor = (mod.default ?? mod) as BetterSqliteCtor;
    this.db = new Ctor(this.dbPath);
    if (this.dbPath !== ":memory:") this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this.db.exec(SCHEMA);
    // F08 增量列迁移：G2 已建的库没有 lifecycle 列，逐列补齐（幂等）。
    const existing = new Set(
      (this.db.prepare("PRAGMA table_info(memory_records)").all() as Array<{ name: string }>).map(
        (c) => c.name,
      ),
    );
    const addColumns: Array<[string, string]> = [
      ["kind", "TEXT NOT NULL DEFAULT 'note'"],
      ["assertion", "TEXT NOT NULL DEFAULT 'assistant'"],
      ["validUntil", "REAL"],
      ["canonicalKey", "TEXT"],
      ["supersededBy", "TEXT"],
    ];
    for (const [name, def] of addColumns) {
      if (!existing.has(name)) this.db.exec(`ALTER TABLE memory_records ADD COLUMN ${name} ${def}`);
    }
    this.db
      .prepare("INSERT OR IGNORE INTO memory_control (id, epoch, updatedAt) VALUES (1, 1, ?)")
      .run(Date.now());
  }

  async close(): Promise<void> {
    this.db?.close();
    delete this.db;
  }

  private requireDb(): BetterSqliteDB {
    if (!this.db) throw new Error("SqliteCanonicalMemoryStore not initialized");
    return this.db;
  }

  async listRecords(opts?: { includeInactive?: boolean }): Promise<CanonicalMemoryRecordRow[]> {
    const db = this.requireDb();
    const rows = (
      opts?.includeInactive
        ? db.prepare("SELECT * FROM memory_records").all()
        : db.prepare("SELECT * FROM memory_records WHERE state = 'active'").all()
    ) as RecordRowDb[];
    return rows.map(rowToRecord);
  }

  async getRecord(fragmentId: string): Promise<CanonicalMemoryRecordRow | undefined> {
    const db = this.requireDb();
    const row = db.prepare("SELECT * FROM memory_records WHERE fragmentId = ?").get(fragmentId) as
      | RecordRowDb
      | undefined;
    return row ? rowToRecord(row) : undefined;
  }

  async applyMutation(req: CanonicalMutationRequest): Promise<CanonicalMutationReceipt> {
    const db = this.requireDb();
    const tx = db.transaction((input: unknown) => {
      const r = input as CanonicalMutationRequest;
      const prior = db.prepare("SELECT * FROM memory_mutations WHERE writeKey = ?").get(r.writeKey) as
        | { inputHash: string; fragmentId: string | null; resultRevision: number | null; appliedAt: number }
        | undefined;
      if (prior) {
        if (prior.inputHash !== r.inputHash) {
          throw new CanonicalMutationConflictError(
            `writeKey ${r.writeKey} already used with different input`,
          );
        }
        return {
          writeKey: r.writeKey,
          outcome: "replayed" as const,
          ...(prior.fragmentId ? { fragmentId: prior.fragmentId } : {}),
          ...(prior.resultRevision !== null ? { revision: prior.resultRevision } : {}),
          appliedAt: prior.appliedAt,
        };
      }
      const now = Date.now();
      const generation = this.currentGeneration(db);
      const enqueue = (fragmentId: string, kind: "upsert" | "delete", revision: number) =>
        db
          .prepare(
            "INSERT INTO memory_index_outbox (fragmentId, kind, revision, generation, createdAt) VALUES (?, ?, ?, ?, ?)",
          )
          .run(fragmentId, kind, revision, generation, now);
      let fragmentId = r.fragmentId ?? r.record?.fragmentId;
      let revision: number | undefined;
      if (r.kind === "upsert") {
        if (!r.record) throw new Error("upsert requires record");
        const existing = db
          .prepare("SELECT revision, state FROM memory_records WHERE fragmentId = ?")
          .get(r.record.fragmentId) as { revision: number; state: string } | undefined;
        if (r.expectedRevision !== undefined && (existing?.revision ?? 0) !== r.expectedRevision) {
          throw new CanonicalMutationConflictError(
            `CAS failed for ${r.record.fragmentId}: expected ${r.expectedRevision}, actual ${existing?.revision ?? 0}`,
          );
        }
        revision = (existing?.revision ?? 0) + 1;
        const state = existing?.state ?? r.initialState ?? "active";
        db.prepare(
          `INSERT INTO memory_records (fragmentId, content, memoryType, importance, taskTagsJson, scopeTagsJson, metadataJson, summariesJson, embeddingJson, source, state, revision, timestamp, accessCount, lastAccessed, kind, assertion, validUntil, canonicalKey, supersededBy)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(fragmentId) DO UPDATE SET content=excluded.content, memoryType=excluded.memoryType,
             importance=excluded.importance, taskTagsJson=excluded.taskTagsJson, scopeTagsJson=excluded.scopeTagsJson,
             metadataJson=excluded.metadataJson, summariesJson=excluded.summariesJson, embeddingJson=excluded.embeddingJson,
             source=excluded.source, revision=excluded.revision, timestamp=excluded.timestamp,
             accessCount=excluded.accessCount, lastAccessed=excluded.lastAccessed, kind=excluded.kind,
             assertion=excluded.assertion, validUntil=excluded.validUntil, canonicalKey=excluded.canonicalKey,
             supersededBy=excluded.supersededBy`,
        ).run(
          r.record.fragmentId,
          r.record.content,
          r.record.memoryType,
          r.record.importance,
          JSON.stringify(r.record.taskTags),
          JSON.stringify(r.record.scopeTags),
          JSON.stringify(r.record.metadata),
          JSON.stringify(r.record.summaries),
          JSON.stringify(r.record.embedding),
          r.record.source,
          state,
          revision,
          r.record.timestamp,
          r.record.accessCount,
          r.record.lastAccessed,
          r.record.kind ?? "note",
          r.record.assertion ?? "assistant",
          r.record.validUntil ?? null,
          r.record.canonicalKey ?? null,
          r.record.supersededBy ?? null,
        );
        enqueue(r.record.fragmentId, "upsert", revision);
        fragmentId = r.record.fragmentId;
      } else if (r.kind === "delete") {
        if (!fragmentId) throw new Error("delete requires fragmentId");
        const existing = db
          .prepare("SELECT revision FROM memory_records WHERE fragmentId = ?")
          .get(fragmentId) as { revision: number } | undefined;
        if (existing) {
          revision = existing.revision + 1;
          db.prepare("UPDATE memory_records SET state='tombstoned', revision=? WHERE fragmentId=?").run(
            revision,
            fragmentId,
          );
          enqueue(fragmentId, "delete", revision);
        }
      } else if (r.kind === "clear_tier") {
        if (!r.tier) throw new Error("clear_tier requires tier");
        const rows = db
          .prepare("SELECT fragmentId, revision FROM memory_records WHERE memoryType=? AND state='active'")
          .all(r.tier) as Array<{ fragmentId: string; revision: number }>;
        for (const row of rows) {
          const rev = row.revision + 1;
          db.prepare("UPDATE memory_records SET state='tombstoned', revision=? WHERE fragmentId=?").run(
            rev,
            row.fragmentId,
          );
          enqueue(row.fragmentId, "delete", rev);
        }
      } else if (r.kind === "reassign") {
        if (!fragmentId || !r.newType) throw new Error("reassign requires fragmentId+newType");
        const existing = db
          .prepare("SELECT revision FROM memory_records WHERE fragmentId = ?")
          .get(fragmentId) as { revision: number } | undefined;
        if (existing) {
          revision = existing.revision + 1;
          db.prepare("UPDATE memory_records SET memoryType=?, revision=? WHERE fragmentId=?").run(
            r.newType,
            revision,
            fragmentId,
          );
          enqueue(fragmentId, "upsert", revision);
        }
      } else if (r.kind === "mark_superseded") {
        if (!fragmentId || !r.supersededBy) throw new Error("mark_superseded requires fragmentId+supersededBy");
        const existing = db
          .prepare("SELECT revision FROM memory_records WHERE fragmentId = ?")
          .get(fragmentId) as { revision: number } | undefined;
        if (existing) {
          revision = existing.revision + 1;
          db.prepare(
            "UPDATE memory_records SET state='superseded', supersededBy=?, revision=? WHERE fragmentId=?",
          ).run(r.supersededBy, revision, fragmentId);
          enqueue(fragmentId, "delete", revision);
        }
      }
      db.prepare(
        "INSERT INTO memory_mutations (writeKey, inputHash, kind, fragmentId, resultRevision, appliedAt) VALUES (?, ?, ?, ?, ?, ?)",
      ).run(r.writeKey, r.inputHash, r.kind, fragmentId ?? null, revision ?? null, now);
      return {
        writeKey: r.writeKey,
        outcome: "applied" as const,
        ...(fragmentId ? { fragmentId } : {}),
        ...(revision !== undefined ? { revision } : {}),
        appliedAt: now,
      };
    });
    return tx(req);
  }

  async receiveCandidates(input: {
    key: { sourceJournalId: string; sourceCommitId: string; consumer: string; policyVersion: string; extractionRevision: number };
    inputHash: string;
    candidates: readonly Record<string, unknown>[];
  }): Promise<{ receiveKey: string; outcome: "applied" | "replayed"; candidateCount: number; receivedAt: number }> {
    const db = this.requireDb();
    const receiveKey = [
      input.key.sourceJournalId,
      input.key.sourceCommitId,
      input.key.consumer,
      input.key.policyVersion,
      String(input.key.extractionRevision),
    ].join(":");
    const prior = db.prepare("SELECT * FROM memory_candidates WHERE receiveKey = ?").get(receiveKey) as
      | { inputHash: string; candidateCount: number; receivedAt: number }
      | undefined;
    if (prior) {
      if (prior.inputHash !== input.inputHash) {
        throw new CanonicalCandidateConflictError(
          `candidate batch ${receiveKey} redelivered with different input`,
        );
      }
      return { receiveKey, outcome: "replayed", candidateCount: prior.candidateCount, receivedAt: prior.receivedAt };
    }
    const now = Date.now();
    db.prepare(
      "INSERT INTO memory_candidates (receiveKey, inputHash, status, candidateCount, payloadJson, receivedAt) VALUES (?, ?, 'received', ?, ?, ?)",
    ).run(receiveKey, input.inputHash, input.candidates.length, JSON.stringify(input.candidates), now);
    return { receiveKey, outcome: "applied", candidateCount: input.candidates.length, receivedAt: now };
  }

  async candidateBatchStatus(
    key: { sourceJournalId: string; sourceCommitId: string; consumer: string; policyVersion: string; extractionRevision: number },
  ): Promise<{ status: string; candidateCount: number } | undefined> {
    const db = this.requireDb();
    const receiveKey = [key.sourceJournalId, key.sourceCommitId, key.consumer, key.policyVersion, String(key.extractionRevision)].join(":");
    const row = db.prepare("SELECT status, candidateCount FROM memory_candidates WHERE receiveKey = ?").get(receiveKey) as
      | { status: string; candidateCount: number }
      | undefined;
    return row;
  }

  async pendingIndexJobs(limit = 100): Promise<CanonicalIndexJob[]> {
    const db = this.requireDb();
    const rows = db
      .prepare("SELECT * FROM memory_index_outbox WHERE appliedAt IS NULL ORDER BY jobId ASC LIMIT ?")
      .all(limit) as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      jobId: Number(r.jobId),
      fragmentId: String(r.fragmentId),
      kind: r.kind as "upsert" | "delete",
      revision: Number(r.revision),
      generation: Number(r.generation),
      attempts: Number(r.attempts),
      ...(r.lastError ? { lastError: String(r.lastError) } : {}),
      createdAt: Number(r.createdAt),
    }));
  }

  async markIndexJobApplied(jobId: number): Promise<void> {
    this.requireDb()
      .prepare("UPDATE memory_index_outbox SET appliedAt = ? WHERE jobId = ?")
      .run(Date.now(), jobId);
  }

  async markIndexJobFailed(jobId: number, error: string): Promise<void> {
    this.requireDb()
      .prepare("UPDATE memory_index_outbox SET attempts = attempts + 1, lastError = ? WHERE jobId = ?")
      .run(error.slice(0, 500), jobId);
  }

  private currentGeneration(db: BetterSqliteDB): number {
    const row = db.prepare("SELECT MAX(generation) AS g FROM memory_index_generations").get() as
      | { g: number | null }
      | undefined;
    return row?.g ?? 1;
  }

  async indexStats(): Promise<{ pending: number; failed: number; generation: number }> {
    const db = this.requireDb();
    const pending = db
      .prepare("SELECT COUNT(*) AS c FROM memory_index_outbox WHERE appliedAt IS NULL")
      .get() as { c: number };
    const failed = db
      .prepare("SELECT COUNT(*) AS c FROM memory_index_outbox WHERE appliedAt IS NULL AND attempts > 0")
      .get() as { c: number };
    return { pending: pending.c, failed: failed.c, generation: this.currentGeneration(db) };
  }

  async startIndexGeneration(spaceFingerprint: string): Promise<number> {
    const db = this.requireDb();
    const next = this.currentGeneration(db) + 1;
    db.prepare("INSERT INTO memory_index_generations (generation, spaceFingerprint, createdAt) VALUES (?, ?, ?)").run(
      next,
      spaceFingerprint,
      Date.now(),
    );
    return next;
  }

  async indexGenerationFingerprint(): Promise<string | undefined> {
    const db = this.requireDb();
    const row = db
      .prepare("SELECT spaceFingerprint FROM memory_index_generations ORDER BY generation DESC LIMIT 1")
      .get() as { spaceFingerprint: string } | undefined;
    return row?.spaceFingerprint;
  }

  async memoryEpoch(): Promise<number> {
    const row = this.requireDb().prepare("SELECT epoch FROM memory_control WHERE id = 1").get() as
      | { epoch: number }
      | undefined;
    return row?.epoch ?? 1;
  }

  async bumpMemoryEpoch(): Promise<number> {
    const db = this.requireDb();
    db.prepare("UPDATE memory_control SET epoch = epoch + 1, updatedAt = ? WHERE id = 1").run(Date.now());
    return this.memoryEpoch();
  }

  async addSuppression(fragmentId: string, principalId: string, reason: string): Promise<void> {
    this.requireDb()
      .prepare(
        "INSERT INTO memory_suppressions (fragmentId, principalId, reason, createdAt) VALUES (?, ?, ?, ?) ON CONFLICT(fragmentId) DO UPDATE SET principalId=excluded.principalId, reason=excluded.reason",
      )
      .run(fragmentId, principalId, reason, Date.now());
  }

  async isSuppressed(fragmentId: string): Promise<boolean> {
    const row = this.requireDb()
      .prepare("SELECT fragmentId FROM memory_suppressions WHERE fragmentId = ?")
      .get(fragmentId);
    return row !== undefined;
  }

  async listSuppressions(): Promise<Array<{ fragmentId: string; principalId: string; reason: string }>> {
    return this.requireDb().prepare("SELECT fragmentId, principalId, reason FROM memory_suppressions").all() as Array<{
      fragmentId: string;
      principalId: string;
      reason: string;
    }>;
  }
}
