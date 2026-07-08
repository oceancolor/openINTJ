import type {
  Skill,
  SkillProposal,
  SkillStore,
  SkillStoreSnapshot,
  SkillWeight,
} from "@openintj/skills";
import { z } from "zod";

export const SqliteSkillConfigSchema = z.object({
  /** 技能库文件路径；":memory:" 走纯内存。 */
  dbPath: z.string(),
  /** WAL 模式（建议本地开启）。 */
  wal: z.boolean().default(true),
});
export type SqliteSkillConfig = z.infer<typeof SqliteSkillConfigSchema>;
export type SqliteSkillConfigInput = z.input<typeof SqliteSkillConfigSchema>;

// 载入校验：结构容错（坏行跳过），taskTypes 宽松为 string[]（TaskTypeType 是 string 联合）。
const SkillSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  triggers: z.array(z.string()).default([]),
  taskTypes: z.array(z.string()).default([]),
  priority: z.number().default(0),
  version: z.string().default("0.0.0"),
  body: z.string(),
  source: z.string(),
});

const SkillProposalSchema = z.object({
  proposalId: z.string(),
  candidate: SkillSchema,
  evidence: z.object({
    queries: z.array(z.string()).default([]),
    taskType: z.string().optional(),
    count: z.number().default(0),
  }),
  status: z.enum(["pending", "approved", "rejected", "revoked"]),
  ts: z.number(),
  decidedAt: z.number().optional(),
});

const SkillWeightSchema = z.object({
  skillId: z.string(),
  weight: z.number(),
  lastUsed: z.number(),
});

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
    CREATE TABLE IF NOT EXISTS skill_schema_version (
      version INTEGER PRIMARY KEY
    );

    CREATE TABLE IF NOT EXISTS skill_approved (
      skillId   TEXT PRIMARY KEY,
      json      TEXT NOT NULL,
      updatedAt INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS skill_proposals (
      proposalId  TEXT PRIMARY KEY,
      skillId     TEXT NOT NULL,
      json        TEXT NOT NULL,
      status      TEXT NOT NULL,
      ts          INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sprop_status ON skill_proposals(status);
    CREATE INDEX IF NOT EXISTS idx_sprop_ts ON skill_proposals(ts);

    CREATE TABLE IF NOT EXISTS skill_weights (
      skillId   TEXT PRIMARY KEY,
      weight    REAL NOT NULL,
      lastUsed  INTEGER NOT NULL
    );
  `,
};

interface ApprovedRow {
  skillId: string;
  json: string;
}
interface ProposalRow {
  proposalId: string;
  skillId: string;
  json: string;
  status: string;
  ts: number;
}
interface WeightRow {
  skillId: string;
  weight: number;
  lastUsed: number;
}

/**
 * SqliteSkillStore —— 技能自学习子系统的 SQLite 持久化（Phase 2）。
 *
 * 与 {@link import("./dormant.js").SqliteDormantStore} 同构：better-sqlite3 peer dep 动态 import、
 * WAL、版本化迁移、每库独立文件（默认 `skills.sqlite`）。热路径同步写、不抛错。
 *
 * approved/proposals 各存整对象 JSON（技能形状会演进），weights 拆列（便于聚合/调试）。
 */
export class SqliteSkillStore implements SkillStore {
  readonly name: string;
  readonly config: SqliteSkillConfig;
  private db?: BetterSqliteDB;
  private stmtUpsertApproved?: BetterSqliteStmt;
  private stmtRemoveApproved?: BetterSqliteStmt;
  private stmtUpsertProposal?: BetterSqliteStmt;
  private stmtSaveWeight?: BetterSqliteStmt;
  private stmtSelectApproved?: BetterSqliteStmt;
  private stmtSelectProposals?: BetterSqliteStmt;
  private stmtSelectWeights?: BetterSqliteStmt;

  constructor(config: SqliteSkillConfigInput) {
    this.config = SqliteSkillConfigSchema.parse(config);
    this.name = `sqlite-skills:${this.config.dbPath}`;
  }

  /** 必须在使用前 await：dynamic import → WAL → migrate → prepare。 */
  async init(): Promise<void> {
    const moduleName = "better-sqlite3";
    const mod = (await import(moduleName).catch((e) => {
      throw new Error(
        `SqliteSkillStore: failed to load better-sqlite3 (peer dep). Install: pnpm add better-sqlite3. Cause: ${(e as Error).message}`,
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

  private async migrate(): Promise<{ from: number; to: number }> {
    if (!this.db) throw new Error("SqliteSkillStore not initialized");
    this.db.exec(MIGRATIONS[1] ?? "");
    const row = this.db.prepare("SELECT version FROM skill_schema_version").get() as
      | { version: number }
      | undefined;
    const currentVersion = row?.version ?? 0;
    if (currentVersion === TARGET_VERSION) return { from: currentVersion, to: currentVersion };
    for (let v = currentVersion + 1; v <= TARGET_VERSION; v++) {
      const sql = MIGRATIONS[v];
      if (sql) this.db.exec(sql);
    }
    this.db.exec("DELETE FROM skill_schema_version");
    this.db.prepare("INSERT INTO skill_schema_version (version) VALUES (?)").run(TARGET_VERSION);
    return { from: currentVersion, to: TARGET_VERSION };
  }

  private prepareStatements(): void {
    if (!this.db) throw new Error("SqliteSkillStore not initialized");
    this.stmtUpsertApproved = this.db.prepare(
      `INSERT INTO skill_approved (skillId, json, updatedAt) VALUES (?, ?, ?)
       ON CONFLICT(skillId) DO UPDATE SET json = excluded.json, updatedAt = excluded.updatedAt`,
    );
    this.stmtRemoveApproved = this.db.prepare("DELETE FROM skill_approved WHERE skillId = ?");
    this.stmtUpsertProposal = this.db.prepare(
      `INSERT INTO skill_proposals (proposalId, skillId, json, status, ts) VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(proposalId) DO UPDATE SET
         skillId = excluded.skillId,
         json = excluded.json,
         status = excluded.status,
         ts = excluded.ts`,
    );
    this.stmtSaveWeight = this.db.prepare(
      `INSERT INTO skill_weights (skillId, weight, lastUsed) VALUES (?, ?, ?)
       ON CONFLICT(skillId) DO UPDATE SET weight = excluded.weight, lastUsed = excluded.lastUsed`,
    );
    this.stmtSelectApproved = this.db.prepare("SELECT skillId, json FROM skill_approved");
    this.stmtSelectProposals = this.db.prepare(
      "SELECT proposalId, skillId, json, status, ts FROM skill_proposals ORDER BY ts ASC",
    );
    this.stmtSelectWeights = this.db.prepare("SELECT skillId, weight, lastUsed FROM skill_weights");
  }

  async loadAll(): Promise<SkillStoreSnapshot> {
    if (!this.stmtSelectApproved || !this.stmtSelectProposals || !this.stmtSelectWeights) {
      throw new Error("SqliteSkillStore not initialized");
    }
    const approvedSkills: Skill[] = [];
    for (const r of this.stmtSelectApproved.all() as ApprovedRow[]) {
      const parsed = SkillSchema.safeParse(safeJsonParse(r.json, {}));
      if (parsed.success) approvedSkills.push(parsed.data as Skill);
    }
    const proposals: SkillProposal[] = [];
    for (const r of this.stmtSelectProposals.all() as ProposalRow[]) {
      const parsed = SkillProposalSchema.safeParse(safeJsonParse(r.json, {}));
      if (parsed.success) proposals.push(parsed.data as SkillProposal);
    }
    const weights: SkillWeight[] = [];
    for (const r of this.stmtSelectWeights.all() as WeightRow[]) {
      const parsed = SkillWeightSchema.safeParse(r);
      if (parsed.success) weights.push(parsed.data);
    }
    return { approvedSkills, proposals, weights };
  }

  upsertApprovedSkill(skill: Skill): void {
    if (!this.stmtUpsertApproved) return;
    try {
      this.stmtUpsertApproved.run(skill.id, JSON.stringify(skill), Date.now());
    } catch (e) {
      console.error("[SqliteSkillStore] upsertApprovedSkill failed:", (e as Error).message);
    }
  }

  removeApprovedSkill(skillId: string): void {
    if (!this.stmtRemoveApproved) return;
    try {
      this.stmtRemoveApproved.run(skillId);
    } catch (e) {
      console.error("[SqliteSkillStore] removeApprovedSkill failed:", (e as Error).message);
    }
  }

  upsertProposal(proposal: SkillProposal): void {
    if (!this.stmtUpsertProposal) return;
    try {
      this.stmtUpsertProposal.run(
        proposal.proposalId,
        proposal.candidate.id,
        JSON.stringify(proposal),
        proposal.status,
        proposal.ts,
      );
    } catch (e) {
      console.error("[SqliteSkillStore] upsertProposal failed:", (e as Error).message);
    }
  }

  saveWeight(weight: SkillWeight): void {
    if (!this.stmtSaveWeight) return;
    try {
      this.stmtSaveWeight.run(weight.skillId, weight.weight, weight.lastUsed);
    } catch (e) {
      console.error("[SqliteSkillStore] saveWeight failed:", (e as Error).message);
    }
  }

  clearAll(): void {
    if (!this.db) return;
    try {
      this.db.exec(
        "DELETE FROM skill_approved; DELETE FROM skill_proposals; DELETE FROM skill_weights;",
      );
    } catch (e) {
      console.error("[SqliteSkillStore] clearAll failed:", (e as Error).message);
    }
  }

  async close(): Promise<void> {
    this.db?.close();
    delete this.db;
    delete this.stmtUpsertApproved;
    delete this.stmtRemoveApproved;
    delete this.stmtUpsertProposal;
    delete this.stmtSaveWeight;
    delete this.stmtSelectApproved;
    delete this.stmtSelectProposals;
    delete this.stmtSelectWeights;
  }
}

const safeJsonParse = <T>(raw: string, fallback: T): T => {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

/** 工厂方法 —— 一次性 new + init，方便装配点直接 await。 */
export const createSqliteSkillStore = async (
  config: SqliteSkillConfigInput,
): Promise<SqliteSkillStore> => {
  const s = new SqliteSkillStore(config);
  await s.init();
  return s;
};
