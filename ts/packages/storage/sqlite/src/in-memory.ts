import {
  type AuditRow,
  AuditRowSchema,
  type FragmentMeta,
  FragmentMetaSchema,
  type MetadataStore,
  type SessionRow,
  SessionRowSchema,
} from "./types.js";

/**
 * 内存版元数据存储 —— 用于测试 + CI 兜底（无 better-sqlite3 依赖）。
 */
export class InMemoryMetadataStore implements MetadataStore {
  readonly name = "in-memory-metadata";
  private fragments = new Map<string, FragmentMeta>();
  private audits: AuditRow[] = [];
  private sessions = new Map<string, SessionRow>();

  async init(): Promise<void> {}

  async migrate(): Promise<{ from: number; to: number }> {
    return { from: 1, to: 1 };
  }

  async putFragmentMeta(rows: readonly FragmentMeta[]): Promise<void> {
    for (const r of rows) {
      const v = FragmentMetaSchema.parse(r);
      this.fragments.set(v.fragmentId, v);
    }
  }

  async getFragmentMeta(fragmentId: string): Promise<FragmentMeta | undefined> {
    return this.fragments.get(fragmentId);
  }

  async listFragmentMeta(
    opts: {
      memoryType?: "short_term" | "working" | "long_term";
      limit?: number;
    } = {},
  ): Promise<FragmentMeta[]> {
    let arr = [...this.fragments.values()];
    if (opts.memoryType) arr = arr.filter((r) => r.memoryType === opts.memoryType);
    arr.sort((a, b) => b.timestamp - a.timestamp);
    if (opts.limit !== undefined) arr = arr.slice(0, opts.limit);
    return arr;
  }

  async deleteFragmentMeta(fragmentIds: readonly string[]): Promise<number> {
    let n = 0;
    for (const id of fragmentIds) if (this.fragments.delete(id)) n++;
    return n;
  }

  async recordAudit(row: AuditRow): Promise<void> {
    this.audits.push(AuditRowSchema.parse(row));
  }

  async queryAudit(
    opts: { eventType?: string; since?: number; limit?: number } = {},
  ): Promise<AuditRow[]> {
    let arr = [...this.audits];
    if (opts.eventType) arr = arr.filter((r) => r.eventType === opts.eventType);
    if (opts.since !== undefined) {
      arr = arr.filter((r) => r.timestamp >= opts.since!);
    }
    arr.sort((a, b) => b.timestamp - a.timestamp);
    if (opts.limit !== undefined) arr = arr.slice(0, opts.limit);
    return arr;
  }

  async pruneAudit(beforeTimestamp: number): Promise<number> {
    const before = this.audits.length;
    this.audits = this.audits.filter((r) => r.timestamp >= beforeTimestamp);
    return before - this.audits.length;
  }

  async putSession(row: SessionRow): Promise<void> {
    const v = SessionRowSchema.parse(row);
    this.sessions.set(v.sessionId, v);
  }

  async getSession(sessionId: string): Promise<SessionRow | undefined> {
    return this.sessions.get(sessionId);
  }

  async touchSession(sessionId: string, ts: number): Promise<void> {
    const r = this.sessions.get(sessionId);
    if (r) r.lastActiveAt = ts;
  }

  async close(): Promise<void> {
    this.fragments.clear();
    this.audits.length = 0;
    this.sessions.clear();
  }
}
