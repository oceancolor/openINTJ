import { z } from "zod";

/** Fragments 元数据（与 LanceDB 行同 fragmentId 关联）。 */
export const FragmentMetaSchema = z.object({
  fragmentId: z.string(),
  memoryType: z.enum(["short_term", "working", "long_term"]),
  importance: z.number().min(0).max(1),
  contentHash: z.string(),
  taskTagsCsv: z.string().default(""),
  metadataJson: z.string().default("{}"),
  summariesJson: z.string().default("{}"),
  timestamp: z.number(),
  accessCount: z.number().int().nonnegative().default(0),
  lastAccessed: z.number().nonnegative().default(0),
});
export type FragmentMeta = z.infer<typeof FragmentMetaSchema>;

/** 审计事件（治理平面持久化）。 */
export const AuditRowSchema = z.object({
  eventId: z.string(),
  eventType: z.string(),
  command: z.string().nullable().default(null),
  riskLevel: z.string().nullable().default(null),
  approved: z.number().int().nullable().default(null),
  reason: z.string().nullable().default(null),
  metadataJson: z.string().default("{}"),
  timestamp: z.number(),
});
export type AuditRow = z.infer<typeof AuditRowSchema>;

/** 会话记录。 */
export const SessionRowSchema = z.object({
  sessionId: z.string(),
  startedAt: z.number(),
  lastActiveAt: z.number(),
  metadataJson: z.string().default("{}"),
});
export type SessionRow = z.infer<typeof SessionRowSchema>;

export interface MetadataStore {
  readonly name: string;
  init(): Promise<void>;
  /** 升级数据库 schema 至最新版本。 */
  migrate(): Promise<{ from: number; to: number }>;

  putFragmentMeta(rows: readonly FragmentMeta[]): Promise<void>;
  getFragmentMeta(fragmentId: string): Promise<FragmentMeta | undefined>;
  listFragmentMeta(opts?: {
    memoryType?: "short_term" | "working" | "long_term";
    limit?: number;
  }): Promise<FragmentMeta[]>;
  deleteFragmentMeta(fragmentIds: readonly string[]): Promise<number>;

  recordAudit(row: AuditRow): Promise<void>;
  queryAudit(opts?: {
    eventType?: string;
    since?: number;
    limit?: number;
  }): Promise<AuditRow[]>;
  pruneAudit(beforeTimestamp: number): Promise<number>;

  putSession(row: SessionRow): Promise<void>;
  getSession(sessionId: string): Promise<SessionRow | undefined>;
  touchSession(sessionId: string, ts: number): Promise<void>;

  close(): Promise<void>;
}
