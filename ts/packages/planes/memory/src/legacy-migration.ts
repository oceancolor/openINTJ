import { contentHash } from "@openintj/core";
import type { VectorStore } from "@openintj/storage-lance";
import type { MetadataStore } from "@openintj/storage-sqlite";
import type { CanonicalMemoryRecord, CanonicalMemoryRepository } from "./canonical.js";

/**
 * G2 迁移：legacy（Lance 正文 + SQLite meta）→ canonical 对账。
 *
 * 铁律：dry-run 是默认且只读；不一致、无正文、scope 不明的数据一律进 quarantine
 * 报告，绝不静默激活；不向旧库反向双写。真实数据迁移需显式 dryRun:false 调用。
 */
export interface LegacyAuditReport {
  vectorRows: number;
  metaRows: number;
  matched: number;
  missingMeta: string[];
  missingVector: string[];
  emptyContent: string[];
  scopeUnknown: string[];
}

const SCOPE_TAG_RE = /^(workspace|task|conversation|principal):/;

export const auditLegacyMemory = async (deps: {
  vectorStore: VectorStore;
  metadataStore: MetadataStore;
}): Promise<LegacyAuditReport> => {
  const rows = await deps.vectorStore.scanAll();
  const metas = await deps.metadataStore.listFragmentMeta({});
  const metaIds = new Set(metas.map((m) => m.fragmentId));
  const report: LegacyAuditReport = {
    vectorRows: rows.length,
    metaRows: metas.length,
    matched: 0,
    missingMeta: [],
    missingVector: [],
    emptyContent: [],
    scopeUnknown: [],
  };
  const seen = new Set<string>();
  for (const r of rows) {
    seen.add(r.fragmentId);
    if (metaIds.has(r.fragmentId)) report.matched++;
    else report.missingMeta.push(r.fragmentId);
    if (!r.content?.trim()) report.emptyContent.push(r.fragmentId);
    if (!r.taskTags.some((t) => SCOPE_TAG_RE.test(t))) report.scopeUnknown.push(r.fragmentId);
  }
  for (const m of metas) {
    if (!seen.has(m.fragmentId)) report.missingVector.push(m.fragmentId);
  }
  return report;
};

export interface LegacyMigrationResult extends LegacyAuditReport {
  dryRun: boolean;
  migratedActive: number;
  migratedQuarantined: number;
}

/**
 * 执行迁移（dryRun 默认 true 只出报告）。scope 不明或无正文的记为 quarantined，
 * 留在 canonical 里可审计但不参与召回，等人工确认归属后再激活。
 */
export const migrateLegacyMemory = async (
  deps: {
    vectorStore: VectorStore;
    metadataStore: MetadataStore;
    canonical: CanonicalMemoryRepository;
  },
  opts: { dryRun?: boolean } = {},
): Promise<LegacyMigrationResult> => {
  const dryRun = opts.dryRun ?? true;
  const report = await auditLegacyMemory(deps);
  let migratedActive = 0;
  let migratedQuarantined = 0;
  if (!dryRun) {
    const rows = await deps.vectorStore.scanAll();
    const quarantined = new Set([...report.scopeUnknown, ...report.emptyContent]);
    for (const r of rows) {
      const state = quarantined.has(r.fragmentId) ? "quarantined" : "active";
      const record: CanonicalMemoryRecord = {
        fragmentId: r.fragmentId,
        content: r.content,
        memoryType: r.memoryType,
        importance: r.importance,
        taskTags: [...r.taskTags],
        scopeTags: r.taskTags.filter((t) => SCOPE_TAG_RE.test(t)),
        metadata: safeParse(r.metadataJson),
        summaries: safeSummaries(r.summariesJson),
        embedding: [...r.embedding],
        source: "migration:legacy",
        state,
        revision: 0,
        timestamp: r.timestamp,
        accessCount: r.accessCount,
        lastAccessed: r.lastAccessed,
        kind: "note",
        assertion: "assistant",
      };
      await deps.canonical.applyMutation({
        writeKey: `migration:${r.fragmentId}`,
        inputHash: contentHash({ content: r.content }),
        kind: "upsert",
        record,
        initialState: state,
      });
      if (state === "active") migratedActive++;
      else migratedQuarantined++;
    }
  }
  return { ...report, dryRun, migratedActive, migratedQuarantined };
};

const safeParse = (json: string): Record<string, unknown> => {
  try {
    return JSON.parse(json || "{}") as Record<string, unknown>;
  } catch {
    return {};
  }
};

const safeSummaries = (json: string): Record<number, string> => {
  try {
    const raw = JSON.parse(json || "{}") as Record<string, string>;
    return Object.fromEntries(Object.entries(raw).map(([k, v]) => [Number(k), v]));
  } catch {
    return {};
  }
};
