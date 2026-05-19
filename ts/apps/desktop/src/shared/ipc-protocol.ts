/**
 * Shared IPC protocol types between main and renderer.
 *
 * 严格按 RFC-004 规定：
 *  - request/response by ipcMain.handle / ipcRenderer.invoke
 *  - server-push events by webContents.send / ipcRenderer.on
 *  - 所有 payload 在两端都用 zod 校验
 */

import { z } from "zod";

// ---------- Request/Response ----------

export const ChatRequestSchema = z.object({
  query: z.string().min(1),
  systemPrompt: z.string().optional(),
});
export type ChatRequest = z.infer<typeof ChatRequestSchema>;

export const ChatResponseSchema = z.object({
  finalAnswer: z.string(),
  iterations: z.number().int().nonnegative(),
  status: z.string(),
  traceId: z.string(),
});
export type ChatResponse = z.infer<typeof ChatResponseSchema>;

export const StatusResponseSchema = z.object({
  llm: z.object({
    provider: z.string(),
    status: z.string(),
    model: z.string().optional(),
  }),
  memory: z.object({
    counts: z.record(z.string(), z.number()),
    total: z.number(),
  }),
  governance: z.object({
    audit: z.object({
      totalEvents: z.number(),
      blockedCount: z.number(),
      warningCount: z.number(),
      allowedCount: z.number(),
    }),
  }),
  tools: z.array(z.string()),
  /** Phase 3.1 起：当前持久化模式 + 数据目录（real 模式时存在）。 */
  persistence: z
    .object({
      mode: z.enum(["memory", "real"]),
      dataDir: z.string().optional(),
    })
    .optional(),
  /** Phase 3.3 起：默认检索模式（vector / hybrid）。 */
  retrievalMode: z.enum(["vector", "hybrid"]).optional(),
  /** Phase 3.3/3.4 起：Dormant 子系统状态（仅启用时存在）。 */
  dormant: z
    .object({
      enabled: z.literal(true),
      passiveSize: z.number(),
      pendingProposals: z.number(),
      persistence: z
        .object({
          adapter: z.string(),
          dbPath: z.string().optional(),
        })
        .optional(),
    })
    .optional(),
});
export type StatusResponse = z.infer<typeof StatusResponseSchema>;

export const MemoryQueryRequestSchema = z.object({
  query: z.string().optional(),
  topK: z.number().int().positive().max(50).default(10),
  /** 检索模式覆盖；不传则按 agent.retrievalMode 默认。 */
  mode: z.enum(["vector", "hybrid"]).optional(),
  /** hybrid 模式下是否启用 RRF 融合。 */
  rrf: z.boolean().optional(),
});
export type MemoryQueryRequest = z.infer<typeof MemoryQueryRequestSchema>;

export const MemoryQueryResultSchema = z.object({
  fragmentId: z.string(),
  content: z.string(),
  score: z.number().optional(),
  memoryType: z.string(),
  taskTags: z.array(z.string()),
});
export type MemoryQueryResult = z.infer<typeof MemoryQueryResultSchema>;

// ---------- Dormant Memory Learning (RFC-003 方向 3) ----------

export const DormantProposalDecisionSchema = z.object({
  proposalId: z.string(),
});
export type DormantProposalDecision = z.infer<typeof DormantProposalDecisionSchema>;

export const DormantListRequestSchema = z.object({
  status: z.enum(["pending", "approved", "rejected", "applied"]).optional(),
});
export type DormantListRequest = z.infer<typeof DormantListRequestSchema>;

/** 单条 proposal 在 IPC 上的精简形式（pattern 只保留 description）。 */
export const DormantProposalDtoSchema = z.object({
  proposalId: z.string(),
  targetField: z.string(),
  value: z.unknown(),
  status: z.enum(["pending", "approved", "rejected", "applied"]),
  ts: z.number(),
  decidedAt: z.number().optional(),
  patternDescription: z.string(),
  confidence: z.number(),
  frequency: z.number(),
});
export type DormantProposalDto = z.infer<typeof DormantProposalDtoSchema>;

/** Phase 3.3 起：未启用时所有 dormant.* channel 返回该 shape。 */
export const DormantErrorSchema = z.object({
  error: z.literal("dormant_not_enabled"),
  hint: z.string().optional(),
});
export type DormantError = z.infer<typeof DormantErrorSchema>;

export const DormantPatternDtoSchema = z.object({
  patternId: z.string(),
  description: z.string(),
  category: z.enum(["preference", "phrase", "habit", "context", "other"]),
  frequency: z.number(),
  confidence: z.number(),
});
export type DormantPatternDto = z.infer<typeof DormantPatternDtoSchema>;

export const DormantMineResponseSchema = z.object({
  scannedEvents: z.number(),
  patterns: z.array(DormantPatternDtoSchema),
  proposals: z.array(
    z.object({
      proposalId: z.string(),
      targetField: z.string(),
      value: z.unknown(),
      status: z.enum(["pending", "approved", "rejected", "applied"]),
      patternDescription: z.string(),
    }),
  ),
});
export type DormantMineResponse = z.infer<typeof DormantMineResponseSchema>;

export const DormantListResponseSchema = z.object({
  total: z.number(),
  proposals: z.array(DormantProposalDtoSchema),
});
export type DormantListResponse = z.infer<typeof DormantListResponseSchema>;

export const DormantDecisionResponseSchema = z.object({
  proposalId: z.string(),
  status: z.enum(["pending", "approved", "rejected", "applied"]),
  decidedAt: z.number().optional(),
});
export type DormantDecisionResponse = z.infer<typeof DormantDecisionResponseSchema>;

export const DormantPersonaResponseSchema = z.object({
  preferences: z.record(z.string(), z.unknown()),
  phrases: z.record(z.string(), z.string()),
  habits: z.record(z.string(), z.unknown()),
  context: z.record(z.string(), z.unknown()),
  meta: z.object({ lastUpdated: z.number(), version: z.number() }),
});
export type DormantPersonaResponse = z.infer<typeof DormantPersonaResponseSchema>;

/** Phase 3.3 起：approve/reject 失败时返回的错误 shape。 */
export const DormantDecisionErrorSchema = z.object({
  error: z.union([
    z.literal("dormant_not_enabled"),
    z.literal("not_found_or_already_decided"),
    z.literal("invalid_request"),
  ]),
  issues: z.array(z.unknown()).optional(),
  hint: z.string().optional(),
});
export type DormantDecisionError = z.infer<typeof DormantDecisionErrorSchema>;

// ---------- Channel constants ----------

export const IPC = {
  PING: "openintj:ping",
  STATUS: "openintj:status",
  CHAT: "openintj:chat",
  MEMORY_QUERY: "openintj:memory.query",
  AUDIT_RECENT: "openintj:audit.recent",
  // RFC-003 方向 3：Dormant Memory Learning
  DORMANT_MINE: "openintj:dormant.mine",
  DORMANT_LIST: "openintj:dormant.list",
  DORMANT_APPROVE: "openintj:dormant.approve",
  DORMANT_REJECT: "openintj:dormant.reject",
  DORMANT_PERSONA: "openintj:dormant.persona",
  // server-push events
  EVT_TAO: "openintj:evt.tao",
  EVT_REACT: "openintj:evt.react",
  EVT_AUDIT: "openintj:evt.audit",
} as const;

export type IpcChannel = (typeof IPC)[keyof typeof IPC];
