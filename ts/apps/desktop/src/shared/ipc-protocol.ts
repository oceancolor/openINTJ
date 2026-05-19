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
