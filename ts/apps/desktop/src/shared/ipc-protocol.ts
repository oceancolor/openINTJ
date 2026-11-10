/**
 * Shared IPC protocol types between main and renderer.
 *
 * ????RFC-004 ???? *  - request/response by ipcMain.handle / ipcRenderer.invoke
 *  - server-push events by webContents.send / ipcRenderer.on
 *  - ???payload ??????zod ??
 */

import { z } from "zod";

// ---------- Request/Response ----------

export const ChatRequestSchema = z.object({
  query: z.string().min(1),
  systemPrompt: z.string().optional(),
  conversationId: z.string().min(1).optional(),
  modelProfileId: z.string().min(1).optional(),
});
export type ChatRequest = z.infer<typeof ChatRequestSchema>;

export const isChatClearCommand = (text: string): boolean =>
  /^\/clear(?:\s+history)?$/i.test(text.trim());

export const StructuredTaskSchema = z.object({
  goal: z.string(),
  context: z.array(z.string()),
  relations: z.array(z.string()),
  constraints: z.array(z.string()),
  deliverables: z.array(z.string()),
  dependencies: z.array(z.string()),
  assumptions: z.array(z.string()),
});

export const InputStructureSchema = z.object({
  action: z.enum(["proceed", "clarify"]),
  mode: z.enum(["pass-through", "structured", "fallback", "clarification"]),
  executionInput: z.string(),
  structure: StructuredTaskSchema,
  ambiguityScore: z.number().min(0).max(1),
  questions: z.array(z.string()).max(3),
  tokensSpent: z.number().int().nonnegative(),
  durationMs: z.number().nonnegative(),
  reason: z.string().optional(),
});
export type InputStructure = z.infer<typeof InputStructureSchema>;

export const ContextUsageSchema = z.object({
  used: z.number().int().nonnegative(),
  max: z.number().int().positive(),
  ratio: z.number().min(0).max(1),
});
export type ContextUsage = z.infer<typeof ContextUsageSchema>;

export const WorkspaceArtifactSchema = z.object({
  path: z.string().min(1),
  kind: z.enum(["image", "text", "file"]),
});
export type WorkspaceArtifact = z.infer<typeof WorkspaceArtifactSchema>;

export const SlashCommandResultSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("ok"), message: z.string().optional() }),
  z.object({ kind: z.literal("error"), message: z.string() }),
]);
export type SlashCommandResultView = z.infer<typeof SlashCommandResultSchema>;

export const CommandsInvokeRequestSchema = z.object({
  text: z.string().min(1),
  conversationId: z.string().min(1).optional(),
});
export type CommandsInvokeRequest = z.infer<typeof CommandsInvokeRequestSchema>;

export const CommandsInvokeResponseSchema = z.object({
  name: z.string(),
  result: SlashCommandResultSchema,
});
export type CommandsInvokeResponse = z.infer<typeof CommandsInvokeResponseSchema>;

export const AgentSteerRequestSchema = z.object({
  conversationId: z.string().min(1),
  message: z.string().min(1),
});
export type AgentSteerRequest = z.infer<typeof AgentSteerRequestSchema>;

export const AgentCancelRequestSchema = z.object({
  conversationId: z.string().min(1),
  cause: z.enum(["user", "command", "dispose"]).default("user"),
});
export type AgentCancelRequest = z.infer<typeof AgentCancelRequestSchema>;

// RFC-013 interactive approval + capability grants
export const ToolRiskSchema = z.enum(["read", "write", "execute", "external"]);
export type ToolRiskView = z.infer<typeof ToolRiskSchema>;

export const ApprovalRequestViewSchema = z.object({
  id: z.string(),
  tool: z.string(),
  risk: ToolRiskSchema,
  params: z.record(z.string(), z.unknown()),
  requestedAt: z.number(),
});
export type ApprovalRequestView = z.infer<typeof ApprovalRequestViewSchema>;

export const ApprovalDecideRequestSchema = z.object({
  id: z.string().min(1),
  decision: z.enum(["approved", "denied"]),
  reason: z.string().optional(),
  grant: z
    .object({ scope: z.string().optional(), ttlMs: z.number().positive().optional() })
    .optional(),
});
export type ApprovalDecideRequest = z.infer<typeof ApprovalDecideRequestSchema>;

export const CapabilityGrantViewSchema = z.object({
  id: z.string(),
  tool: z.string(),
  risk: ToolRiskSchema,
  scope: z.string().optional(),
  expiresAt: z.number().optional(),
  createdAt: z.number(),
  label: z.string().optional(),
});
export type CapabilityGrantView = z.infer<typeof CapabilityGrantViewSchema>;

export const GrantCreateRequestSchema = z.object({
  tool: z.string().min(1),
  risk: ToolRiskSchema,
  scope: z.string().optional(),
  ttlMs: z.number().positive().optional(),
  label: z.string().optional(),
});
export type GrantCreateRequest = z.infer<typeof GrantCreateRequestSchema>;

// RFC-014 personal profile (memory control plane)
export const ProfileSectionSchema = z.enum([
  "identity",
  "preferences",
  "goals",
  "projects",
  "relationships",
  "constraints",
  "habits",
  "notes",
]);
export type ProfileSectionView = z.infer<typeof ProfileSectionSchema>;

export const ProfileEntryViewSchema = z.object({
  id: z.string(),
  section: ProfileSectionSchema,
  key: z.string(),
  value: z.string(),
  source: z.enum(["user", "dormant", "import"]),
  provenance: z.string().optional(),
  validUntil: z.number().optional(),
  createdAt: z.number(),
  updatedAt: z.number(),
});
export type ProfileEntryView = z.infer<typeof ProfileEntryViewSchema>;

export const ProfileUpsertRequestSchema = z.object({
  section: ProfileSectionSchema,
  key: z.string().min(1),
  value: z.string(),
  validUntil: z.number().positive().optional(),
});
export type ProfileUpsertRequest = z.infer<typeof ProfileUpsertRequestSchema>;

// RFC-015 scheduler (persistent jobs + notifications)
export const ScheduleTriggerSchema = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("once"), at: z.number() }),
  z.object({
    kind: z.literal("interval"),
    everyMs: z.number().int().min(1_000),
    startAt: z.number().optional(),
  }),
  z.object({ kind: z.literal("daily"), time: z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/) }),
]);
export type ScheduleTriggerView = z.infer<typeof ScheduleTriggerSchema>;

export const ScheduledJobViewSchema = z.object({
  id: z.string(),
  title: z.string(),
  prompt: z.string(),
  conversationId: z.string().optional(),
  trigger: ScheduleTriggerSchema,
  enabled: z.boolean(),
  maxRuns: z.number().optional(),
  createdAt: z.number(),
  lastRunAt: z.number().optional(),
  nextRunAt: z.number().optional(),
  runCount: z.number(),
  lastStatus: z.enum(["completed", "failed"]).optional(),
  lastError: z.string().optional(),
});
export type ScheduledJobView = z.infer<typeof ScheduledJobViewSchema>;

export const ScheduleAddRequestSchema = z.object({
  title: z.string().min(1),
  prompt: z.string().min(1),
  conversationId: z.string().optional(),
  trigger: ScheduleTriggerSchema,
  enabled: z.boolean().optional(),
  maxRuns: z.number().int().positive().optional(),
});
export type ScheduleAddRequest = z.infer<typeof ScheduleAddRequestSchema>;

export const ScheduleUpdateRequestSchema = z.object({
  title: z.string().min(1).optional(),
  prompt: z.string().min(1).optional(),
  conversationId: z.string().optional(),
  trigger: ScheduleTriggerSchema.optional(),
  enabled: z.boolean().optional(),
  maxRuns: z.number().int().positive().optional(),
});
export type ScheduleUpdateRequest = z.infer<typeof ScheduleUpdateRequestSchema>;

// RFC-020 identity graph (proposal-confirm account linking)
export const IdentitySeedKindSchema = z.enum(["email", "phone", "wechat", "username", "url"]);
export type IdentitySeedKindView = z.infer<typeof IdentitySeedKindSchema>;

export const LinkedAccountViewSchema = z.object({
  id: z.string(),
  platform: z.string(),
  kind: IdentitySeedKindSchema,
  value: z.string(),
  url: z.string().optional(),
  source: z.enum(["gravatar", "mail", "calendar", "notes", "manual"]),
  evidence: z.string(),
  confidence: z.number(),
  verifiedAt: z.number().nullable(),
  createdAt: z.number(),
});
export type LinkedAccountView = z.infer<typeof LinkedAccountViewSchema>;

export const IdentityPersonViewSchema = z.object({
  id: z.string(),
  displayName: z.string(),
  accounts: z.array(LinkedAccountViewSchema),
  createdAt: z.number(),
  updatedAt: z.number(),
});
export type IdentityPersonView = z.infer<typeof IdentityPersonViewSchema>;

export const IdentityProposalViewSchema = z.object({
  id: z.string(),
  personId: z.string().nullable(),
  groupKey: z.string().optional(),
  displayName: z.string().optional(),
  account: z.object({
    platform: z.string(),
    kind: IdentitySeedKindSchema,
    value: z.string(),
    url: z.string().optional(),
    source: z.enum(["gravatar", "mail", "calendar", "notes", "manual"]),
    evidence: z.string(),
    confidence: z.number(),
    verifiedAt: z.number().nullable(),
  }),
});
export type IdentityProposalView = z.infer<typeof IdentityProposalViewSchema>;

export const IdentityResolveRequestSchema = z.object({
  kind: IdentitySeedKindSchema.optional(),
  value: z.string().min(1),
});
export type IdentityResolveRequest = z.infer<typeof IdentityResolveRequestSchema>;

export const IdentityResolveResultViewSchema = z.object({
  seed: z.object({ kind: IdentitySeedKindSchema, value: z.string() }),
  proposals: z.array(IdentityProposalViewSchema),
  reports: z.array(
    z.object({
      resolverId: z.string(),
      ok: z.boolean(),
      detail: z.string().optional(),
      count: z.number(),
    }),
  ),
});
export type IdentityResolveResultView = z.infer<typeof IdentityResolveResultViewSchema>;

export const IdentityConfirmRequestSchema = z.object({
  proposals: z.array(IdentityProposalViewSchema).min(1),
});
export type IdentityConfirmRequest = z.infer<typeof IdentityConfirmRequestSchema>;

export const IdentityAssertRequestSchema = z.object({
  personId: z.string().nullable().optional(),
  displayName: z.string().optional(),
  platform: z.string().min(1),
  kind: IdentitySeedKindSchema,
  value: z.string().min(1),
  url: z.string().optional(),
  evidence: z.string().optional(),
});
export type IdentityAssertRequest = z.infer<typeof IdentityAssertRequestSchema>;

export const IdentityUnlinkRequestSchema = z.object({
  accountId: z.string().min(1),
});
export type IdentityUnlinkRequest = z.infer<typeof IdentityUnlinkRequestSchema>;

export const SchedulerEventViewSchema = z.object({
  kind: z.enum(["fired", "completed", "failed", "deferred"]),
  job: ScheduledJobViewSchema,
  at: z.number(),
  lateMs: z.number().optional(),
  error: z.string().optional(),
  willRetry: z.boolean().optional(),
});
export type SchedulerEventView = z.infer<typeof SchedulerEventViewSchema>;

// RFC-016 connectors (status surface; tools stay agent-internal)
export const ConnectorStatusViewSchema = z.object({
  id: z.string(),
  displayName: z.string(),
  ok: z.boolean(),
  detail: z.string().optional(),
  toolCount: z.number(),
});
export type ConnectorStatusView = z.infer<typeof ConnectorStatusViewSchema>;

// RFC-012 TaskPool checkpoints (durable run snapshots + manual control)
export const TaskRunStateSchema = z.enum([
  "pending",
  "ready",
  "running",
  "completed",
  "failed",
  "timed_out",
  "cancelled",
]);
export type TaskRunStateView = z.infer<typeof TaskRunStateSchema>;

export const TaskCheckpointNodeViewSchema = z.object({
  taskId: z.string(),
  state: TaskRunStateSchema,
  attempt: z.number(),
  result: z.unknown().optional(),
  error: z.string().optional(),
  updatedAt: z.number(),
});
export type TaskCheckpointNodeView = z.infer<typeof TaskCheckpointNodeViewSchema>;

export const TaskCheckpointViewSchema = z.object({
  runId: z.string(),
  planId: z.string(),
  status: z.enum(["running", "completed", "failed", "cancelled"]),
  graph: z.object({
    planId: z.string(),
    goalIntent: z.string(),
    goalInput: z.string().optional(),
    nodes: z.array(
      z.object({
        id: z.string(),
        deps: z.array(z.string()),
        action: z.string(),
        description: z.string(),
        timeoutMs: z.number().optional(),
      }),
    ),
  }),
  nodes: z.array(TaskCheckpointNodeViewSchema),
  createdAt: z.number(),
  updatedAt: z.number(),
});
export type TaskCheckpointView = z.infer<typeof TaskCheckpointViewSchema>;

export const TaskRunResultViewSchema = z.object({
  runId: z.string(),
  planId: z.string(),
  status: z.enum(["completed", "failed", "cancelled"]),
  results: z.record(z.string(), z.unknown()),
  errors: z.record(z.string(), z.string()),
  states: z.record(z.string(), TaskRunStateSchema),
  attempts: z.record(z.string(), z.number()),
});
export type TaskRunResultView = z.infer<typeof TaskRunResultViewSchema>;

// RFC-012 §3 subagents (records surface; spawning stays model-side)
export const SubagentRecordViewSchema = z.object({
  id: z.string(),
  label: z.string(),
  task: z.string(),
  childConversationId: z.string(),
  status: z.enum(["running", "completed", "failed", "cancelled"]),
  startedAt: z.number(),
  endedAt: z.number().optional(),
  finalAnswer: z.string().optional(),
  error: z.string().optional(),
});
export type SubagentRecordView = z.infer<typeof SubagentRecordViewSchema>;

// RFC-018 outbound channels
export const ChannelStatusViewSchema = z.object({
  id: z.string(),
  kind: z.string(),
  ok: z.boolean(),
  detail: z.string().optional(),
  lastSentAt: z.number().optional(),
  lastError: z.string().optional(),
  sentCount: z.number(),
});
export type ChannelStatusView = z.infer<typeof ChannelStatusViewSchema>;

// RFC-019 sync + voice (bundle is opaque JSON at the IPC boundary; the main
// process validates structure via the sync package before applying)
export const SyncBundleViewSchema = z.object({
  version: z.literal(1),
  exportedAt: z.string(),
  profile: z.object({
    entries: z.array(z.record(z.string(), z.unknown())),
    dismissed: z.array(z.string()),
  }),
  schedules: z.array(z.record(z.string(), z.unknown())),
  // RFC-020: identity graph section (main validates deeply via the sync pkg).
  identity: z.object({ persons: z.array(z.record(z.string(), z.unknown())) }).optional(),
});
export type SyncBundleView = z.infer<typeof SyncBundleViewSchema>;

export const SyncApplyReportViewSchema = z.object({
  profileUpserted: z.number(),
  profileSkipped: z.number(),
  dismissedApplied: z.number(),
  schedulesAdded: z.number(),
  schedulesKept: z.number(),
  identityPersonsAdded: z.number(),
  identityAccountsAdded: z.number(),
});
export type SyncApplyReportView = z.infer<typeof SyncApplyReportViewSchema>;

export const VoiceTranscribeRequestSchema = z.object({
  audioBase64: z.string(),
  mimeType: z.string(),
  language: z.string().optional(),
});
export type VoiceTranscribeRequest = z.infer<typeof VoiceTranscribeRequestSchema>;

export const ChatResponseSchema = z.object({
  finalAnswer: z.string(),
  iterations: z.number().int().nonnegative(),
  status: z.string(),
  traceId: z.string(),
  provider: z.string().optional(),
  model: z.string().optional(),
  inputStructure: InputStructureSchema.optional(),
  context: ContextUsageSchema.optional(),
  artifacts: z.array(WorkspaceArtifactSchema).optional(),
  command: z
    .object({
      name: z.string(),
      kind: z.enum(["ok", "error"]),
    })
    .optional(),
});
export type ChatResponse = z.infer<typeof ChatResponseSchema>;

const RuntimeErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  retriable: z.boolean(),
  at: z.number(),
});

const ProviderAttemptSchema = z.object({
  provider: z.string(),
  outcome: z.string(),
  durationMs: z.number(),
  errorCode: z.string().optional(),
  errorMessage: z.string().optional(),
  ok: z.boolean(),
  reason: z.string().optional(),
});

export const StatusResponseSchema = z.object({
  llm: z.object({
    provider: z.string(),
    status: z.string(),
    model: z.string().optional(),
    mode: z.string().optional(),
    runtime: z
      .object({
        requestedProvider: z.string(),
        provider: z.string(),
        model: z.string(),
        mode: z.string(),
        status: z.string(),
        fallbackFrom: z.string().optional(),
        lastError: RuntimeErrorSchema.optional(),
      })
      .optional(),
  }),
  embed: z
    .object({
      requestedProvider: z.string(),
      provider: z.string(),
      model: z.string(),
      dimension: z.number(),
      mode: z.string(),
      status: z.string(),
      fallbackFrom: z.string().optional(),
      lastError: RuntimeErrorSchema.optional(),
      attempts: z.array(ProviderAttemptSchema).optional(),
    })
    .optional(),
  modelRuntime: z.object({
    llm: z.object({
      requestedProvider: z.string(),
      provider: z.string(),
      model: z.string(),
      mode: z.string(),
      status: z.string(),
      fallbackFrom: z.string().optional(),
      lastError: RuntimeErrorSchema.optional(),
      attempts: z.array(ProviderAttemptSchema),
    }),
    embed: z.object({
      requestedProvider: z.string(),
      provider: z.string(),
      model: z.string(),
      dimension: z.number(),
      mode: z.string(),
      status: z.string(),
      fallbackFrom: z.string().optional(),
      lastError: RuntimeErrorSchema.optional(),
      attempts: z.array(ProviderAttemptSchema),
    }),
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
  /** Phase 3.1 ??????????+ ?????real ????????*/
  persistence: z
    .object({
      mode: z.enum(["memory", "real"]),
      dataDir: z.string().optional(),
    })
    .optional(),
  /** Phase 3.3 ?????????vector / hybrid???*/
  retrievalMode: z.enum(["vector", "hybrid"]).optional(),
  classifier: z.object({
    enabled: z.boolean(),
    impliedByTaskPool: z.boolean(),
  }),
  taskPool: z
    .object({
      requested: z.boolean(),
      active: z.boolean(),
      classifierRequired: z.literal(true),
      classifierEnabled: z.boolean(),
      reason: z.enum(["disabled", "classifier_required", "ready"]),
      eligibleTaskTypes: z.tuple([z.literal("planning"), z.literal("analysis")]),
      precedence: z.literal("taskpool-before-self-consistency"),
      persistence: z.enum(["none", "sqlite"]),
      recovery: z.enum(["unsupported", "cancel", "resume"]),
      recoverySummary: z
        .object({
          policy: z.enum(["cancel", "resume"]),
          found: z.number(),
          resumed: z.number(),
          completed: z.number(),
          cancelled: z.number(),
          failed: z.number(),
        })
        .optional(),
    })
    .optional(),
  /** RFC-006 当前行为契约版本与 A/B cohort。 */
  productBehavior: z
    .object({
      version: z.string(),
      enabled: z.boolean(),
      cohort: z.enum(["treatment", "control"]),
    })
    .optional(),
  /** Phase 3.3/3.4 ??Dormant ???????????????*/
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
  /** 技能系统 / 自学习：始终返回，便于状态栏与技能面板开关。 */
  skills: z
    .object({
      enabled: z.boolean(),
      learning: z.boolean(),
      catalogSize: z.number(),
      pendingProposals: z.number(),
      activeSkills: z.number(),
    })
    .optional(),
  context: ContextUsageSchema.optional(),
});
export type StatusResponse = z.infer<typeof StatusResponseSchema>;

export const MemoryQueryRequestSchema = z.object({
  query: z.string().optional(),
  topK: z.number().int().positive().max(50).default(10),
  /** ??????????? agent.retrievalMode ????*/
  mode: z.enum(["vector", "hybrid"]).optional(),
  /** hybrid ????????RRF ????*/
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

// ---------- Dormant Memory Learning (RFC-003 ?? 3) ----------

export const DormantProposalDecisionSchema = z.object({
  proposalId: z.string(),
});
export type DormantProposalDecision = z.infer<typeof DormantProposalDecisionSchema>;

export const DormantListRequestSchema = z.object({
  status: z.enum(["pending", "approved", "rejected", "applied", "revoked"]).optional(),
});
export type DormantListRequest = z.infer<typeof DormantListRequestSchema>;

// ---------- Skill Learning (技能自学习 Phase 2) ----------

export const SkillProposalDecisionSchema = z.object({
  proposalId: z.string(),
});
export type SkillProposalDecision = z.infer<typeof SkillProposalDecisionSchema>;

export const SkillListRequestSchema = z.object({
  status: z.enum(["pending", "approved", "rejected", "revoked"]).optional(),
});
export type SkillListRequest = z.infer<typeof SkillListRequestSchema>;

/** 技能提案 DTO（main → renderer；来自 learning-runtime 的 SkillProposal 投影）。 */
export const SkillProposalDtoSchema = z.object({
  proposalId: z.string(),
  skillId: z.string(),
  name: z.string(),
  description: z.string(),
  status: z.enum(["pending", "approved", "rejected", "revoked"]),
  ts: z.number(),
  decidedAt: z.number().optional(),
  evidence: z.object({
    queries: z.array(z.string()),
    taskType: z.string().optional(),
    count: z.number(),
  }),
});
export type SkillProposalDto = z.infer<typeof SkillProposalDtoSchema>;

export const SkillListResponseSchema = z.object({
  total: z.number(),
  proposals: z.array(SkillProposalDtoSchema),
});
export type SkillListResponse = z.infer<typeof SkillListResponseSchema>;

export const SkillDistillResponseSchema = z.object({
  produced: z.number(),
  proposals: z.array(SkillProposalDtoSchema),
});
export type SkillDistillResponse = z.infer<typeof SkillDistillResponseSchema>;

export const SkillDecisionResponseSchema = z.object({
  proposalId: z.string(),
  status: z.enum(["pending", "approved", "rejected", "revoked"]),
  decidedAt: z.number().optional(),
});
export type SkillDecisionResponse = z.infer<typeof SkillDecisionResponseSchema>;

/** 当前生效的学习技能 + 权重。 */
export const SkillActiveDtoSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  source: z.string().optional(),
  weight: z.number(),
});
export type SkillActiveDto = z.infer<typeof SkillActiveDtoSchema>;

export const SkillActiveResponseSchema = z.object({
  total: z.number(),
  skills: z.array(SkillActiveDtoSchema),
});
export type SkillActiveResponse = z.infer<typeof SkillActiveResponseSchema>;

/** 已装技能目录（作者 SKILL.md + 已批准学习技能；不含正文）。 */
export const SkillCatalogDtoSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  source: z.string(),
  version: z.string(),
  tools: z.array(z.string()),
  triggers: z.array(z.string()),
  taskTypes: z.array(z.string()),
  origin: z.enum(["authored", "learned"]),
});
export type SkillCatalogDto = z.infer<typeof SkillCatalogDtoSchema>;

export const SkillCatalogResponseSchema = z.object({
  enabled: z.boolean(),
  learning: z.boolean(),
  total: z.number(),
  skills: z.array(SkillCatalogDtoSchema),
});
export type SkillCatalogResponse = z.infer<typeof SkillCatalogResponseSchema>;

/** 技能自学习未启用 / 请求非法时的统一错误 shape。 */
export const SkillLearningErrorSchema = z.object({
  error: z.union([
    z.literal("skills_learning_not_enabled"),
    z.literal("invalid_request"),
    z.literal("not_found_or_already_decided"),
    z.literal("not_found_or_not_approved"),
  ]),
  issues: z.array(z.unknown()).optional(),
  hint: z.string().optional(),
});
export type SkillLearningError = z.infer<typeof SkillLearningErrorSchema>;

/** ?? proposal ??IPC ???????pattern ????description???*/
export const DormantProposalDtoSchema = z.object({
  proposalId: z.string(),
  targetField: z.string(),
  value: z.unknown(),
  status: z.enum(["pending", "approved", "rejected", "applied", "revoked"]),
  ts: z.number(),
  decidedAt: z.number().optional(),
  patternDescription: z.string(),
  confidence: z.number(),
  frequency: z.number(),
});
export type DormantProposalDto = z.infer<typeof DormantProposalDtoSchema>;

/** Phase 3.3 ?????????dormant.* channel ????shape??*/
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
      status: z.enum(["pending", "approved", "rejected", "applied", "revoked"]),
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
  status: z.enum(["pending", "approved", "rejected", "applied", "revoked"]),
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

// ---------- Auto Update (electron-updater, #6) ----------

/** ?????????main ??renderer ?????*/
export const UpdateEventSchema = z.object({
  status: z.enum([
    "checking",
    "available",
    "not-available",
    "downloading",
    "downloaded",
    "error",
    "disabled",
  ]),
  /** ??????available / downloaded ??????*/
  version: z.string().optional(),
  /** ????????0-100?downloading ??????*/
  percent: z.number().optional(),
  /** ?????????*/
  message: z.string().optional(),
});
export type UpdateEvent = z.infer<typeof UpdateEventSchema>;

// ---------- Workspace ??????RFC-004 ?8??----------

/** ????????path ??????????????*/
export const WorkspaceReadRequestSchema = z.object({
  path: z.string().min(1),
});
export type WorkspaceReadRequest = z.infer<typeof WorkspaceReadRequestSchema>;

export const WorkspaceReadResponseSchema = z.object({
  path: z.string(),
  content: z.string(),
  bytes: z.number().int().nonnegative(),
});
export type WorkspaceReadResponse = z.infer<typeof WorkspaceReadResponseSchema>;

/** ????????path ??????????????*/
export const WorkspaceWriteRequestSchema = z.object({
  path: z.string().min(1),
  content: z.string(),
});
export type WorkspaceWriteRequest = z.infer<typeof WorkspaceWriteRequestSchema>;

export const WorkspaceWriteResponseSchema = z.object({
  path: z.string(),
  bytesWritten: z.number().int().nonnegative(),
});
export type WorkspaceWriteResponse = z.infer<typeof WorkspaceWriteResponseSchema>;

/** ???????????????*/
export const WorkspaceInfoSchema = z.object({
  root: z.string(),
  enableCommands: z.boolean(),
  allowedCommands: z.array(z.string()),
});
export type WorkspaceInfo = z.infer<typeof WorkspaceInfoSchema>;

export const WorkspaceListResponseSchema = z.object({
  root: z.string(),
  entries: z.array(
    z.object({
      path: z.string(),
      name: z.string(),
      type: z.enum(["file", "dir"]),
      bytes: z.number().int().nonnegative().optional(),
    }),
  ),
});
export type WorkspaceListResponse = z.infer<typeof WorkspaceListResponseSchema>;

export const WorkspaceBytesRequestSchema = z.object({
  path: z.string().min(1),
});
export type WorkspaceBytesRequest = z.infer<typeof WorkspaceBytesRequestSchema>;

export const WorkspaceBytesResponseSchema = z.object({
  path: z.string(),
  mimeType: z.string(),
  base64: z.string(),
  bytes: z.number().int().nonnegative(),
});
export type WorkspaceBytesResponse = z.infer<typeof WorkspaceBytesResponseSchema>;

export const WorkspaceRevealRequestSchema = z.object({
  path: z.string().optional(),
});
export type WorkspaceRevealRequest = z.infer<typeof WorkspaceRevealRequestSchema>;

export const WorkspaceRevealResponseSchema = z.object({
  ok: z.boolean(),
  error: z.string().optional(),
});
export type WorkspaceRevealResponse = z.infer<typeof WorkspaceRevealResponseSchema>;

/** pickWorkspaceDir ??????????????*/
export const WorkspacePickResponseSchema = z.object({
  canceled: z.boolean(),
  root: z.string().optional(),
});
export type WorkspacePickResponse = z.infer<typeof WorkspacePickResponseSchema>;

/** ??????????main ??renderer ????? fs.watch???*/
export const WorkspaceChangeEventSchema = z.object({
  event: z.enum(["rename", "change"]),
  /** ???????????fs.watch ??????????*/
  path: z.string(),
});
export type WorkspaceChangeEvent = z.infer<typeof WorkspaceChangeEventSchema>;

/** ???????????? shape??*/
export const WorkspaceErrorSchema = z.object({
  error: z.literal("workspace_error"),
  message: z.string(),
});
export type WorkspaceError = z.infer<typeof WorkspaceErrorSchema>;

// ---------- ??????getConfig / updateConfig??----------

export const ModelProviderSchema = z.enum([
  "auto",
  "ollama",
  "hunyuan",
  "kimi",
  "minimax",
  "glm",
  "mock",
]);
export type ModelProvider = z.infer<typeof ModelProviderSchema>;

export const ModelProfileSchema = z.object({
  id: z.string().min(1).max(128),
  name: z.string().min(1).max(80),
  provider: ModelProviderSchema,
  model: z.string().min(1).max(200),
  baseUrl: z.string().url().optional(),
  hasCredential: z.boolean().optional(),
});
export type ModelProfile = z.infer<typeof ModelProfileSchema>;

export const DEFAULT_DESKTOP_MODEL_PROFILE_ID = "hunyuan-hy4-preview";
export const MODEL_DEFAULTS_VERSION = 2;

export const DEFAULT_MODEL_PROFILES: readonly ModelProfile[] = [
  {
    id: "hunyuan-hy4-preview",
    name: "腾讯混元 Hy4 Preview",
    provider: "hunyuan",
    model: "hy4-preview",
  },
  { id: "hunyuan-hy3", name: "腾讯混元 Hy3", provider: "hunyuan", model: "hy3" },
  { id: "auto", name: "自动选择", provider: "auto", model: "auto", hasCredential: true },
  {
    id: "ollama",
    name: "Ollama 本地",
    provider: "ollama",
    model: "qwen2.5:7b",
    hasCredential: true,
  },
  { id: "kimi-k3", name: "Kimi K3", provider: "kimi", model: "kimi-k3" },
  { id: "minimax-m3", name: "MiniMax M3", provider: "minimax", model: "MiniMax-M3" },
  { id: "glm-5.2", name: "GLM 5.2", provider: "glm", model: "glm-5.2" },
  { id: "mock", name: "Mock（开发）", provider: "mock", model: "mock", hasCredential: true },
];

export const ModelCredentialSetSchema = z.object({
  profileId: z.string().min(1).max(128),
  apiKey: z.string().min(1).max(4096),
});
export type ModelCredentialSet = z.infer<typeof ModelCredentialSetSchema>;

export const WorkbenchWorkspaceSchema = z.object({
  id: z.string(),
  name: z.string(),
  rootPath: z.string(),
  dataDir: z.string().optional(),
  createdAt: z.number(),
  updatedAt: z.number(),
});
export type WorkbenchWorkspace = z.infer<typeof WorkbenchWorkspaceSchema>;

export const WorkbenchTaskSchema = z.object({
  id: z.string(),
  workspaceId: z.string(),
  title: z.string(),
  status: z.enum(["active", "completed", "archived"]),
  taskPoolRunId: z.string().optional(),
  createdAt: z.number(),
  updatedAt: z.number(),
});
export type WorkbenchTask = z.infer<typeof WorkbenchTaskSchema>;

export const WorkbenchConversationSchema = z.object({
  id: z.string(),
  taskId: z.string(),
  title: z.string(),
  modelProfileId: z.string(),
  createdAt: z.number(),
  updatedAt: z.number(),
});
export type WorkbenchConversation = z.infer<typeof WorkbenchConversationSchema>;

export const WorkbenchMessageSchema = z.object({
  id: z.string(),
  conversationId: z.string(),
  role: z.enum(["user", "assistant", "system"]),
  content: z.string(),
  traceId: z.string().optional(),
  tokens: z.number().int().nonnegative().optional(),
  status: z.string().optional(),
  messageKind: z.enum(["message", "answer", "clarification"]).default("message"),
  inputStructure: InputStructureSchema.optional(),
  createdAt: z.number(),
});
export type WorkbenchMessage = z.infer<typeof WorkbenchMessageSchema>;

export const WorkbenchCreateSchema = z.object({
  parentId: z.string().optional(),
  name: z.string().min(1).max(200).optional(),
  title: z.string().min(1).max(200).optional(),
  rootPath: z.string().optional(),
  modelProfileId: z.string().optional(),
});

export const WorkbenchUpdateSchema = z.object({
  id: z.string().min(1),
  title: z.string().min(1).max(200).optional(),
  name: z.string().min(1).max(200).optional(),
  status: z.enum(["active", "completed", "archived"]).optional(),
  modelProfileId: z.string().optional(),
  taskPoolRunId: z.string().optional(),
});

export const WindowChromeActionSchema = z.enum([
  "quit",
  "reload",
  "toggleDevTools",
  "toggleFullScreen",
  "minimize",
  "maximize",
  "close",
]);

export type WindowChromeAction = z.infer<typeof WindowChromeActionSchema>;

/** 与 titleBarOverlay.height 一致，自定义标题栏和原生窗口按钮同一行。 */
export const TITLE_BAR_HEIGHT = 32;

/**
 * ?????????????????updateConfig ?????? * ??????**????**??????????workspaceDir ????????fs.watch ?????? */
export const AppConfigSchema = z.object({
  workspaceDir: z.string().optional(),
  llmProvider: ModelProviderSchema.optional(),
  modelProfiles: z.array(ModelProfileSchema.omit({ hasCredential: true })).optional(),
  activeModelProfileId: z.string().optional(),
  modelDefaultsVersion: z.number().int().nonnegative().optional(),
  activeWorkspaceId: z.string().optional(),
  activeTaskId: z.string().optional(),
  activeConversationId: z.string().optional(),
  embedProvider: z.enum(["auto", "simple", "ollama", "xenova", "mock"]).optional(),
  ollamaBaseUrl: z.string().optional(),
  ollamaModel: z.string().optional(),
  ollamaEmbedModel: z.string().optional(),
  retrievalMode: z.enum(["vector", "hybrid"]).optional(),
  enableCommands: z.boolean().optional(),
  allowedCommands: z.array(z.string()).optional(),
  enableDormant: z.boolean().optional(),
  /** 是否注入已批准的钝化记忆 persona（A/B 杠杆，仅 enableDormant 时有意义）。 */
  enablePersona: z.boolean().optional(),
  /** RFC-006 Product Behavior treatment/control A/B。 */
  enableProductBehavior: z.boolean().optional(),
  /** 技能系统（Phase 1 作者能力包）。 */
  enableSkills: z.boolean().optional(),
  /** 技能自学习闭环（Phase 2，隐含开启 enableSkills）。 */
  enableSkillLearning: z.boolean().optional(),
  /** 前端可强化分类器。 */
  enableClassifier: z.boolean().optional(),
  /** RFC-007 bounded DAG orchestration for planning/analysis. */
  enableTaskPool: z.boolean().optional(),
  autoUpdate: z.boolean().optional(),
});
export type AppConfig = z.infer<typeof AppConfigSchema>;

/** updateConfig ???AppConfig ???????*/
export const AppConfigPatchSchema = AppConfigSchema.partial();
export type AppConfigPatch = z.infer<typeof AppConfigPatchSchema>;

/** Keys that require reassembling the Desktop agent (in-process hot reload). */
export const AGENT_RUNTIME_CONFIG_KEYS = [
  "workspaceDir",
  "llmProvider",
  "embedProvider",
  "ollamaBaseUrl",
  "ollamaModel",
  "ollamaEmbedModel",
  "retrievalMode",
  "enableCommands",
  "allowedCommands",
  "enableDormant",
  "enablePersona",
  "enableProductBehavior",
  "enableSkills",
  "enableSkillLearning",
  "enableClassifier",
  "enableTaskPool",
] as const satisfies ReadonlyArray<keyof AppConfig>;

export const isAgentRuntimeConfigPatch = (patch: AppConfigPatch): boolean =>
  AGENT_RUNTIME_CONFIG_KEYS.some((key) => patch[key] !== undefined);

/** Phase 3.3 ??approve/reject ???????? shape??*/
export const DormantDecisionErrorSchema = z.object({
  error: z.union([
    z.literal("dormant_not_enabled"),
    z.literal("not_found_or_already_decided"),
    z.literal("not_found_or_not_applied"),
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
  // G5.2 记忆审核台：状态面 + 影子候选审核（approve 由 host 执行 mutation）
  MEMORY_STATUS: "openintj:memory.status",
  MEMORY_CANDIDATES: "openintj:memory.candidates",
  MEMORY_CANDIDATE_APPROVE: "openintj:memory.candidate.approve",
  MEMORY_CANDIDATE_REJECT: "openintj:memory.candidate.reject",
  AUDIT_RECENT: "openintj:audit.recent",
  // RFC-003 ?? 3?Dormant Memory Learning
  DORMANT_MINE: "openintj:dormant.mine",
  DORMANT_LIST: "openintj:dormant.list",
  DORMANT_APPROVE: "openintj:dormant.approve",
  DORMANT_REJECT: "openintj:dormant.reject",
  DORMANT_REVOKE: "openintj:dormant.revoke",
  DORMANT_PERSONA: "openintj:dormant.persona",
  // 技能自学习 Phase 2
  SKILLS_DISTILL: "openintj:skills.distill",
  SKILLS_LIST: "openintj:skills.list",
  SKILLS_APPROVE: "openintj:skills.approve",
  SKILLS_REJECT: "openintj:skills.reject",
  SKILLS_REVOKE: "openintj:skills.revoke",
  SKILLS_ACTIVE: "openintj:skills.active",
  SKILLS_CATALOG: "openintj:skills.catalog",
  // G5.3 技能验证器：用户对最近一轮回答的反馈（事后追认/否决）
  SKILLS_FEEDBACK: "openintj:skills.feedback",
  // #6 ????
  UPDATE_CHECK: "openintj:update.check",
  UPDATE_INSTALL: "openintj:update.install",
  // RFC-004 ?8 ????????
  WORKSPACE_READ: "openintj:workspace.read",
  WORKSPACE_WRITE: "openintj:workspace.write",
  WORKSPACE_INFO: "openintj:workspace.info",
  WORKSPACE_PICK_DIR: "openintj:workspace.pickDir",
  WORKSPACE_LIST: "openintj:workspace.list",
  WORKSPACE_READ_BYTES: "openintj:workspace.readBytes",
  WORKSPACE_REVEAL: "openintj:workspace.reveal",
  // ??????
  CONFIG_GET: "openintj:config.get",
  CONFIG_UPDATE: "openintj:config.update",
  MODEL_PROFILES: "openintj:model.profiles",
  MODEL_CREDENTIAL_SET: "openintj:model.credential.set",
  MODEL_CREDENTIAL_DELETE: "openintj:model.credential.delete",
  MODEL_TEST: "openintj:model.test",
  APP_RESTART: "openintj:app.restart",
  WINDOW_CHROME: "openintj:window.chrome",
  WORKBENCH_BOOTSTRAP: "openintj:workbench.bootstrap",
  WORKBENCH_WORKSPACE_CREATE: "openintj:workbench.workspace.create",
  WORKBENCH_WORKSPACE_CLOSE: "openintj:workbench.workspace.close",
  WORKBENCH_TASK_CREATE: "openintj:workbench.task.create",
  WORKBENCH_TASK_UPDATE: "openintj:workbench.task.update",
  WORKBENCH_CONVERSATION_CREATE: "openintj:workbench.conversation.create",
  WORKBENCH_CONVERSATION_UPDATE: "openintj:workbench.conversation.update",
  WORKBENCH_MESSAGES: "openintj:workbench.messages",
  WORKBENCH_MESSAGES_CLEAR: "openintj:workbench.messages.clear",
  COMMANDS_INVOKE: "openintj:commands.invoke",
  AGENT_STEER: "openintj:agent.steer",
  AGENT_INJECT: "openintj:agent.inject",
  AGENT_CANCEL: "openintj:agent.cancel",
  APPROVAL_LIST: "openintj:approval.list",
  APPROVAL_DECIDE: "openintj:approval.decide",
  GRANTS_LIST: "openintj:grants.list",
  GRANTS_GRANT: "openintj:grants.grant",
  GRANTS_REVOKE: "openintj:grants.revoke",
  PROFILE_LIST: "openintj:profile.list",
  PROFILE_UPSERT: "openintj:profile.upsert",
  PROFILE_REMOVE: "openintj:profile.remove",
  PROFILE_EXPORT: "openintj:profile.export",
  SCHEDULE_LIST: "openintj:schedule.list",
  SCHEDULE_ADD: "openintj:schedule.add",
  SCHEDULE_UPDATE: "openintj:schedule.update",
  SCHEDULE_REMOVE: "openintj:schedule.remove",
  IDENTITY_PERSONS: "openintj:identity.persons",
  IDENTITY_RESOLVE: "openintj:identity.resolve",
  IDENTITY_CONFIRM: "openintj:identity.confirm",
  IDENTITY_ASSERT: "openintj:identity.assert",
  IDENTITY_UNLINK: "openintj:identity.unlink",
  CONNECTORS_LIST: "openintj:connectors.list",
  TASKPOOL_CHECKPOINTS: "openintj:taskpool.checkpoints",
  TASKPOOL_RESUME: "openintj:taskpool.resume",
  TASKPOOL_CANCEL: "openintj:taskpool.cancel",
  SUBAGENTS_LIST: "openintj:subagents.list",
  CHANNELS_LIST: "openintj:channels.list",
  CHANNELS_TEST: "openintj:channels.test",
  SYNC_EXPORT: "openintj:sync.export",
  SYNC_IMPORT: "openintj:sync.import",
  VOICE_TRANSCRIBE: "openintj:voice.transcribe",
  // server-push events
  EVT_TAO: "openintj:evt.tao",
  EVT_REACT: "openintj:evt.react",
  EVT_AUDIT: "openintj:evt.audit",
  EVT_UPDATE: "openintj:evt.update",
  EVT_WORKSPACE: "openintj:evt.workspace",
  EVT_SKILL: "openintj:evt.skill",
  EVT_ASSISTANT_DELTA: "openintj:evt.assistant.delta",
  EVT_APPROVAL: "openintj:evt.approval",
  EVT_SCHEDULE: "openintj:evt.schedule",
} as const;

export type IpcChannel = (typeof IPC)[keyof typeof IPC];
