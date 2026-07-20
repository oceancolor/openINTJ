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

export const ChatResponseSchema = z.object({
  finalAnswer: z.string(),
  iterations: z.number().int().nonnegative(),
  status: z.string(),
  traceId: z.string(),
  provider: z.string().optional(),
  model: z.string().optional(),
  inputStructure: InputStructureSchema.optional(),
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
  /** 技能自学习（Phase 2）：只有 enableSkillLearning 时存在。 */
  skills: z
    .object({
      enabled: z.literal(true),
      pendingProposals: z.number(),
      activeSkills: z.number(),
    })
    .optional(),
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

export const DEFAULT_DESKTOP_MODEL_PROFILE_ID = "hunyuan-hy3";

export const DEFAULT_MODEL_PROFILES: readonly ModelProfile[] = [
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

/**
 * ?????????????????updateConfig ?????? * ??????**????**??????????workspaceDir ????????fs.watch ?????? */
export const AppConfigSchema = z.object({
  workspaceDir: z.string().optional(),
  llmProvider: ModelProviderSchema.optional(),
  modelProfiles: z.array(ModelProfileSchema.omit({ hasCredential: true })).optional(),
  activeModelProfileId: z.string().optional(),
  modelDefaultsVersion: z.literal(1).optional(),
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
  // #6 ????
  UPDATE_CHECK: "openintj:update.check",
  UPDATE_INSTALL: "openintj:update.install",
  // RFC-004 ?8 ????????
  WORKSPACE_READ: "openintj:workspace.read",
  WORKSPACE_WRITE: "openintj:workspace.write",
  WORKSPACE_INFO: "openintj:workspace.info",
  WORKSPACE_PICK_DIR: "openintj:workspace.pickDir",
  // ??????
  CONFIG_GET: "openintj:config.get",
  CONFIG_UPDATE: "openintj:config.update",
  MODEL_PROFILES: "openintj:model.profiles",
  MODEL_CREDENTIAL_SET: "openintj:model.credential.set",
  MODEL_CREDENTIAL_DELETE: "openintj:model.credential.delete",
  MODEL_TEST: "openintj:model.test",
  APP_RESTART: "openintj:app.restart",
  WORKBENCH_BOOTSTRAP: "openintj:workbench.bootstrap",
  WORKBENCH_WORKSPACE_CREATE: "openintj:workbench.workspace.create",
  WORKBENCH_TASK_CREATE: "openintj:workbench.task.create",
  WORKBENCH_TASK_UPDATE: "openintj:workbench.task.update",
  WORKBENCH_CONVERSATION_CREATE: "openintj:workbench.conversation.create",
  WORKBENCH_CONVERSATION_UPDATE: "openintj:workbench.conversation.update",
  WORKBENCH_MESSAGES: "openintj:workbench.messages",
  // server-push events
  EVT_TAO: "openintj:evt.tao",
  EVT_REACT: "openintj:evt.react",
  EVT_AUDIT: "openintj:evt.audit",
  EVT_UPDATE: "openintj:evt.update",
  EVT_WORKSPACE: "openintj:evt.workspace",
} as const;

export type IpcChannel = (typeof IPC)[keyof typeof IPC];
