import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import {
  type ChannelDispatcher,
  channelEventFromScheduler,
  channelsFromEnv,
} from "@openintj/channels";
import { DEFAULT_SEEDS, ReinforcingClassifier, outcomeSignal } from "@openintj/classifier";
import { type RateLimitOpts, RateLimitedLlmClient, forkJoin } from "@openintj/concurrency";
import {
  type Connector,
  ConnectorRegistry,
  connectorsFromEnv,
  parseIcs,
  parseMbox,
} from "@openintj/connectors";
import {
  type AgentHandle,
  type ChatMessage,
  DEFAULT_REACT_CONFIG,
  DEFAULT_TAO_CONFIG,
  HookBus,
  type InputStructuredTaoResult,
  type InputStructuringPolicy,
  type Journal,
  type JournalTurnContext,
  type LlmClient,
  ReactStateMachine,
  type SlashCommandRegistry,
  TaoLoop,
  TaskType,
  type TaskTypeType,
  attachProductTraitSignals,
  createAgentHandle,
  createBuiltinSlashCommands,
  createInMemoryJournal,
  createSlashCommandRegistry,
  detectTaskType,
  getShaderForTask,
  inputClarificationResult,
  invokeSlashCommand,
  resolveInputStructuringConfig,
  slashCommandResultToTao,
} from "@openintj/core";
import {
  type DormantPersistenceAdapter,
  DormantRuntime,
  type DormantRuntimeOpts,
  runMineInWorker,
} from "@openintj/dormant";
import {
  type IdentityDocument,
  type IdentityPerson,
  IdentityService,
  InMemoryIdentityStore,
  JsonIdentityStore,
  createGravatarResolver,
  createLocalCorpusResolver,
  derivePrimaryIdentity,
} from "@openintj/identity";
import { HunyuanClient, createHunyuanSearchTool } from "@openintj/llm-hunyuan";
import {
  type EmbedProviderId,
  type LlmProviderId,
  type ModelRuntimeStatus,
  resolveModelRuntime,
  validateEmbeddingFingerprintForDataDir,
} from "@openintj/model-runtime";
import { ControlPlane } from "@openintj/plane-control";
import {
  Executor,
  ToolHub,
  type WorkspaceTools,
  createWebSearchTool,
  createWorkspaceTools,
  resolveWebSearchConfig,
} from "@openintj/plane-execution";
import {
  type ApprovalBroker,
  GovernancePlane,
  type GrantStore,
  JsonGrantStore,
  type PendingApprovalBroker,
  TOOL_RISK_ORDER,
  type ToolRisk,
  classifyToolRisk,
  createApprovalSetup,
  createToolCallGate,
  riskAtLeast,
} from "@openintj/plane-governance";
import {
  ContextEngine,
  MemoryPlane,
  type PersistenceMode,
  type PersistentMemoryStore,
  createPersistentMemoryStore,
  fragmentsToRanked,
} from "@openintj/plane-memory";
import { InMemoryProfileStore, JsonProfileStore, ProfileService } from "@openintj/profile";
import {
  type Compactor,
  type PromptSectionRegistry,
  type SubagentSpawner,
  buildJournalTurnContext,
  createCompactor,
  createDefaultPromptSections,
  createDelegateTool,
  createPromptSectionRegistry,
  createSubagentSpawner,
  prepareRunPrelude,
  runCompactCommand,
} from "@openintj/runtime";
import {
  InMemoryScheduleStore,
  JsonScheduleStore,
  type QuietHours,
  SchedulerService,
} from "@openintj/scheduler";
import {
  DEFAULT_AGENT_SYSTEM_PROMPT,
  PRODUCT_BEHAVIOR_VERSION,
  type ResolvedWorkspaceConfig,
  type SelfConsistencyStrategy,
  appendSourcesFooter,
  enforceProductBehaviorAnswer,
  isCommandExecutionAvailable,
  resolvePersonaInjection,
  resolveProductBehaviorEnabled,
  resolveSearchEvidenceStatus,
  resolveSelfConsistency,
  resolveWorkspaceConfig,
  selectConsistentAnswer,
} from "@openintj/shared";
import {
  DbSkillSource,
  InMemorySkillStore,
  type SkillContext,
  SkillLearningRuntime,
  type SkillStore,
  assembleSkillContext,
  createLlmSkillDistiller,
  resolveSkillWeightHalfLifeSec,
} from "@openintj/skills";
import {
  SqliteTaskStore,
  createSqliteClassifierStore,
  createSqliteDormantStore,
  createSqliteSkillStore,
} from "@openintj/storage-sqlite";
import {
  type SyncApplyReport,
  type SyncBundle,
  applySyncBundle,
  buildSyncBundle,
} from "@openintj/sync";
import {
  type HybridConfig,
  type MemoryHybridHit,
  MemoryHybridIndex,
  type StoredTaskRun,
  type TaskGraphNode,
  TaskPool,
  type TaskPoolActivationStatus,
  type TaskPoolRecoveryPolicy,
  type TaskPoolRecoverySummary,
  type TaskRunResult,
  type TaskWorkerContext,
  planGraphToTaskGraph,
  resolveOrchestrationMode,
  resolveTaskPoolActivation,
  resolveTaskPoolEnabled,
  resolveTaskPoolRecoveryPolicy,
  shouldUseTaskPool,
  synthesizeTaskPoolAnswer,
} from "@openintj/taskpool";
import {
  type AttachOtelOpts,
  type AttachedOtel,
  attachOtelToHooks,
} from "@openintj/telemetry-otel";
import { type SpeechToTextProvider, sttFromEnv } from "@openintj/voice";
import {
  fulfillImageGenerationTask,
  requestsImageGeneration,
  taskIdFromContextTags,
} from "./image-task.js";

export type LlmProvider = LlmProviderId;

export interface DesktopAgentOpts {
  llmProvider?: LlmProviderId;
  embedProvider?: EmbedProviderId;
  systemPrompt?: string;
  maxTaoIterations?: number;
  /**
   * 持久化数据目录。Electron 主入口通常传 `path.join(app.getPath("userData"), "memory-store")`。
   * 不传 → in-memory（仅测试 / 显式 OPENINTJ_DESKTOP_NO_PERSIST=1 时用）。
   */
  dataDir?: string;
  /** 显式锁定模式（覆盖 dataDir 推断）。 */
  persistenceMode?: PersistenceMode;
  /** 向量维度（默认 64）。 */
  embeddingDim?: number;
  /** RFC-003 方向 3：Dormant Memory Learning（默认关）。env OPENINTJ_DORMANT=1 也启用。 */
  enableDormant?: boolean;
  dormantOpts?: DormantRuntimeOpts;
  /**
   * 是否把已批准的钝化记忆 persona 注入 system prompt（A/B 杠杆，仅 enableDormant 时有意义）。
   * 默认开；env `OPENINTJ_PERSONA=0` 关闭 → 无 persona 基线组（RFC-003 §3.6 验收 #3）。
   */
  enablePersona?: boolean;
  /** RFC-006 Product Behavior A/B；默认开，env OPENINTJ_PRODUCT_BEHAVIOR=0 可关。 */
  enableProductBehavior?: boolean;
  /** RFC-008 输入结构化；Product Behavior control 组强制关闭。 */
  inputStructuring?: InputStructuringPolicy;
  /**
   * Dormant 持久化策略（仅 enableDormant=true 时生效）。
   * - 'auto'（默认）：跟随主持久化（real → SqliteDormantStore，memory → 不挂 adapter）
   * - 'memory'：强制不挂 adapter
   * - 'real'：强制用 SqliteDormantStore（缺 dbPath 报错）
   */
  dormantPersistence?: "auto" | "memory" | "real";
  /** 自定义 dormant SQLite 文件路径；缺省 `${dataDir}/dormant.sqlite`。 */
  dormantDbPath?: string;
  /**
   * 把 CPU 密集的钝化记忆挖掘下放 worker 线程（RFC-004 §2 utility 蒸馏 worker）。
   * 默认关；env `OPENINTJ_DORMANT_WORKER=1` 也启用。失败自动回退主线程内联挖掘。
   */
  dormantMineWorker?: boolean;
  /** RFC-003 方向 2：默认检索模式。env OPENINTJ_RETRIEVAL_MODE=hybrid 也启用。 */
  retrievalMode?: "vector" | "hybrid";
  /** RFC-003 方向 1：LLM 速率限制。env OPENINTJ_RATE_LIMIT_QPS 也启用。 */
  rateLimit?: RateLimitOpts;
  /**
   * RFC-003 方向一/二接入：opt-in 自一致性（并行多采样 + 投票）。
   * samples>1 时每次 run 用 forkJoin 并行跑 N 个 tao.run，再按 strategy 选最终答案。
   * 默认关闭；env OPENINTJ_SELF_CONSISTENCY=N / OPENINTJ_SELF_CONSISTENCY_STRATEGY 也可启用。
   */
  selfConsistency?: {
    samples: number;
    strategy?: SelfConsistencyStrategy;
    maxConcurrency?: number;
  };
  /**
   * 前端可强化分类器：开启后每次 run 先分类 → 注入 taskType + 记忆 label，高置信简单类
   * 路由单次 LLM 降 token，收尾用 outcome 强化。real 模式自动挂 SqliteClassifierStore 持久化。
   * 默认关（env OPENINTJ_CLASSIFIER=1 也可开）。
   */
  enableClassifier?: boolean;
  /**
   * 技能系统（Phase 1 作者能力包）：开启后每轮 query 经「目录 + 嵌入检索」两级预筛，
   * 命中的 SKILL.md 全文注入 system prompt（persona 之后、记忆参考之前），未命中零注入。
   * 默认关（env OPENINTJ_SKILLS=1 也可开）；技能目录另可用 OPENINTJ_SKILLS_DIR 追加。
   */
  enableSkills?: boolean;
  /**
   * 技能自学习闭环（Phase 2，在 enableSkills 之上）：outcome 加权 + 成功轨迹蒸馏候选技能 → 人审批。
   * real 模式挂 SqliteSkillStore 跨重启；否则 InMemorySkillStore。
   * 默认关（env OPENINTJ_SKILLS_LEARN=1 也可开，隐含开启 enableSkills）。
   */
  enableSkillLearning?: boolean;
  /**
   * 工作区根目录：read_file / write_file 被沙箱限制在此目录内（RFC-004 §8）。
   * 缺省走 env OPENINTJ_WORKSPACE_DIR，再退到 process.cwd()。
   */
  workspaceDir?: string;
  /** 是否允许 execute_command（默认关；env OPENINTJ_ENABLE_COMMANDS=1 也启用）。命令执行高危。 */
  enableCommands?: boolean;
  /** 命令白名单（按可执行文件名）；缺省走 env OPENINTJ_ALLOWED_COMMANDS（逗号分隔）。 */
  allowedCommands?: string[];
  /**
   * Phase 3.8：把 HookBus 接到 OpenTelemetry。
   * - true：attachOtelToHooks(hooks)
   * - AttachOtelOpts：透传
   * - env OPENINTJ_OTEL=1 也启用
   * 默认关闭；未注册 OTel provider 时 attach 也是 no-op。
   */
  enableOtel?: boolean | AttachOtelOpts;
  /** RFC-007：opt-in TaskPool（env OPENINTJ_TASK_POOL=1）。 */
  enableTaskPool?: boolean;
  /**
   * 遗留 TaskPool run 的启动恢复策略。默认 cancel（避免重复外部副作用）；
   * 仅显式 resume / OPENINTJ_TASK_POOL_RECOVERY=resume 时重跑未完成节点。
   */
  taskPoolRecoveryPolicy?: TaskPoolRecoveryPolicy;
  /** RFC-013：需要交互审批的风险级（或 env OPENINTJ_APPROVAL_RISKS）。 */
  approvalRisks?: ToolRisk[];
  /** RFC-013：自定义审批 broker；缺省用 PendingApprovalBroker（无人决定即拒绝）。 */
  approvalBroker?: ApprovalBroker;
  /** RFC-013：注入自定义 grant 存储；缺省 real 持久化时用 <dataDir>/grants.json（仅显式 grant 落盘）。 */
  grants?: GrantStore;
  /** RFC-014：注入自定义个人档案服务；缺省 real 持久化时用 <dataDir>/profile.json。 */
  profile?: ProfileService;
  /** RFC-014：个人档案开关；默认开，env OPENINTJ_PROFILE=0 可关。 */
  enableProfile?: boolean;
  /** RFC-015：注入自定义调度服务；缺省 real 持久化时用 <dataDir>/schedules.json。 */
  scheduler?: SchedulerService;
  /** RFC-015：调度开关；默认开，env OPENINTJ_SCHEDULER=0 可关（关时 start() 为 no-op）。 */
  enableScheduler?: boolean;
  /** RFC-015：安静时段，命中时任务顺延（如 { start: "22:00", end: "07:00" }）。 */
  schedulerQuietHours?: QuietHours;
  /** RFC-020：注入自定义身份服务；缺省 real 持久化时用 <dataDir>/identity.json。 */
  identity?: IdentityService;
  /** RFC-016：注入额外连接器；缺省叠加 env 连接器（OPENINTJ_NOTES_DIR 等）。 */
  connectors?: Connector[];
  /** RFC-016：连接器开关；默认开，env OPENINTJ_CONNECTORS=0 可关。 */
  enableConnectors?: boolean;
  /** RFC-012：压缩开关；默认开，env OPENINTJ_COMPACT=0 可关。 */
  enableCompaction?: boolean;
  /** RFC-012：压缩参数（压力阈值 tokens / 保留最近轮数）。 */
  compaction?: { pressureTokens?: number; keepRecentTurns?: number };
  /** RFC-012 §3：进程内 Subagent（delegate_task 工具）。默认关；env OPENINTJ_SUBAGENTS=1。 */
  enableSubagents?: boolean;
  /** RFC-012 §3：子会话工具风险上限（默认 read；env OPENINTJ_SUBAGENT_MAX_RISK 可调）。 */
  subagentMaxRisk?: ToolRisk;
}

const parseDesktopLlmProvider = (opts: DesktopAgentOpts): LlmProviderId => {
  if (opts.llmProvider) return opts.llmProvider;
  const raw = process.env["LLM_PROVIDER"]?.trim().toLowerCase();
  if (raw === "auto" || raw === "ollama" || raw === "hunyuan" || raw === "mock") return raw;
  return "auto";
};

const parseDesktopEmbedProvider = (opts: DesktopAgentOpts): EmbedProviderId | undefined => {
  if (opts.embedProvider) return opts.embedProvider;
  const raw = (process.env["EMBEDDING_PROVIDER"] ?? process.env["EMBED_PROVIDER"])
    ?.trim()
    .toLowerCase();
  if (
    raw === "auto" ||
    raw === "simple" ||
    raw === "ollama" ||
    raw === "xenova" ||
    raw === "mock"
  ) {
    return raw;
  }
  return undefined;
};

const resolvePersistence = (
  opts: DesktopAgentOpts,
): { mode: PersistenceMode; dataDir?: string } => {
  if (opts.persistenceMode === "memory") return { mode: "memory" };
  if (process.env["OPENINTJ_DESKTOP_NO_PERSIST"] === "1") {
    return { mode: "memory" };
  }
  if (opts.persistenceMode === "real") {
    if (!opts.dataDir) {
      throw new Error("DesktopAgent: persistenceMode='real' requires dataDir");
    }
    return { mode: "real", dataDir: opts.dataDir };
  }
  if (opts.dataDir) return { mode: "real", dataDir: opts.dataDir };
  return { mode: "memory" };
};

const resolveDormantEnabled = (opts: DesktopAgentOpts): boolean => {
  if (opts.enableDormant !== undefined) return opts.enableDormant;
  return process.env["OPENINTJ_DORMANT"] === "1";
};

const resolveRetrievalMode = (opts: DesktopAgentOpts): "vector" | "hybrid" => {
  if (opts.retrievalMode) return opts.retrievalMode;
  if (process.env["OPENINTJ_RETRIEVAL_MODE"] === "hybrid") return "hybrid";
  return "vector";
};

type DesktopDormantPersistenceResolution = { kind: "memory" } | { kind: "real"; dbPath: string };

const resolveDesktopDormantPersistence = (
  opts: DesktopAgentOpts,
  persistence: { mode: PersistenceMode; dataDir?: string },
): DesktopDormantPersistenceResolution => {
  const strategy = opts.dormantPersistence ?? "auto";
  const dbPathOverride = opts.dormantDbPath ?? process.env["OPENINTJ_DORMANT_DB_PATH"];
  if (strategy === "memory") return { kind: "memory" };
  if (strategy === "real") {
    const dbPath =
      dbPathOverride ?? (persistence.dataDir ? `${persistence.dataDir}/dormant.sqlite` : undefined);
    if (!dbPath) {
      throw new Error(
        "DesktopAgent: dormantPersistence='real' requires dataDir, dormantDbPath, or OPENINTJ_DORMANT_DB_PATH",
      );
    }
    return { kind: "real", dbPath };
  }
  if (dbPathOverride) return { kind: "real", dbPath: dbPathOverride };
  if (persistence.mode === "real" && persistence.dataDir) {
    return { kind: "real", dbPath: `${persistence.dataDir}/dormant.sqlite` };
  }
  return { kind: "memory" };
};

const resolveRateLimit = (opts: DesktopAgentOpts): RateLimitOpts | undefined => {
  if (opts.rateLimit) return opts.rateLimit;
  const qpsRaw = process.env["OPENINTJ_RATE_LIMIT_QPS"];
  if (!qpsRaw) return undefined;
  const qps = Number(qpsRaw);
  if (!Number.isFinite(qps) || qps <= 0) return undefined;
  const burstRaw = process.env["OPENINTJ_RATE_LIMIT_BURST"];
  const burst = burstRaw ? Number(burstRaw) : undefined;
  return burst && Number.isFinite(burst) && burst > 0 ? { qps, burst } : { qps };
};

/** 桌面端的 hybrid 检索辅助 —— 与 server 端实现保持等价（A1 起走 session 级增量索引）。 */
export type DesktopHybridHit = MemoryHybridHit;

export interface DesktopRetrieveHybridOpts {
  topK?: number;
  config?: Partial<HybridConfig>;
  memoryTypes?: readonly string[];
  taskTags?: readonly string[];
  queryEmbedding?: readonly number[];
}

const buildHybridRetrieve = (
  store: PersistentMemoryStore,
  hybridIndex: MemoryHybridIndex,
): ((q: string, opts?: DesktopRetrieveHybridOpts) => Promise<DesktopHybridHit[]>) => {
  return async (query, opts = {}) => {
    if (hybridIndex.size === 0) return [];
    let qVec: readonly number[] | undefined = opts.queryEmbedding;
    if (qVec === undefined) {
      const r = store.embedder.embed(query);
      qVec = r instanceof Promise ? await r : r;
    }
    return hybridIndex.search(query, qVec, {
      topK: opts.topK ?? 10,
      ...(opts.config ? { config: opts.config } : {}),
      ...(opts.memoryTypes ? { memoryTypes: opts.memoryTypes } : {}),
      ...(opts.taskTags ? { taskTags: opts.taskTags } : {}),
    });
  };
};

export interface DesktopAgent {
  hooks: HookBus;
  llm: LlmClient;
  memory: MemoryPlane;
  governance: GovernancePlane;
  toolHub: ToolHub;
  contextEngine: ContextEngine;
  tao: TaoLoop;
  control: ControlPlane;
  execution: Executor;
  persistentStore: PersistentMemoryStore;
  /** session 级增量混合检索索引（订阅 event.MEMORY_WRITTEN 自动维护）。 */
  hybridIndex: MemoryHybridIndex;
  /** 前端可强化分类器；仅 enableClassifier 时存在。 */
  classifier?: ReinforcingClassifier;
  /** 技能自学习运行时；仅 enableSkillLearning 时存在（供审批 IPC 使用）。 */
  skillLearning?: SkillLearningRuntime;
  /** 技能注册表上下文；enableSkills 或自学习开启时存在（供目录 IPC）。 */
  skillContext?: SkillContext;
  persistenceInfo: { mode: PersistenceMode; dataDir?: string };
  retrievalMode: "vector" | "hybrid";
  /** RFC-003 方向 3 蛰伏记忆学习；仅 opts.enableDormant=true 时存在。 */
  dormant?: DormantRuntime;
  /** Dormant 子系统的持久化信息（dormant 启用且挂了 adapter 时存在）。 */
  dormantPersistenceInfo?: { adapter: string; dbPath?: string };
  /** OpenTelemetry 接线状态（enableOtel 真值时存在；含 dispose 钩子）。 */
  otel?: AttachedOtel;
  modelRuntime: ModelRuntimeStatus;
  refreshModelRuntime(): Promise<ModelRuntimeStatus>;
  /** real data dir + TaskPool 开启时的启动恢复结果。 */
  taskPoolRecovery?: TaskPoolRecoverySummary;
  /** RFC-012: durable TaskPool checkpoints, newest first. */
  taskPoolCheckpoints(limit?: number): Promise<readonly StoredTaskRun[]>;
  /** RFC-012: resume a durable incomplete TaskPool run, retaining completed nodes. */
  resumeTaskRun(runId: string): Promise<TaskRunResult | undefined>;
  /** RFC-012: cancel an active or durable incomplete TaskPool run. */
  cancelTaskRun(runId: string): Promise<boolean>;
  /**
   * 工作区系统能力面（RFC-004 §8）：与 Agent 的 read_file/write_file 工具**共用同一沙箱**，
   * 供 IPC handler 直接复用，保证 UI 直接读写与 Agent 工具读写遵循完全相同的边界。
   */
  workspace: { config: ResolvedWorkspaceConfig; tools: WorkspaceTools };
  /** 基于 HybridRetriever 的检索辅助；无论 retrievalMode 都可用。 */
  retrieveHybrid(query: string, opts?: DesktopRetrieveHybridOpts): Promise<DesktopHybridHit[]>;
  commands: SlashCommandRegistry;
  handle: AgentHandle;
  /** RFC-011 prompt section registry (RFC-006 order locked). */
  prompt: PromptSectionRegistry;
  /** RFC-013 pending approvals (present when approval risks are configured). */
  approvals?: PendingApprovalBroker;
  /** RFC-013 capability grants. */
  grants: GrantStore;
  /** RFC-014 personal profile (memory control plane). */
  profile: ProfileService;
  /** RFC-015 scheduler (persistent jobs + notifications). */
  scheduler: SchedulerService;
  /** RFC-020 identity graph (proposal-confirm account linking). */
  identity: IdentityService;
  /** RFC-018 outbound notification channels. */
  channels: ChannelDispatcher;
  /** RFC-019 speech-to-text provider (undefined when unconfigured). */
  stt?: SpeechToTextProvider;
  /** RFC-019 cross-device sync bundle export/import. */
  exportSyncBundle(): SyncBundle;
  importSyncBundle(bundle: unknown): SyncApplyReport;
  /** RFC-016 connector registry (ToolHub-registered external data sources). */
  connectors: ConnectorRegistry;
  /** RFC-016: initialize async connectors (MCP) and register discovered tools. */
  initConnectors(): Promise<void>;
  /** RFC-012 compaction driver (pressure / command triggers). */
  compactor: Compactor;
  /** RFC-012 §3 subagent spawner（delegate_task 仅启用时注册）。 */
  subagents: SubagentSpawner;
  handleFor(conversationId: string, journal?: Journal): AgentHandle;
  run(query: string, opts?: DesktopRunOptions): Promise<InputStructuredTaoResult>;
  status(): {
    llm: ReturnType<LlmClient["getStatus"]> & { runtime?: ModelRuntimeStatus["llm"] };
    embed?: ModelRuntimeStatus["embed"];
    modelRuntime: ModelRuntimeStatus;
    memory: ReturnType<MemoryPlane["getStats"]>;
    governance: ReturnType<GovernancePlane["getStats"]>;
    tools: string[];
    persistence: { mode: PersistenceMode; dataDir?: string };
    retrievalMode: "vector" | "hybrid";
    classifier: { enabled: boolean; impliedByTaskPool: boolean };
    taskPool: TaskPoolActivationStatus;
    productBehavior: {
      version: string;
      enabled: boolean;
      cohort: "treatment" | "control";
    };
    dormant?: {
      enabled: true;
      passiveSize: number;
      pendingProposals: number;
      persistence?: { adapter: string; dbPath?: string };
    };
    skills: {
      enabled: boolean;
      learning: boolean;
      catalogSize: number;
      pendingProposals: number;
      activeSkills: number;
    };
    context: { used: number; max: number; ratio: number };
  };
  close(): Promise<void>;
}

export interface DesktopRunOptions {
  signal?: AbortSignal;
  history?: readonly ChatMessage[];
  llm?: LlmClient;
  /** Stable workspace/task/conversation tags written with both sides of the turn. */
  contextTags?: readonly string[];
  /** RFC-009 optional journal; missing or throwing sink never fails the loop. */
  journal?: JournalTurnContext;
  skipSlashCommand?: boolean;
}

const requestsWorkspaceArtifact = (query: string): boolean =>
  /(?:创建|生成|写入|保存|输出|落盘|create|generate|write|save).{0,40}(?:文件|文档|代码|脚本|file|document|script|[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)/iu.test(
    query,
  );

const resolveDesktopOtel = (opts: DesktopAgentOpts): AttachOtelOpts | undefined => {
  if (opts.enableOtel === true) return {};
  if (opts.enableOtel && typeof opts.enableOtel === "object") return opts.enableOtel;
  if (opts.enableOtel === false) return undefined;
  if (process.env["OPENINTJ_OTEL"] === "1") return {};
  return undefined;
};

/**
 * RFC-020: read-only local corpus for identity resolution, built from the
 * same env-configured paths as the connectors. Best-effort; documents never
 * leave the process.
 */
const identityCorpusDocuments = (): IdentityDocument[] => {
  const docs: IdentityDocument[] = [];
  try {
    const notesDir = process.env["OPENINTJ_NOTES_DIR"];
    if (notesDir && existsSync(notesDir)) {
      for (const name of readdirSync(notesDir).slice(0, 200)) {
        if (!/\.(md|txt)$/i.test(name)) continue;
        docs.push({
          source: "notes",
          title: `notes: ${name}`,
          text: readFileSync(join(notesDir, name), "utf8").slice(0, 16_000),
        });
      }
    }
    const mboxPath = process.env["OPENINTJ_MAIL_MBOX"];
    if (mboxPath && existsSync(mboxPath)) {
      for (const msg of parseMbox(readFileSync(mboxPath, "utf8")).slice(0, 300)) {
        docs.push({
          source: "mail",
          title: `mail: ${msg.subject || msg.from}`,
          text: `${msg.from}\n${msg.to ?? ""}\n${msg.subject}\n${msg.body}`.slice(0, 8_000),
        });
      }
    }
    const icsPath = process.env["OPENINTJ_CALENDAR_ICS"];
    if (icsPath && existsSync(icsPath)) {
      for (const ev of parseIcs(readFileSync(icsPath, "utf8")).slice(0, 300)) {
        docs.push({
          source: "calendar",
          title: `calendar: ${ev.summary}`,
          text: [ev.summary, ev.description ?? "", ev.location ?? ""].join("\n"),
        });
      }
    }
  } catch {
    // Corpus is an optional recall aid; never block identity resolution.
  }
  return docs;
};

export const assembleDesktopAgent = async (opts: DesktopAgentOpts = {}): Promise<DesktopAgent> => {
  const hooks = new HookBus();
  attachProductTraitSignals(hooks);
  const otelOpts = resolveDesktopOtel(opts);
  const otel = otelOpts ? attachOtelToHooks(hooks, otelOpts) : undefined;
  const llmProvider = parseDesktopLlmProvider(opts);
  const embedProvider = parseDesktopEmbedProvider(opts);
  const runtime = await resolveModelRuntime({
    provider: llmProvider,
    ...(embedProvider ? { embedProvider } : {}),
    hooks,
  });
  const rawLlm = runtime.llm.client;
  const rateLimit = resolveRateLimit(opts);
  const llm: LlmClient = rateLimit ? new RateLimitedLlmClient(rawLlm, rateLimit) : rawLlm;
  const persistence = resolvePersistence(opts);
  const embeddingDim = runtime.embed.dimension;
  const dormantEnabled = resolveDormantEnabled(opts);
  let dormantAdapter: DormantPersistenceAdapter | undefined;
  let dormantPersistenceInfo: { adapter: string; dbPath?: string } | undefined;
  if (dormantEnabled) {
    if (opts.dormantOpts?.adapter) {
      dormantAdapter = opts.dormantOpts.adapter;
      dormantPersistenceInfo = { adapter: opts.dormantOpts.adapter.name };
    } else {
      const dormantPersistence = resolveDesktopDormantPersistence(opts, persistence);
      if (dormantPersistence.kind === "real") {
        dormantAdapter = await createSqliteDormantStore({ dbPath: dormantPersistence.dbPath });
        dormantPersistenceInfo = {
          adapter: dormantAdapter.name,
          dbPath: dormantPersistence.dbPath,
        };
      }
    }
  }
  // utility 蒸馏 worker（opt-in，RFC-004 §2）：OPENINTJ_DORMANT_WORKER=1 时把 CPU 密集的
  // n-gram 挖掘下放 worker 线程，避免 mine() 阻塞 main 事件循环 / 卡 UI。失败自动回退内联。
  const useMineWorker = opts.dormantMineWorker ?? process.env["OPENINTJ_DORMANT_WORKER"] === "1";
  const dormant = dormantEnabled
    ? new DormantRuntime({
        eventIdPrefix: "desktop",
        // 默认给磁盘事件一个 LRU 上限，防 dormant_events 无限增长；显式 dormantOpts 可覆盖。
        maxDiskEvents: 50_000,
        ...(useMineWorker ? { mineRunner: (e, o) => runMineInWorker(e, o) } : {}),
        ...(opts.dormantOpts ?? {}),
        ...(dormantAdapter ? { adapter: dormantAdapter } : {}),
      })
    : undefined;
  if (dormant) await dormant.hydrate();
  const retrievalMode = resolveRetrievalMode(opts);

  const approvalSetup = createApprovalSetup({
    ...(opts.approvalRisks ? { requiredRisks: opts.approvalRisks } : {}),
    ...(opts.approvalBroker ? { broker: opts.approvalBroker } : {}),
  });
  // RFC-013: explicit grants persist to <dataDir>/grants.json in real mode;
  // approval-derived grants stay session-only (enforced by JsonGrantStore).
  const grantStore =
    opts.grants ??
    (persistence.mode === "real" && persistence.dataDir
      ? new JsonGrantStore(`${persistence.dataDir}/grants.json`)
      : undefined);
  const governance = new GovernancePlane({
    hooks,
    approval: {
      requiredRisks: approvalSetup.requiredRisks,
      ...(approvalSetup.broker ? { broker: approvalSetup.broker } : {}),
      ...(grantStore ? { grants: grantStore } : {}),
    },
  });
  if (persistence.mode === "real" && persistence.dataDir) {
    await validateEmbeddingFingerprintForDataDir(
      persistence.dataDir,
      runtime.embeddingFingerprint,
      undefined,
      { hooks },
    );
  }
  const persistentStore = await createPersistentMemoryStore({
    ...(persistence.dataDir ? { dataDir: persistence.dataDir } : {}),
    mode: persistence.mode,
    embeddingDim,
    embedder: runtime.embed.embedder,
    storeConfig: { embeddingDim },
    hooks,
  });

  const memory = new MemoryPlane({ hooks });
  const rememberTurn = async (
    userText: string,
    assistantText: string,
    tags: readonly string[],
  ): Promise<void> => {
    try {
      await memory.recordUserInputAsync(userText, tags);
      await memory.recordAssistantOutputAsync(assistantText, tags);
    } catch (error) {
      console.warn("[OpenINTJ desktop] memory persist failed; continuing without it", error);
    }
  };
  Object.defineProperty(memory, "store", {
    value: persistentStore,
    writable: false,
  });
  Object.defineProperty(memory.retriever, "store", {
    value: persistentStore,
    writable: false,
  });

  // session 级共享 HybridRetriever：开局 seed + 订阅 change-feed 增量维护。
  const hybridIndex = new MemoryHybridIndex();
  hybridIndex.seed(persistentStore.all);
  hybridIndex.subscribe(hooks);

  const control = new ControlPlane();
  // A1.3 opt-in：OPENINTJ_LOOP_HYBRID=1 时主循环检索改走 session 级增量 HybridRetriever。
  const loopHybrid = process.env["OPENINTJ_LOOP_HYBRID"] === "1";
  const candidateRetrieve = loopHybrid
    ? async (query: string, ro: { topK?: number; taskType?: TaskTypeType }) => {
        const e = persistentStore.embedder.embed(query);
        const qVec = e instanceof Promise ? await e : e;
        const hits = hybridIndex.search(query, qVec, { topK: ro.topK ?? 6 });
        return fragmentsToRanked(
          persistentStore,
          hits.map((h) => ({ id: h.doc.id, score: h.score })),
          ro.taskType ? { taskType: ro.taskType } : {},
        );
      }
    : undefined;
  const contextEngine = new ContextEngine({
    store: persistentStore,
    hooks,
    ...(candidateRetrieve ? { candidateRetrieve } : {}),
  });
  // gate：每次工具调用前跑治理平面（策略黑名单 + 工具配额），拒绝即 success:false + 审计（RFC-004 §8）。
  const toolHub = new ToolHub({ hooks, gate: createToolCallGate(governance) });
  const noop = () => ({ note: "[mock]" });
  // search 工具优先级：外部 Web Search（Tavily/Brave，provider 中立）> 混元内建联网搜索（仅旧平台有效）> 占位。
  // 旧混元平台搜索已随平台下线、TokenHub 改 Responses API，因此 TokenHub 用户应配 OPENINTJ_SEARCH_API_KEY 走外部搜索。
  const webSearchCfg = resolveWebSearchConfig();
  const searchHandler = webSearchCfg
    ? createWebSearchTool(webSearchCfg)
    : runtime.status.llm.provider === "hunyuan"
      ? createHunyuanSearchTool(HunyuanClient.fromEnv())
      : noop;
  // 真实工作区工具：read_file / write_file 沙箱限定在 workspace 根内，execute_command 默认禁用。
  const wsConfig = resolveWorkspaceConfig(opts, process.cwd());
  const wsTools = createWorkspaceTools(wsConfig);
  toolHub.registerBuiltinTools({
    readFile: wsTools.readFile,
    writeFile: wsTools.writeFile,
    executeCommand: wsTools.executeCommand,
    search: searchHandler,
  });
  if (!isCommandExecutionAvailable(wsConfig)) {
    toolHub.unregister("execute_command");
  }

  // RFC-016: connectors register their tools into the same ToolHub, so every
  // call keeps flowing through the governance gate.
  const connectorsEnabled = opts.enableConnectors ?? process.env["OPENINTJ_CONNECTORS"] !== "0";
  const connectorRegistry = new ConnectorRegistry();
  if (connectorsEnabled) {
    for (const connector of [...(opts.connectors ?? []), ...connectorsFromEnv()]) {
      connectorRegistry.register(connector);
    }
  }
  connectorRegistry.attachHub(toolHub);
  const execution = new Executor({ toolHub, hooks, registerBuiltins: false });

  const react = new ReactStateMachine({
    config: DEFAULT_REACT_CONFIG,
    hooks,
    llm,
    toolRunner: (
      name: string,
      params: Record<string, unknown>,
      callOpts?: { traceId?: string; timeoutMs?: number; signal?: AbortSignal },
    ) => toolHub.call(name, params, callOpts ?? {}),
  });
  const baseSystemPrompt =
    opts.systemPrompt ??
    [
      DEFAULT_AGENT_SYSTEM_PROMPT,
      `当前工作区根目录是 ${wsConfig.root}。`,
      "当用户要求创建、生成、保存或修改文件时，必须调用 write_file 将完整产物写入工作区；",
      "生成封面、海报或图片时直接 write_file 写入 SVG，不要调用 python/PIL 或 execute_command。",
      "不得只在回答中展示代码或声称文件已创建。写入后在最终回答中明确列出相对路径。",
    ].join("\n");
  const personaEnabled = resolvePersonaInjection(opts);
  const productBehaviorEnabled = resolveProductBehaviorEnabled(opts.enableProductBehavior);
  const inputStructuringConfig = resolveInputStructuringConfig(
    productBehaviorEnabled ? opts.inputStructuring : "off",
  );
  const taskPoolEnabled = resolveTaskPoolEnabled(opts.enableTaskPool);
  const classifierConfigured = opts.enableClassifier ?? process.env["OPENINTJ_CLASSIFIER"] === "1";
  const enableClassifier = taskPoolEnabled || classifierConfigured;
  const taskStore =
    taskPoolEnabled && persistence.mode === "real" && persistence.dataDir
      ? new SqliteTaskStore(`${persistence.dataDir}/taskpool.sqlite`)
      : undefined;
  await taskStore?.init();
  const taskPool = taskPoolEnabled
    ? new TaskPool({ hooks, ...(taskStore ? { store: taskStore } : {}) })
    : undefined;
  const taskPoolRecoveryPolicy = resolveTaskPoolRecoveryPolicy(opts.taskPoolRecoveryPolicy);

  // 技能系统（opt-in）：OPENINTJ_SKILLS=1 时装配，复用 store embedder，命中才注入能力包全文。
  // 自学习（Phase 2）：OPENINTJ_SKILLS_LEARN=1 隐含开启注入 + outcome 加权 + 蒸馏/审批闭环。
  const enableSkillLearning =
    opts.enableSkillLearning ?? process.env["OPENINTJ_SKILLS_LEARN"] === "1";
  const enableSkills =
    (opts.enableSkills ?? process.env["OPENINTJ_SKILLS"] === "1") || enableSkillLearning;

  let skillLearning: SkillLearningRuntime | undefined;
  if (enableSkillLearning) {
    const skillStore: SkillStore =
      persistence.mode === "real" && persistence.dataDir
        ? await createSqliteSkillStore({ dbPath: `${persistence.dataDir}/skills.sqlite` })
        : new InMemorySkillStore();
    const skillHalfLife = resolveSkillWeightHalfLifeSec();
    skillLearning = new SkillLearningRuntime({
      store: skillStore,
      hooks,
      ...(skillHalfLife ? { weightHalfLifeSec: skillHalfLife } : {}),
      llmDistill: createLlmSkillDistiller({
        generate: (prompt) => llm.chat([{ role: "user" as const, content: prompt }]),
      }),
      // skillContext 在下方构建，approve/revoke 仅装配完成后触发，故此处前向引用安全。
      onSkillsChanged: async () => {
        await skillContext?.reload();
      },
    });
    await skillLearning.hydrate();
  }

  const skillContext: SkillContext | undefined = enableSkills
    ? await assembleSkillContext({
        embedder: persistentStore.embedder,
        hooks,
        onSelected: (query, taskType, ids, tools) => {
          toolHub.setRunAllowlist(tools);
          skillLearning?.noteSelected(query, taskType, ids);
        },
        ...(skillLearning
          ? {
              extraSources: [
                new DbSkillSource({ approvedSkills: () => skillLearning.listApproved() }),
              ],
              weightFor: (id: string) => skillLearning.weightFor(id),
            }
          : {}),
      })
    : undefined;

  // RFC-014: the profile is the editable control plane above Dormant persona.
  // When enabled it owns the user-persona section (importing dormant fields
  // first); when disabled, fall back to the raw persona prompt.
  const profileEnabled = opts.enableProfile ?? process.env["OPENINTJ_PROFILE"] !== "0";
  const profile =
    opts.profile ??
    new ProfileService(
      persistence.mode === "real" && persistence.dataDir
        ? new JsonProfileStore(`${persistence.dataDir}/profile.json`)
        : new InMemoryProfileStore(),
    );

  // RFC-012: compaction driver (pressure check per run + /compact command).
  const compactionEnabled = opts.enableCompaction ?? process.env["OPENINTJ_COMPACT"] !== "0";
  const compactor = createCompactor({
    llm,
    ...(opts.compaction?.pressureTokens ? { pressureTokens: opts.compaction.pressureTokens } : {}),
    ...(opts.compaction?.keepRecentTurns
      ? { keepRecentTurns: opts.compaction.keepRecentTurns }
      : {}),
  });

  // RFC-015: scheduler (default runner attached at the end of assemble;
  // registerIpcHandlers may replace it with a workbench-journal-aware one).
  const schedulerEnabled = opts.enableScheduler ?? process.env["OPENINTJ_SCHEDULER"] !== "0";
  const scheduler =
    opts.scheduler ??
    new SchedulerService(
      persistence.mode === "real" && persistence.dataDir
        ? new JsonScheduleStore(`${persistence.dataDir}/schedules.json`)
        : new InMemoryScheduleStore(),
      {
        enabled: schedulerEnabled,
        ...(opts.schedulerQuietHours ? { quietHours: opts.schedulerQuietHours } : {}),
      },
    );

  // RFC-020: identity graph. Gravatar covers the public email surface; the
  // local corpus resolver links phone/wechat/handles from the user's own
  // data. The primary person's email/name mirror into the profile identity
  // section (provenance "identity-graph"); hand-written profile entries are
  // never overwritten — the graph yields to them.
  const applyIdentityToProfile = (persons: readonly IdentityPerson[]): void => {
    const derived = derivePrimaryIdentity(persons);
    for (const key of ["email", "name"] as const) {
      const existing = profile.get("identity", key);
      const value = derived[key];
      if (value) {
        if (!existing || existing.provenance === "identity-graph") {
          profile.upsert({
            section: "identity",
            key,
            value,
            source: "import",
            provenance: "identity-graph",
          });
        }
      } else if (existing?.provenance === "identity-graph") {
        profile.remove(existing.id);
      }
    }
  };
  const identity =
    opts.identity ??
    new IdentityService({
      store:
        persistence.mode === "real" && persistence.dataDir
          ? new JsonIdentityStore(`${persistence.dataDir}/identity.json`)
          : new InMemoryIdentityStore(),
      resolvers: [
        createGravatarResolver(),
        createLocalCorpusResolver({ documents: identityCorpusDocuments }),
      ],
      onGraphChanged: applyIdentityToProfile,
    });
  // Mirror the persisted graph once at startup (mutations re-fire the hook).
  if (!opts.identity) applyIdentityToProfile(identity.listPersons());

  // RFC-018: outbound channels; scheduler lifecycle events fan out to them.
  const channels = channelsFromEnv();
  const stt = sttFromEnv();
  scheduler.on((event) => {
    void channels.dispatch(channelEventFromScheduler(event));
  });

  const prompt = createPromptSectionRegistry(
    createDefaultPromptSections({
      productBehaviorEnabled,
      persona: () => {
        if (!profileEnabled) {
          return personaEnabled ? (dormant?.personaSystemPrompt() ?? "") : "";
        }
        if (personaEnabled && dormant) profile.importPersona(dormant.snapshot());
        return profile.renderPromptSection();
      },
      skills: async ({ query, taskType, traceId }) => {
        if (!skillContext) return "";
        return skillContext.render(query, {
          ...(taskType ? { taskType } : {}),
          ...(traceId ? { traceId } : {}),
        });
      },
    }),
  );

  const tao = new TaoLoop({
    config: {
      ...DEFAULT_TAO_CONFIG,
      maxTaoIterations: opts.maxTaoIterations ?? 2,
    },
    hooks,
    react,
    availableTools: () => toolHub.list(),
    systemPrompt: baseSystemPrompt,
    // 每轮注入：①已批准的钝化记忆 persona ②命中的技能包 ③检索到的 [记忆参考]。
    // personaEnabled 是 A/B 杠杆：关闭即得到无 persona 基线（RFC-003 §3.6 #3）。
    contextProvider: async ({ query, history, taskType, topK, traceId, planAdviceBlock }) => {
      const prefix = await prompt.render({
        query,
        history,
        ...(taskType ? { taskType } : {}),
        ...(traceId ? { traceId } : {}),
        ...(planAdviceBlock ? { planAdviceBlock } : {}),
      });
      const stacked = prefix ? `${baseSystemPrompt}\n\n${prefix}` : baseSystemPrompt;
      const snap = await contextEngine.build({
        query,
        history,
        taskType,
        systemPrompt: stacked,
        topK: topK ?? 6,
        ...(traceId ? { traceId } : {}),
      });
      return snap.systemPrompt;
    },
  });

  const runTao: TaoLoop["run"] = (query, runOpts) =>
    toolHub.runInAllowlistScope(() => tao.run(query, runOpts), { inherit: true });

  const retrieveHybrid = buildHybridRetrieve(persistentStore, hybridIndex);
  const selfConsistency = resolveSelfConsistency(opts.selfConsistency);

  // 前端可强化分类器（opt-in）。real 模式挂 SqliteClassifierStore 让强化跨重启。
  let classifier: ReinforcingClassifier | undefined;
  let classifierStore: Awaited<ReturnType<typeof createSqliteClassifierStore>> | undefined;
  if (enableClassifier) {
    if (persistence.mode === "real" && persistence.dataDir) {
      classifierStore = await createSqliteClassifierStore({
        dbPath: `${persistence.dataDir}/classifier.sqlite`,
      });
    }
    classifier = new ReinforcingClassifier({
      embedder: persistentStore.embedder,
      ...(classifierStore ? { store: classifierStore } : {}),
    });
    await classifier.hydrate();
    if (classifier.size === 0) await classifier.addSeeds(DEFAULT_SEEDS);
  }

  const runTaskPoolNode = async (
    node: TaskGraphNode,
    ctx: TaskWorkerContext,
    extraOpts: Parameters<TaoLoop["run"]>[1] = {},
  ): Promise<Awaited<ReturnType<TaoLoop["run"]>>> => {
    const stepQuery = `[${node.description}]（步骤 ${node.id}/${node.action}）\n${ctx.goalInput}`;
    return runTao(stepQuery, {
      traceId: ctx.traceId,
      signal: ctx.signal,
      ...extraOpts,
    });
  };

  const taskPoolRecovery =
    taskPool && taskStore
      ? await taskPool.recoverIncomplete(
          (node, ctx) => runTaskPoolNode(node, ctx),
          taskPoolRecoveryPolicy,
        )
      : undefined;

  const commands = createSlashCommandRegistry(createBuiltinSlashCommands());
  const fallbackJournal = createInMemoryJournal();
  const handles = new Map<string, AgentHandle>();
  const agent: DesktopAgent = {
    hooks,
    llm,
    memory,
    governance,
    toolHub,
    contextEngine,
    tao,
    control,
    execution,
    persistentStore,
    hybridIndex,
    ...(classifier ? { classifier } : {}),
    ...(skillLearning ? { skillLearning } : {}),
    ...(skillContext ? { skillContext } : {}),
    persistenceInfo: persistence,
    retrievalMode,
    ...(dormant ? { dormant } : {}),
    ...(dormantPersistenceInfo ? { dormantPersistenceInfo } : {}),
    ...(otel ? { otel } : {}),
    modelRuntime: runtime.status,
    refreshModelRuntime: () => runtime.refreshHealth(),
    ...(taskPoolRecovery ? { taskPoolRecovery } : {}),
    async taskPoolCheckpoints(limit?: number) {
      return taskPool ? taskPool.listCheckpoints(limit) : [];
    },
    async resumeTaskRun(runId: string) {
      if (!taskPool) return undefined;
      const run = await taskPool.resumeRun(runId, (node, ctx) => runTaskPoolNode(node, ctx));
      return run?.result;
    },
    async cancelTaskRun(runId: string) {
      return taskPool ? taskPool.cancelRun(runId) : false;
    },
    workspace: { config: wsConfig, tools: wsTools },
    retrieveHybrid,
    commands,
    prompt,
    profile,
    scheduler,
    identity,
    channels,
    ...(stt ? { stt } : {}),
    exportSyncBundle() {
      return buildSyncBundle({ profile, scheduler, identity });
    },
    importSyncBundle(bundle: unknown) {
      return applySyncBundle(bundle as SyncBundle, { profile, scheduler, identity });
    },
    connectors: connectorRegistry,
    async initConnectors() {
      await connectorRegistry.initAll();
    },
    compactor,
    subagents: undefined as unknown as SubagentSpawner,
    grants: governance.grants,
    ...(approvalSetup.pending ? { approvals: approvalSetup.pending } : {}),
    handle: undefined as unknown as AgentHandle,
    handleFor(conversationId, journal) {
      const existing = handles.get(conversationId);
      if (existing) return existing;
      const created = createAgentHandle({
        id: conversationId,
        journal: journal ?? fallbackJournal,
        hooks,
        runTurn: (query, runOpts) => agent.run(query, { ...runOpts, skipSlashCommand: true }),
      });
      handles.set(conversationId, created);
      return created;
    },
    async run(query: string, runOpts: DesktopRunOptions = {}) {
      const conversationId = runOpts.journal?.conversationId ?? "desktop";
      const sink = runOpts.journal?.sink;
      const journal =
        sink && "list" in sink && "listActive" in sink ? (sink as Journal) : fallbackJournal;
      const handle = agent.handleFor(conversationId, journal);
      if (!runOpts.skipSlashCommand) {
        const invoked = await invokeSlashCommand(commands, query, {
          journal,
          handle,
          conversationId,
          ...(compactionEnabled
            ? {
                compact: () => runCompactCommand(compactor, journal, conversationId, runOpts.llm),
              }
            : {}),
        });
        if (invoked) return slashCommandResultToTao(invoked.result);
        return handle.run(query, runOpts);
      }
      await hooks.emit("event.PRODUCT_BEHAVIOR", {
        version: PRODUCT_BEHAVIOR_VERSION,
        enabled: productBehaviorEnabled,
      });
      if (dormant) dormant.record(query, "user", { stage: "run.input" });
      if (requestsImageGeneration(query)) {
        const image = await fulfillImageGenerationTask({
          query,
          workspaceRoot: wsConfig.root,
          taskId: taskIdFromContextTags(runOpts.contextTags),
          llm: runOpts.llm ?? llm,
          ...(runOpts.signal ? { signal: runOpts.signal } : {}),
        });
        const result: InputStructuredTaoResult = {
          traceId: randomUUID(),
          status: "completed",
          finalAnswer: image.finalAnswer,
          iterations: 0,
          reactTotalSteps: 0,
          totalTokensSpent: 0,
          durationMs: 0,
          trajectory: [
            {
              timestamp: Date.now() / 1000,
              state: { type: "final", answer: image.finalAnswer },
              durationMs: 0,
            },
          ],
          taskType: TaskType.CODE_GENERATION,
          shaderMode: getShaderForTask(TaskType.CODE_GENERATION),
          metrics: image.fallback ? { imageGenerationFallback: 1 } : { imageGenerated: 1 },
        };
        await rememberTurn(query, result.finalAnswer, [...(runOpts.contextTags ?? [])]);
        if (dormant)
          dormant.record(result.finalAnswer, "agent", {
            stage: "run.output",
            iterations: 0,
          });
        try {
          await persistentStore.awaitPendingWrites();
        } catch (error) {
          console.warn("[OpenINTJ desktop] memory flush failed; continuing without it", error);
        }
        return result;
      }
      const scopedMemories =
        runOpts.contextTags && runOpts.contextTags.length > 0
          ? memory.store.all.filter((fragment) =>
              runOpts.contextTags?.every((tag) => fragment.taskTags.includes(tag)),
            )
          : memory.store.all;
      const artifactRequest = requestsWorkspaceArtifact(query);
      const prelude = await prepareRunPrelude(
        {
          hooks,
          llm,
          control,
          // Artifact requests keep the historical preflight exemption.
          productBehaviorEnabled: productBehaviorEnabled && !artifactRequest,
          inputStructuringConfig,
          ...(classifier ? { classifier } : {}),
          skipClassifierAndPlan: artifactRequest,
          ...(artifactRequest ? { forcedTaskType: TaskType.CODE_GENERATION } : {}),
        },
        {
          query,
          memories: scopedMemories,
          ...(runOpts.history ? { history: runOpts.history } : {}),
          ...(runOpts.signal ? { signal: runOpts.signal } : {}),
          ...(runOpts.llm ? { llm: runOpts.llm } : {}),
        },
      );
      const { preflight, inputStructure, executionQuery, cls, route, routedTaskType } = prelude;
      const structured = prelude.structured;
      const workQuery = prelude.workQuery;
      const planAdviceBlock = prelude.planAdviceBlock;
      const journalCtx = buildJournalTurnContext({
        ...(runOpts.journal ? { provided: runOpts.journal } : {}),
        query,
        ...(structured ? { structured } : {}),
      });
      // RFC-012: pressure compaction — fold old turns into a summary when the
      // history crosses the threshold; fail-soft, never blocks the run.
      let effectiveHistory = runOpts.history;
      let didCompact = false;
      let overflowRecovered = false;
      if (compactionEnabled) {
        const compacted = await compactor.compactIfNeeded({
          journal,
          conversationId,
          trigger: "pressure",
          ...(runOpts.llm ? { llm: runOpts.llm } : {}),
        });
        if (compacted.compacted) {
          effectiveHistory = compacted.history;
          didCompact = true;
        }
      }
      const taoOpts = (
        traceId?: string,
        signal: AbortSignal | undefined = runOpts.signal,
        attachJournal = false,
      ) => ({
        taskType: routedTaskType,
        ...(route?.single || (!cls && routedTaskType === TaskType.QUICK_RESPONSE)
          ? { enableReact: false }
          : {}),
        ...(route ? { topK: route.topK } : {}),
        ...(traceId ? { traceId } : {}),
        ...(signal ? { signal } : {}),
        ...(effectiveHistory ? { history: effectiveHistory } : {}),
        ...(runOpts.llm ? { llm: runOpts.llm } : {}),
        ...(planAdviceBlock ? { planAdviceBlock } : {}),
        ...(attachJournal && journalCtx ? { journal: journalCtx } : {}),
      });
      // 先跑（contextProvider 会检索此前已落盘的记忆），再记录本轮 → 避免检索命中当前输入本身。
      let result: InputStructuredTaoResult;
      if (structured?.action === "clarify") {
        result = inputClarificationResult(structured);
      } else if (preflight) {
        const taskType = detectTaskType(query);
        result = {
          traceId: randomUUID(),
          status: "completed",
          finalAnswer: preflight.answer,
          iterations: 0,
          reactTotalSteps: 0,
          totalTokensSpent: 0,
          durationMs: 0,
          trajectory: [
            {
              timestamp: Date.now() / 1000,
              state: { type: "final", answer: preflight.answer },
              durationMs: 0,
            },
          ],
          taskType,
          shaderMode: getShaderForTask(taskType),
          metrics: { productBehaviorPreflight: 1 },
        };
      } else {
        const orchestrationMode = resolveOrchestrationMode(
          Boolean(taskPool && cls && shouldUseTaskPool(taskPoolEnabled, cls.label)),
          Boolean(selfConsistency),
        );
        if (orchestrationMode === "taskpool" && taskPool && cls) {
          const { plan } = control.processInput(workQuery, cls.label);
          const graph = planGraphToTaskGraph(plan);
          const poolResult = await taskPool.submitRun(graph, (node, ctx) =>
            runTaskPoolNode(node, ctx, taoOpts(ctx.traceId, ctx.signal)),
          );
          result = synthesizeTaskPoolAnswer(poolResult, workQuery);
        } else if (orchestrationMode === "self-consistency" && selfConsistency) {
          // 方向一/二：并行多采样 + 投票。forkJoin 会发 forkjoin.* 事件 → OTel span/metric。
          const { fulfilled } = await forkJoin(
            Array.from({ length: selfConsistency.samples }, (_, i) => i),
            (i) => runTao(workQuery, taoOpts(`${randomUUID()}-sc${i}`)),
            {
              hooks,
              group: "self-consistency",
              minSuccess: 1,
              ...(selfConsistency.maxConcurrency
                ? { concurrency: selfConsistency.maxConcurrency }
                : {}),
            },
          );
          result = selectConsistentAnswer(fulfilled, selfConsistency.strategy) ?? fulfilled[0]!;
        } else {
          result = await runTao(workQuery, {
            ...taoOpts(undefined, runOpts.signal, true),
            ...(compactionEnabled
              ? {
                  onContextOverflow: async () => {
                    const out = await compactor.compactIfNeeded({
                      journal,
                      conversationId,
                      trigger: "overflow",
                      maxSummaryInputChars: 16_000,
                      ...(runOpts.llm ? { llm: runOpts.llm } : {}),
                    });
                    if (!out.compacted) return undefined;
                    didCompact = true;
                    overflowRecovered = true;
                    return out.history;
                  },
                }
              : {}),
          });
        }
      }
      if (prelude.usedPlanAdvice) result.metrics.planAdvice = 1;
      if (didCompact) result.metrics.compacted = 1;
      if (overflowRecovered) result.metrics.overflowRecovered = 1;
      if (inputStructure?.action === "proceed") {
        result.inputStructure = inputStructure;
        result.totalTokensSpent += inputStructure.tokensSpent;
        result.durationMs += inputStructure.durationMs;
        result.metrics.inputStructuringTokens = inputStructure.tokensSpent;
        if (inputStructure.mode === "structured") result.metrics.inputStructured = 1;
        if (inputStructure.mode === "fallback") result.metrics.inputStructuringFallback = 1;
      }
      if (productBehaviorEnabled && structured?.action !== "clarify") {
        const enforced = await enforceProductBehaviorAnswer({
          query,
          draft: result.finalAnswer,
          searchEvidence: resolveSearchEvidenceStatus(result.trajectory),
          revise: async (instruction) =>
            (runOpts.llm ?? llm).chat(
              [
                {
                  role: "system",
                  content: "你是最终答案编辑器。严格完成用户要求，只输出修订后的答案。",
                },
                { role: "user", content: instruction },
              ],
              { temperature: 0, maxTokens: 768 },
            ),
        });
        result.finalAnswer = enforced.answer;
      }
      // 把 search 工具命中的联网来源追加到答案末尾 → 随记忆/dormant 一起入库。
      result.finalAnswer = appendSourcesFooter(result.finalAnswer, result.trajectory);
      const labelTags = [...(cls ? [cls.label] : []), ...(runOpts.contextTags ?? [])];
      await rememberTurn(query, result.finalAnswer, labelTags);
      if (classifier && cls) {
        await classifier.reinforce(executionQuery, cls.label, {
          signal: outcomeSignal(result.status),
        });
      }
      if (skillLearning) {
        const toolsUsed = [
          ...new Set(
            result.trajectory
              .map((e) => {
                const st = (e as { state?: { toolResult?: { toolName?: unknown } } }).state;
                const name = st?.toolResult?.toolName;
                return typeof name === "string" ? name : undefined;
              })
              .filter((n): n is string => Boolean(n)),
          ),
        ];
        skillLearning.recordOutcome(query, cls?.label, result.status, {
          finalAnswer: result.finalAnswer,
          toolsUsed,
        });
      }
      if (dormant)
        dormant.record(result.finalAnswer, "agent", {
          stage: "run.output",
          iterations: result.iterations,
        });
      try {
        await persistentStore.awaitPendingWrites();
      } catch (error) {
        console.warn("[OpenINTJ desktop] memory flush failed; continuing without it", error);
      }
      return result;
    },
    status() {
      const llmSt = llm.getStatus();
      return {
        llm: { ...llmSt, runtime: runtime.status.llm },
        embed: runtime.status.embed,
        modelRuntime: runtime.status,
        memory: memory.getStats(),
        governance: governance.getStats(),
        tools: toolHub.list().map((t) => t.name),
        persistence,
        retrievalMode,
        classifier: {
          enabled: Boolean(classifier),
          impliedByTaskPool: taskPoolEnabled && !classifierConfigured,
        },
        taskPool: resolveTaskPoolActivation(taskPoolEnabled, Boolean(classifier), {
          persistence: taskStore ? "sqlite" : "none",
          recovery: taskStore ? taskPoolRecoveryPolicy : "unsupported",
          ...(taskPoolRecovery ? { recoverySummary: taskPoolRecovery } : {}),
        }),
        productBehavior: {
          version: PRODUCT_BEHAVIOR_VERSION,
          enabled: productBehaviorEnabled,
          cohort: productBehaviorEnabled ? "treatment" : "control",
        },
        ...(dormant
          ? {
              dormant: {
                enabled: true as const,
                passiveSize: dormant.passiveSize(),
                pendingProposals: dormant.listProposals("pending").length,
                ...(dormantPersistenceInfo ? { persistence: dormantPersistenceInfo } : {}),
              },
            }
          : {}),
        skills: {
          enabled: Boolean(skillContext),
          learning: Boolean(skillLearning),
          catalogSize: skillContext?.size ?? 0,
          pendingProposals: skillLearning?.listProposals("pending").length ?? 0,
          activeSkills: skillLearning?.listApproved().length ?? 0,
        },
        context: contextEngine.usage(),
      };
    },
    async close() {
      scheduler.stop();
      await connectorRegistry.close();
      hybridIndex.dispose();
      if (classifierStore) await classifierStore.close();
      if (skillLearning) await skillLearning.close();
      if (otel) otel.dispose();
      if (dormant) await dormant.close();
      await taskStore?.close();
      await persistentStore.close();
    },
  };
  agent.handle = agent.handleFor("desktop", fallbackJournal);
  // RFC-015: scheduled jobs run as ordinary turns through per-conversation handles.
  scheduler.attachRunner(async (job) => {
    const handle = agent.handleFor(job.conversationId ?? "scheduler", fallbackJournal);
    const result = await handle.run(job.prompt);
    if (result.status !== "completed") {
      throw new Error(`scheduled run ended with status ${result.status}`);
    }
  });
  // RFC-012 §3: subagent spawner; delegate tool registered only when enabled.
  // Children run under a tool ceiling (default read-risk only): the cap is
  // resolved per spawn so late-registered connector tools are included, and
  // enforced through a fresh ToolHub allowlist scope (see setRunCap).
  const subagentMaxRisk: ToolRisk =
    opts.subagentMaxRisk ??
    (TOOL_RISK_ORDER.includes(process.env["OPENINTJ_SUBAGENT_MAX_RISK"] as ToolRisk)
      ? (process.env["OPENINTJ_SUBAGENT_MAX_RISK"] as ToolRisk)
      : "read");
  const subagents = createSubagentSpawner({
    tools: () =>
      toolHub
        .list()
        .filter(
          (d) =>
            d.name !== "delegate_task" &&
            riskAtLeast(
              subagentMaxRisk,
              classifyToolRisk({
                tool: d.name,
                permissions: d.permissions,
                idempotent: d.idempotent,
              }),
            ),
        )
        .map((d) => d.name),
    runChild: async (childConversationId, task, signal, tools) => {
      const run = async () => {
        const result = await agent
          .handleFor(childConversationId, fallbackJournal)
          .run(task, { signal });
        return { status: result.status, finalAnswer: result.finalAnswer };
      };
      if (!tools) return run();
      return toolHub.runInAllowlistScope(async () => {
        toolHub.setRunCap(tools);
        return run();
      });
    },
  });
  agent.subagents = subagents;
  if (opts.enableSubagents ?? process.env["OPENINTJ_SUBAGENTS"] === "1") {
    const delegate = createDelegateTool(subagents);
    toolHub.register(delegate.descriptor, delegate.handler);
  }
  return agent;
};
