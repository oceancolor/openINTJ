import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import {
  type ChannelDispatcher,
  channelEventFromScheduler,
  channelsFromEnv,
} from "@openintj/channels";
import { DEFAULT_SEEDS, ReinforcingClassifier, outcomeSignal } from "@openintj/classifier";
import { forkJoin } from "@openintj/concurrency";
import {
  type Connector,
  ConnectorRegistry,
  connectorsFromEnv,
  parseIcs,
  parseMbox,
} from "@openintj/connectors";
/**
 * Agent 装配工厂：组合 core + 4 plane + LLM 适配器为可执行的 Agent。
 * 这是 CLI / Server / Desktop 共用的核心装配逻辑。
 */
import {
  type AgentHandle,
  type ChatMessage,
  DEFAULT_REACT_CONFIG,
  DEFAULT_TAO_CONFIG,
  type EmbeddingProvider,
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
  finalizeJournalTurn,
  getShaderForTask,
  inputClarificationResult,
  invokeSlashCommand,
  resolveInputStructuringConfig,
  slashCommandResultToTao,
} from "@openintj/core";
import { DormantRuntime, type DormantRuntimeOpts } from "@openintj/dormant";
import {
  type IdentityDocument,
  type IdentityPerson,
  IdentityService,
  createGravatarResolver,
  createLocalCorpusResolver,
  derivePrimaryIdentity,
} from "@openintj/identity";
import { HunyuanClient, createHunyuanSearchTool } from "@openintj/llm-hunyuan";
import {
  type EmbedProviderId,
  type LlmProviderId,
  type ModelRuntime,
  type ModelRuntimeStatus,
  resolveLlmClientSync,
  resolveModelRuntime,
} from "@openintj/model-runtime";
import { ControlPlane } from "@openintj/plane-control";
import {
  Executor,
  ToolHub,
  createWebSearchTool,
  createWorkspaceTools,
  resolveWebSearchConfig,
} from "@openintj/plane-execution";
import {
  type ApprovalBroker,
  GovernancePlane,
  type GrantStore,
  type PendingApprovalBroker,
  TOOL_RISK_ORDER,
  type ToolRisk,
  classifyToolRisk,
  createApprovalSetup,
  createToolCallGate,
  riskAtLeast,
} from "@openintj/plane-governance";
import { ContextEngine, MemoryPlane, fragmentsToRanked } from "@openintj/plane-memory";
import { ProfileService } from "@openintj/profile";
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
import { type QuietHours, SchedulerService } from "@openintj/scheduler";
import {
  DEFAULT_AGENT_SYSTEM_PROMPT,
  PRODUCT_BEHAVIOR_VERSION,
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
  assembleSkillContext,
  createLlmSkillDistiller,
  resolveSkillWeightHalfLifeSec,
} from "@openintj/skills";
import {
  type SyncApplyReport,
  type SyncBundle,
  applySyncBundle,
  buildSyncBundle,
} from "@openintj/sync";
import {
  MemoryHybridIndex,
  TaskPool,
  type TaskPoolActivationStatus,
  planGraphToTaskGraph,
  resolveOrchestrationMode,
  resolveTaskPoolActivation,
  resolveTaskPoolEnabled,
  shouldUseTaskPool,
  synthesizeTaskPoolAnswer,
} from "@openintj/taskpool";
import { type SpeechToTextProvider, sttFromEnv } from "@openintj/voice";

export type LlmProvider = LlmProviderId;

export interface AgentOptions {
  llmProvider?: LlmProvider;
  embedProvider?: EmbedProviderId;
  systemPrompt?: string;
  /** 注入自定义工具实现（覆盖默认的真实工作区工具；测试常用）。 */
  toolHandlers?: {
    readFile?: (params: Record<string, unknown>) => unknown;
    writeFile?: (params: Record<string, unknown>) => unknown;
    executeCommand?: (params: Record<string, unknown>) => unknown;
    search?: (params: Record<string, unknown>) => unknown;
  };
  /** TAO 多轮上限（默认 1 = 单轮兼容 v2.0）。 */
  maxTaoIterations?: number;
  /** 工作区根目录：read_file / write_file 沙箱根；缺省 env OPENINTJ_WORKSPACE_DIR → process.cwd()。 */
  workspaceDir?: string;
  /** 是否允许 execute_command（默认关；env OPENINTJ_ENABLE_COMMANDS=1 也启用）。 */
  enableCommands?: boolean;
  /** 命令白名单（按可执行文件名）；缺省 env OPENINTJ_ALLOWED_COMMANDS（逗号分隔）。 */
  allowedCommands?: string[];
  selfConsistency?: {
    samples: number;
    strategy?: SelfConsistencyStrategy;
    maxConcurrency?: number;
  };
  enableClassifier?: boolean;
  enableSkills?: boolean;
  enableSkillLearning?: boolean;
  enableDormant?: boolean;
  dormantOpts?: DormantRuntimeOpts;
  enablePersona?: boolean;
  /** RFC-006 Product Behavior A/B；默认开，env OPENINTJ_PRODUCT_BEHAVIOR=0 可关。 */
  enableProductBehavior?: boolean;
  inputStructuring?: InputStructuringPolicy;
  /** RFC-007：opt-in TaskPool。 */
  enableTaskPool?: boolean;
  /** 跳过启动期 Ollama 健康探测（单测用）。 */
  syncLlm?: boolean;
  /** RFC-013：需要交互审批的风险级（或 env OPENINTJ_APPROVAL_RISKS）。 */
  approvalRisks?: ToolRisk[];
  /** RFC-013：自定义审批 broker；缺省用 PendingApprovalBroker（无人决定即拒绝）。 */
  approvalBroker?: ApprovalBroker;
  /** RFC-013: inject a custom grant store; default in-memory (CLI has no dataDir). */
  grants?: GrantStore;
  /** RFC-014：注入自定义个人档案服务；缺省内存档案。 */
  profile?: ProfileService;
  /** RFC-014：个人档案开关；默认开，env OPENINTJ_PROFILE=0 可关。 */
  enableProfile?: boolean;
  /** RFC-015：注入自定义调度服务；缺省内存调度。 */
  scheduler?: SchedulerService;
  /** RFC-015：调度开关；默认开，env OPENINTJ_SCHEDULER=0 可关（关时 start() 为 no-op）。 */
  enableScheduler?: boolean;
  /** RFC-015：安静时段，命中时任务顺延（如 { start: "22:00", end: "07:00" }）。 */
  schedulerQuietHours?: QuietHours;
  /** RFC-020: inject a custom identity service; default in-memory. */
  identity?: IdentityService;
  /** RFC-016：注入额外连接器；缺省叠加 env 连接器（OPENINTJ_NOTES_DIR 等）。 */
  connectors?: Connector[];
  /** RFC-016：连接器开关；默认开，env OPENINTJ_CONNECTORS=0 可关。 */
  enableConnectors?: boolean;
  /** RFC-012：压缩开关；默认开，env OPENINTJ_COMPACT=0 可关。 */
  enableCompaction?: boolean;
  /** RFC-012：压缩参数（压力阈值 tokens / 保留最近轮数）。 */
  compaction?: { pressureTokens?: number; keepRecentTurns?: number };
  /** RFC-012 S3: in-process subagents (delegate_task tool). Default off; env OPENINTJ_SUBAGENTS=1. */
  enableSubagents?: boolean;
  /** RFC-012 S3: subagent tool risk ceiling (default read; env OPENINTJ_SUBAGENT_MAX_RISK). */
  subagentMaxRisk?: ToolRisk;
}

export interface AssembledAgent {
  hooks: HookBus;
  llm: LlmClient;
  control: ControlPlane;
  execution: Executor;
  memory: MemoryPlane;
  governance: GovernancePlane;
  contextEngine: ContextEngine;
  hybridIndex: MemoryHybridIndex;
  classifier?: ReinforcingClassifier;
  skillLearning?: SkillLearningRuntime;
  dormant?: DormantRuntime;
  tao: TaoLoop;
  modelRuntime?: ModelRuntimeStatus;
  refreshModelRuntime?: () => Promise<ModelRuntimeStatus>;
  productBehavior: { version: string; enabled: boolean; cohort: "treatment" | "control" };
  inputStructuringConfig: ReturnType<typeof resolveInputStructuringConfig>;
  taskPoolEnabled: boolean;
  taskPoolActivation: TaskPoolActivationStatus;
  classifierStatus: { enabled: boolean; impliedByTaskPool: boolean };
  journal: Journal;
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
  /** RFC-012 S3 subagent spawner (delegate_task registered only when enabled). */
  subagents: SubagentSpawner;
  run(query: string, opts?: AgentRunOptions): Promise<InputStructuredTaoResult>;
}

export interface AgentRunOptions {
  signal?: AbortSignal;
  history?: readonly ChatMessage[];
  llm?: LlmClient;
  journal?: JournalTurnContext;
  /** Internal: skip slash-command intercept when the handle opens a turn. */
  skipSlashCommand?: boolean;
}

const buildAgentCore = (
  opts: AgentOptions,
  llm: LlmClient,
  llmProviderId: string,
  embedder?: EmbeddingProvider,
  modelRuntime?: ModelRuntime,
  hooksInput?: HookBus,
): Omit<
  AssembledAgent,
  | "run"
  | "productBehavior"
  | "commands"
  | "handle"
  | "initConnectors"
  | "subagents"
  | "exportSyncBundle"
  | "importSyncBundle"
> & {
  tao: TaoLoop;
  runTao: TaoLoop["run"];
  journal: Journal;
  selfConsistency: ReturnType<typeof resolveSelfConsistency>;
  classifier?: ReinforcingClassifier;
  skillLearning?: SkillLearningRuntime;
  dormant?: DormantRuntime;
  ensureClassifier: () => Promise<void>;
  taskPool?: TaskPool;
  taskPoolEnabled: boolean;
  taskPoolActivation: TaskPoolActivationStatus;
  classifierStatus: { enabled: boolean; impliedByTaskPool: boolean };
  productBehaviorEnabled: boolean;
  inputStructuringConfig: ReturnType<typeof resolveInputStructuringConfig>;
  compactionEnabled: boolean;
  toolHub: ToolHub;
  subagentsEnabled: boolean;
  subagentMaxRisk: ToolRisk;
} => {
  const hooks = hooksInput ?? new HookBus();
  attachProductTraitSignals(hooks);
  const approvalSetup = createApprovalSetup({
    ...(opts.approvalRisks ? { requiredRisks: opts.approvalRisks } : {}),
    ...(opts.approvalBroker ? { broker: opts.approvalBroker } : {}),
  });
  const governance = new GovernancePlane({
    hooks,
    approval: {
      requiredRisks: approvalSetup.requiredRisks,
      ...(approvalSetup.broker ? { broker: approvalSetup.broker } : {}),
      ...(opts.grants ? { grants: opts.grants } : {}),
    },
  });
  const memory = new MemoryPlane({ hooks, ...(embedder ? { embedder } : {}) });
  const control = new ControlPlane();
  const hybridIndex = new MemoryHybridIndex();
  hybridIndex.seed(memory.store.all);
  hybridIndex.subscribe(hooks);

  const loopHybrid = process.env["OPENINTJ_LOOP_HYBRID"] === "1";
  const candidateRetrieve = loopHybrid
    ? async (query: string, ro: { topK?: number; taskType?: TaskTypeType }) => {
        const e = memory.store.embedder.embed(query);
        const qVec = e instanceof Promise ? await e : e;
        const hits = hybridIndex.search(query, qVec, { topK: ro.topK ?? 6 });
        return fragmentsToRanked(
          memory.store,
          hits.map((h) => ({ id: h.doc.id, score: h.score })),
          ro.taskType ? { taskType: ro.taskType } : {},
        );
      }
    : undefined;
  const contextEngine = new ContextEngine({
    store: memory.store,
    hooks,
    ...(candidateRetrieve ? { candidateRetrieve } : {}),
  });

  const toolHub = new ToolHub({ hooks, gate: createToolCallGate(governance) });
  const handlers = opts.toolHandlers ?? {};
  const wsConfig = resolveWorkspaceConfig(opts, process.cwd());
  const wsTools = createWorkspaceTools(wsConfig);
  const defaultSearch = (params: Record<string, unknown>) => ({
    note: "[mock search]",
    query: params["query"],
    hits: [],
  });
  const webSearchCfg = resolveWebSearchConfig();
  const defaultSearchHandler = webSearchCfg
    ? createWebSearchTool(webSearchCfg)
    : llmProviderId === "hunyuan"
      ? createHunyuanSearchTool(HunyuanClient.fromEnv())
      : defaultSearch;
  toolHub.registerBuiltinTools({
    readFile: handlers.readFile ?? wsTools.readFile,
    writeFile: handlers.writeFile ?? wsTools.writeFile,
    executeCommand: handlers.executeCommand ?? wsTools.executeCommand,
    search: handlers.search ?? defaultSearchHandler,
  });
  if (!handlers.executeCommand && !isCommandExecutionAvailable(wsConfig)) {
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
  const baseSystemPrompt = opts.systemPrompt ?? DEFAULT_AGENT_SYSTEM_PROMPT;

  const enableDormant = opts.enableDormant ?? process.env["OPENINTJ_DORMANT"] === "1";
  const dormant = enableDormant
    ? new DormantRuntime({ eventIdPrefix: "cli", ...(opts.dormantOpts ?? {}) })
    : undefined;
  const personaEnabled = resolvePersonaInjection(opts);
  const productBehaviorEnabled = resolveProductBehaviorEnabled(opts.enableProductBehavior);
  const inputStructuringConfig = resolveInputStructuringConfig(
    productBehaviorEnabled ? opts.inputStructuring : "off",
  );
  const taskPoolEnabled = resolveTaskPoolEnabled(opts.enableTaskPool);
  const classifierConfigured = opts.enableClassifier ?? process.env["OPENINTJ_CLASSIFIER"] === "1";
  const enableClassifier = taskPoolEnabled || classifierConfigured;
  const taskPool = taskPoolEnabled ? new TaskPool({ hooks }) : undefined;

  const enableSkillLearning =
    opts.enableSkillLearning ?? process.env["OPENINTJ_SKILLS_LEARN"] === "1";
  const enableSkills =
    (opts.enableSkills ?? process.env["OPENINTJ_SKILLS"] === "1") || enableSkillLearning;

  const skillLearning = enableSkillLearning
    ? new SkillLearningRuntime({
        store: new InMemorySkillStore(),
        hooks,
        ...(resolveSkillWeightHalfLifeSec()
          ? { weightHalfLifeSec: resolveSkillWeightHalfLifeSec() as number }
          : {}),
        llmDistill: createLlmSkillDistiller({
          generate: (prompt) => llm.chat([{ role: "user" as const, content: prompt }]),
        }),
        onSkillsChanged: async () => {
          const ctx = skillContextP ? await skillContextP : undefined;
          await ctx?.reload();
        },
      })
    : undefined;

  const skillContextP: Promise<SkillContext | undefined> | undefined = enableSkills
    ? assembleSkillContext({
        embedder: memory.store.embedder,
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
  const profile = opts.profile ?? new ProfileService();

  // RFC-012: compaction driver (pressure check per run + /compact command).
  const compactionEnabled = opts.enableCompaction ?? process.env["OPENINTJ_COMPACT"] !== "0";
  const subagentsEnabled = opts.enableSubagents ?? process.env["OPENINTJ_SUBAGENTS"] === "1";
  const subagentMaxRisk: ToolRisk =
    opts.subagentMaxRisk ??
    (TOOL_RISK_ORDER.includes(process.env["OPENINTJ_SUBAGENT_MAX_RISK"] as ToolRisk)
      ? (process.env["OPENINTJ_SUBAGENT_MAX_RISK"] as ToolRisk)
      : "read");
  const compactor = createCompactor({
    llm,
    ...(opts.compaction?.pressureTokens ? { pressureTokens: opts.compaction.pressureTokens } : {}),
    ...(opts.compaction?.keepRecentTurns
      ? { keepRecentTurns: opts.compaction.keepRecentTurns }
      : {}),
  });

  // RFC-015: scheduler (runner is attached in attachRun, once the handle exists).
  const schedulerEnabled = opts.enableScheduler ?? process.env["OPENINTJ_SCHEDULER"] !== "0";
  const scheduler =
    opts.scheduler ??
    new SchedulerService(undefined, {
      enabled: schedulerEnabled,
      ...(opts.schedulerQuietHours ? { quietHours: opts.schedulerQuietHours } : {}),
    });

  // RFC-020: identity graph (Gravatar public surface + local corpus linking).
  // The primary person's email/name mirror into the profile identity section
  // (provenance "identity-graph"); hand-written entries are never overwritten.
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
        const skillContext = skillContextP ? await skillContextP : undefined;
        if (!skillContext) return "";
        return skillContext.render(query, {
          ...(taskType ? { taskType } : {}),
          ...(traceId ? { traceId } : {}),
        });
      },
    }),
  );

  const tao = new TaoLoop({
    config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: opts.maxTaoIterations ?? 1 },
    hooks,
    react,
    availableTools: () => toolHub.list(),
    systemPrompt: baseSystemPrompt,
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
  const journal = createInMemoryJournal();

  const selfConsistency = resolveSelfConsistency(opts.selfConsistency);
  const classifier = enableClassifier
    ? new ReinforcingClassifier({ embedder: memory.store.embedder })
    : undefined;
  let classifierReady = false;
  const ensureClassifier = async (): Promise<void> => {
    if (!classifier || classifierReady) return;
    classifierReady = true;
    await classifier.hydrate();
    if (classifier.size === 0) await classifier.addSeeds(DEFAULT_SEEDS);
  };

  return {
    hooks,
    llm,
    control,
    execution,
    memory,
    governance,
    contextEngine,
    hybridIndex,
    ...(classifier ? { classifier } : {}),
    ...(skillLearning ? { skillLearning } : {}),
    ...(dormant ? { dormant } : {}),
    tao,
    runTao,
    journal,
    prompt,
    profile,
    scheduler,
    identity,
    channels,
    ...(stt ? { stt } : {}),
    connectors: connectorRegistry,
    compactor,
    compactionEnabled,
    toolHub,
    subagentsEnabled,
    subagentMaxRisk,
    grants: governance.grants,
    ...(approvalSetup.pending ? { approvals: approvalSetup.pending } : {}),
    selfConsistency,
    ensureClassifier,
    taskPoolEnabled,
    taskPoolActivation: resolveTaskPoolActivation(taskPoolEnabled, Boolean(classifier)),
    classifierStatus: {
      enabled: Boolean(classifier),
      impliedByTaskPool: taskPoolEnabled && !classifierConfigured,
    },
    productBehaviorEnabled,
    inputStructuringConfig,
    ...(modelRuntime
      ? {
          modelRuntime: modelRuntime.status,
          refreshModelRuntime: () => modelRuntime.refreshHealth(),
        }
      : {}),
    ...(taskPool ? { taskPool } : {}),
  };
};

const attachRun = (core: ReturnType<typeof buildAgentCore>): AssembledAgent => {
  const commands = createSlashCommandRegistry(createBuiltinSlashCommands());
  const agent: AssembledAgent = {
    hooks: core.hooks,
    llm: core.llm,
    control: core.control,
    execution: core.execution,
    memory: core.memory,
    governance: core.governance,
    contextEngine: core.contextEngine,
    hybridIndex: core.hybridIndex,
    ...(core.classifier ? { classifier: core.classifier } : {}),
    ...(core.skillLearning ? { skillLearning: core.skillLearning } : {}),
    ...(core.dormant ? { dormant: core.dormant } : {}),
    tao: core.tao,
    journal: core.journal,
    ...(core.modelRuntime ? { modelRuntime: core.modelRuntime } : {}),
    ...(core.refreshModelRuntime ? { refreshModelRuntime: core.refreshModelRuntime } : {}),
    productBehavior: {
      version: PRODUCT_BEHAVIOR_VERSION,
      enabled: core.productBehaviorEnabled,
      cohort: core.productBehaviorEnabled ? "treatment" : "control",
    },
    inputStructuringConfig: core.inputStructuringConfig,
    taskPoolEnabled: core.taskPoolEnabled,
    taskPoolActivation: core.taskPoolActivation,
    classifierStatus: core.classifierStatus,
    commands,
    prompt: core.prompt,
    profile: core.profile,
    scheduler: core.scheduler,
    identity: core.identity,
    channels: core.channels,
    ...(core.stt ? { stt: core.stt } : {}),
    exportSyncBundle() {
      return buildSyncBundle({
        profile: core.profile,
        scheduler: core.scheduler,
        identity: core.identity,
      });
    },
    importSyncBundle(bundle: unknown) {
      return applySyncBundle(bundle as SyncBundle, {
        profile: core.profile,
        scheduler: core.scheduler,
        identity: core.identity,
      });
    },
    connectors: core.connectors,
    async initConnectors() {
      await core.connectors.initAll();
    },
    compactor: core.compactor,
    subagents: undefined as unknown as SubagentSpawner,
    grants: core.grants,
    ...(core.approvals ? { approvals: core.approvals } : {}),
    handle: undefined as unknown as AgentHandle,
    async run(query: string, runOpts: AgentRunOptions = {}) {
      if (!runOpts.skipSlashCommand) {
        const invoked = await invokeSlashCommand(commands, query, {
          journal: core.journal,
          handle: agent.handle,
          conversationId: runOpts.journal?.conversationId ?? "cli",
          ...(core.compactionEnabled
            ? {
                compact: () =>
                  runCompactCommand(
                    core.compactor,
                    core.journal,
                    runOpts.journal?.conversationId ?? "cli",
                    runOpts.llm,
                  ),
              }
            : {}),
        });
        if (invoked) return slashCommandResultToTao(invoked.result);
        return agent.handle.run(query, runOpts);
      }
      await core.hooks.emit("event.PRODUCT_BEHAVIOR", {
        version: PRODUCT_BEHAVIOR_VERSION,
        enabled: core.productBehaviorEnabled,
      });
      if (core.dormant) core.dormant.record(query, "user", { stage: "run.input" });
      if (core.classifier) await core.ensureClassifier();
      const prelude = await prepareRunPrelude(
        {
          hooks: core.hooks,
          llm: core.llm,
          control: core.control,
          productBehaviorEnabled: core.productBehaviorEnabled,
          inputStructuringConfig: core.inputStructuringConfig,
          ...(core.classifier ? { classifier: core.classifier } : {}),
        },
        {
          query,
          memories: core.memory.store.all,
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
        fallback: { conversationId: "cli", sink: core.journal },
        query,
        ...(structured ? { structured } : {}),
      }) as JournalTurnContext;
      // RFC-012: pressure compaction — fold old turns into a summary when the
      // history crosses the threshold; fail-soft, never blocks the run.
      let effectiveHistory = runOpts.history;
      let didCompact = false;
      let overflowRecovered = false;
      if (core.compactionEnabled) {
        const compacted = await core.compactor.compactIfNeeded({
          journal: core.journal,
          conversationId: journalCtx.conversationId,
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
        ...(attachJournal ? { journal: journalCtx } : {}),
      });
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
          Boolean(core.taskPool && cls && shouldUseTaskPool(core.taskPoolEnabled, cls.label)),
          Boolean(core.selfConsistency),
        );
        // Explicit TaskPool wins for eligible complex tasks; self-consistency remains
        // the fallback for all other tasks.
        if (orchestrationMode === "taskpool" && core.taskPool && cls) {
          const { plan } = core.control.processInput(workQuery, cls.label);
          const graph = planGraphToTaskGraph(plan);
          const poolResult = await core.taskPool.submitRun(graph, async (node, ctx) => {
            const stepQuery = `[${node.description}]（步骤 ${node.id}/${node.action}）\n${ctx.goalInput}`;
            return core.runTao(stepQuery, taoOpts(ctx.traceId, ctx.signal));
          });
          result = synthesizeTaskPoolAnswer(poolResult, workQuery);
        } else if (orchestrationMode === "self-consistency" && core.selfConsistency) {
          const { fulfilled } = await forkJoin(
            Array.from({ length: core.selfConsistency.samples }, (_, i) => i),
            (i) => core.runTao(workQuery, taoOpts(`${randomUUID()}-sc${i}`)),
            {
              hooks: core.hooks,
              group: "self-consistency",
              minSuccess: 1,
              ...(core.selfConsistency.maxConcurrency
                ? { concurrency: core.selfConsistency.maxConcurrency }
                : {}),
            },
          );
          result =
            selectConsistentAnswer(fulfilled, core.selfConsistency.strategy) ?? fulfilled[0]!;
        } else {
          result = await core.runTao(workQuery, {
            ...taoOpts(undefined, runOpts.signal, true),
            ...(core.compactionEnabled
              ? {
                  onContextOverflow: async () => {
                    const out = await core.compactor.compactIfNeeded({
                      journal: core.journal,
                      conversationId: journalCtx.conversationId,
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
      if (core.productBehaviorEnabled && structured?.action !== "clarify") {
        const enforced = await enforceProductBehaviorAnswer({
          query,
          draft: result.finalAnswer,
          searchEvidence: resolveSearchEvidenceStatus(result.trajectory),
          revise: async (instruction) =>
            (runOpts.llm ?? core.llm).chat(
              [
                {
                  role: "system",
                  content: "你是最终答案编辑器。严格完成用户要求，只输出修订后的答案。",
                },
                { role: "user", content: instruction },
              ],
              {
                temperature: 0,
                maxTokens: 768,
                ...(runOpts.signal ? { signal: runOpts.signal } : {}),
              },
            ),
        });
        result.finalAnswer = enforced.answer;
      }
      result.finalAnswer = appendSourcesFooter(result.finalAnswer, result.trajectory);
      const labelTags = cls ? [cls.label] : [];
      await core.memory.recordUserInputAsync(query, labelTags);
      await core.memory.recordAssistantOutputAsync(result.finalAnswer, labelTags);
      if (core.classifier && cls) {
        await core.classifier.reinforce(executionQuery, cls.label, {
          signal: outcomeSignal(result.status),
        });
      }
      if (core.skillLearning) {
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
        core.skillLearning.recordOutcome(query, cls?.label, result.status, {
          finalAnswer: result.finalAnswer,
          toolsUsed,
        });
      }
      finalizeJournalTurn(core.journal, journalCtx, {
        userContent: workQuery,
        originalInput: query,
        ...(structured ? { inputStructure: structured } : {}),
        assistantContent: result.finalAnswer,
        messageKind: structured?.action === "clarify" ? "clarification" : "answer",
        status: result.status,
        endStatus: structured?.action === "clarify" ? "clarify" : "completed",
        traceId: result.traceId,
      });
      return result;
    },
  };
  agent.handle = createAgentHandle({
    id: "cli",
    journal: core.journal,
    hooks: core.hooks,
    runTurn: (query, runOpts) => agent.run(query, { ...runOpts, skipSlashCommand: true }),
  });
  // RFC-015: scheduled jobs run as ordinary turns through the handle.
  core.scheduler.attachRunner(async (job) => {
    const result = await agent.handle.run(job.prompt);
    if (result.status !== "completed") {
      throw new Error(`scheduled run ended with status ${result.status}`);
    }
  });
  // RFC-012 S3: subagent spawner; the delegate tool is registered only when
  // enabled, and children run as isolated journal conversations.
  const subagents = createSubagentSpawner({
    // RFC-012 S3: children run under a tool ceiling (default read-risk
    // only), resolved per spawn and enforced via a fresh ToolHub scope.
    tools: () =>
      core.toolHub
        .list()
        .filter(
          (d) =>
            d.name !== "delegate_task" &&
            riskAtLeast(
              core.subagentMaxRisk,
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
        const result = await agent.handle.run(task, {
          journal: {
            conversationId: childConversationId,
            turnId: randomUUID(),
            sink: core.journal,
            originalInput: task,
          },
          signal,
        });
        return { status: result.status, finalAnswer: result.finalAnswer };
      };
      if (!tools) return run();
      return core.toolHub.runInAllowlistScope(async () => {
        core.toolHub.setRunCap(tools);
        return run();
      });
    },
  });
  agent.subagents = subagents;
  if (core.subagentsEnabled) {
    const delegate = createDelegateTool(subagents);
    core.toolHub.register(delegate.descriptor, delegate.handler);
  }
  return agent;
};

/** 异步装配（生产路径：含 Ollama 健康探测与 ADR-002 fallback）。 */
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

export const assembleAgentAsync = async (opts: AgentOptions = {}): Promise<AssembledAgent> => {
  const provider = opts.llmProvider ?? "auto";
  if (opts.syncLlm) {
    const resolved = resolveLlmClientSync({ provider });
    return attachRun(buildAgentCore(opts, resolved.client, resolved.status.provider));
  }
  const hooks = new HookBus();
  const runtime = await resolveModelRuntime({
    provider,
    ...(opts.embedProvider ? { embedProvider: opts.embedProvider } : {}),
    hooks,
  });
  const core = buildAgentCore(
    opts,
    runtime.llm.client,
    runtime.status.llm.provider,
    runtime.embed.embedder,
    runtime,
    hooks,
  );
  return attachRun(core);
};

/** 同步装配（单测快捷路径；跳过 Ollama 探测）。 */
export const assembleAgent = (opts: AgentOptions = {}): AssembledAgent => {
  const provider = opts.llmProvider ?? "auto";
  const resolved = resolveLlmClientSync({ provider });
  const core = buildAgentCore(opts, resolved.client, resolved.status.provider);
  return attachRun(core);
};
