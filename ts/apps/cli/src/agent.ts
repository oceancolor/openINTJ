import { randomUUID } from "node:crypto";
import {
  DEFAULT_SEEDS,
  ReinforcingClassifier,
  decideRoute,
  outcomeSignal,
} from "@openintj/classifier";
import { forkJoin } from "@openintj/concurrency";
/**
 * Agent 装配工厂：组合 core + 4 plane + LLM 适配器为可执行的 Agent。
 * 这是 CLI / Server / Desktop 共用的核心装配逻辑。
 */
import {
  DEFAULT_REACT_CONFIG,
  DEFAULT_TAO_CONFIG,
  type EmbeddingProvider,
  HookBus,
  type LlmClient,
  ReactStateMachine,
  TaoLoop,
  type TaoResult,
  type TaskTypeType,
  attachProductTraitSignals,
  detectTaskType,
  getShaderForTask,
} from "@openintj/core";
import { DormantRuntime, type DormantRuntimeOpts } from "@openintj/dormant";
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
import { GovernancePlane, createToolCallGate } from "@openintj/plane-governance";
import { ContextEngine, MemoryPlane, fragmentsToRanked } from "@openintj/plane-memory";
import {
  DEFAULT_AGENT_SYSTEM_PROMPT,
  PRODUCT_BEHAVIOR_VERSION,
  type SelfConsistencyStrategy,
  appendSourcesFooter,
  assembleSystemPromptPrefix,
  enforceProductBehaviorAnswer,
  resolveDeterministicProductBehaviorAnswer,
  resolvePersonaInjection,
  resolveProductBehaviorEnabled,
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
  /** RFC-007：opt-in TaskPool。 */
  enableTaskPool?: boolean;
  /** 跳过启动期 Ollama 健康探测（单测用）。 */
  syncLlm?: boolean;
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
  taskPoolEnabled: boolean;
  taskPoolActivation: TaskPoolActivationStatus;
  classifierStatus: { enabled: boolean; impliedByTaskPool: boolean };
  run(query: string): Promise<TaoResult>;
}

const buildAgentCore = (
  opts: AgentOptions,
  llm: LlmClient,
  llmProviderId: string,
  embedder?: EmbeddingProvider,
  modelRuntime?: ModelRuntime,
  hooksInput?: HookBus,
): Omit<AssembledAgent, "run" | "productBehavior"> & {
  tao: TaoLoop;
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
} => {
  const hooks = hooksInput ?? new HookBus();
  attachProductTraitSignals(hooks);
  const governance = new GovernancePlane({ hooks });
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
  const wsTools = createWorkspaceTools(resolveWorkspaceConfig(opts, process.cwd()));
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
        ...(skillLearning
          ? {
              extraSources: [
                new DbSkillSource({ approvedSkills: () => skillLearning.listApproved() }),
              ],
              weightFor: (id: string) => skillLearning.weightFor(id),
              onSelected: (query, taskType, ids) =>
                skillLearning.noteSelected(query, taskType, ids),
            }
          : {}),
      })
    : undefined;

  const tao = new TaoLoop({
    config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: opts.maxTaoIterations ?? 1 },
    hooks,
    react,
    availableTools: () => toolHub.list(),
    systemPrompt: baseSystemPrompt,
    contextProvider: async ({ query, history, taskType, topK, traceId }) => {
      const persona = personaEnabled ? (dormant?.personaSystemPrompt() ?? "") : "";
      const skillContext = skillContextP ? await skillContextP : undefined;
      const skillBlock = skillContext
        ? await skillContext.render(query, {
            ...(taskType ? { taskType } : {}),
            ...(traceId ? { traceId } : {}),
          })
        : "";
      const stacked = assembleSystemPromptPrefix({
        base: baseSystemPrompt,
        productBehavior: { enabled: productBehaviorEnabled },
        userPersona: persona,
        skillBlock,
      });
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
    selfConsistency,
    ensureClassifier,
    taskPoolEnabled,
    taskPoolActivation: resolveTaskPoolActivation(taskPoolEnabled, Boolean(classifier)),
    classifierStatus: {
      enabled: Boolean(classifier),
      impliedByTaskPool: taskPoolEnabled && !classifierConfigured,
    },
    productBehaviorEnabled,
    ...(modelRuntime
      ? {
          modelRuntime: modelRuntime.status,
          refreshModelRuntime: () => modelRuntime.refreshHealth(),
        }
      : {}),
    ...(taskPool ? { taskPool } : {}),
  };
};

const attachRun = (core: ReturnType<typeof buildAgentCore>): AssembledAgent => ({
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
  ...(core.modelRuntime ? { modelRuntime: core.modelRuntime } : {}),
  ...(core.refreshModelRuntime ? { refreshModelRuntime: core.refreshModelRuntime } : {}),
  productBehavior: {
    version: PRODUCT_BEHAVIOR_VERSION,
    enabled: core.productBehaviorEnabled,
    cohort: core.productBehaviorEnabled ? "treatment" : "control",
  },
  taskPoolEnabled: core.taskPoolEnabled,
  taskPoolActivation: core.taskPoolActivation,
  classifierStatus: core.classifierStatus,
  async run(query: string) {
    await core.hooks.emit("event.PRODUCT_BEHAVIOR", {
      version: PRODUCT_BEHAVIOR_VERSION,
      enabled: core.productBehaviorEnabled,
    });
    if (core.dormant) core.dormant.record(query, "user", { stage: "run.input" });
    const preflight = core.productBehaviorEnabled
      ? resolveDeterministicProductBehaviorAnswer(query)
      : undefined;
    let cls: Awaited<ReturnType<ReinforcingClassifier["classify"]>> | undefined;
    let route: ReturnType<typeof decideRoute> | undefined;
    if (!preflight && core.classifier) {
      await core.ensureClassifier();
      cls = await core.classifier.classify(query);
      route = decideRoute(cls);
    }
    const taoOpts = (traceId?: string, signal?: AbortSignal) => ({
      ...(cls ? { taskType: cls.label } : {}),
      ...(route?.single ? { enableReact: false } : {}),
      ...(route ? { topK: route.topK } : {}),
      ...(traceId ? { traceId } : {}),
      ...(signal ? { signal } : {}),
    });
    let result: TaoResult;
    if (preflight) {
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
        const { plan } = core.control.processInput(query, cls.label);
        const graph = planGraphToTaskGraph(plan);
        const poolResult = await core.taskPool.submitRun(graph, async (node, ctx) => {
          const stepQuery = `[${node.description}]（步骤 ${node.id}/${node.action}）\n${ctx.goalInput}`;
          return core.tao.run(stepQuery, taoOpts(ctx.traceId, ctx.signal));
        });
        result = synthesizeTaskPoolAnswer(poolResult, query);
      } else if (orchestrationMode === "self-consistency" && core.selfConsistency) {
        const { fulfilled } = await forkJoin(
          Array.from({ length: core.selfConsistency.samples }, (_, i) => i),
          (i) => core.tao.run(query, taoOpts(`${randomUUID()}-sc${i}`)),
          {
            hooks: core.hooks,
            group: "self-consistency",
            minSuccess: 1,
            ...(core.selfConsistency.maxConcurrency
              ? { concurrency: core.selfConsistency.maxConcurrency }
              : {}),
          },
        );
        result = selectConsistentAnswer(fulfilled, core.selfConsistency.strategy) ?? fulfilled[0]!;
      } else {
        result = await core.tao.run(query, taoOpts());
      }
    }
    if (core.productBehaviorEnabled) {
      const enforced = await enforceProductBehaviorAnswer({
        query,
        draft: result.finalAnswer,
        revise: async (instruction) =>
          core.llm.chat(
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
    result.finalAnswer = appendSourcesFooter(result.finalAnswer, result.trajectory);
    const labelTags = cls ? [cls.label] : [];
    await core.memory.recordUserInputAsync(query, labelTags);
    await core.memory.recordAssistantOutputAsync(result.finalAnswer, labelTags);
    if (core.classifier && cls) {
      await core.classifier.reinforce(query, cls.label, { signal: outcomeSignal(result.status) });
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
    return result;
  },
});

/** 异步装配（生产路径：含 Ollama 健康探测与 ADR-002 fallback）。 */
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
