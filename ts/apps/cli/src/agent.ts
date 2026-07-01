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
  HookBus,
  type LlmClient,
  ReactStateMachine,
  TaoLoop,
  type TaoResult,
  type TaskTypeType,
} from "@openintj/core";
import { HunyuanClient, createHunyuanSearchTool } from "@openintj/llm-hunyuan";
import { OllamaClient } from "@openintj/llm-ollama";
import { ControlPlane } from "@openintj/plane-control";
import {
  Executor,
  ToolHub,
  createWebSearchTool,
  createWorkspaceTools,
  resolveWebSearchConfig,
} from "@openintj/plane-execution";
import { GovernancePlane } from "@openintj/plane-governance";
import { ContextEngine, MemoryPlane, fragmentsToRanked } from "@openintj/plane-memory";
import {
  DEFAULT_AGENT_SYSTEM_PROMPT,
  type SelfConsistencyStrategy,
  appendSourcesFooter,
  resolveSelfConsistency,
  resolveWorkspaceConfig,
  selectConsistentAnswer,
} from "@openintj/shared";
import { MemoryHybridIndex } from "@openintj/taskpool";

export type LlmProvider = "auto" | "hunyuan" | "ollama" | "mock";

export interface AgentOptions {
  llmProvider?: LlmProvider;
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
  /**
   * RFC-003 方向一/二接入：opt-in 自一致性（并行多采样 + 投票）。
   * samples>1 时每次 run 用 forkJoin 并行跑 N 个 tao.run，再按 strategy 选最终答案。
   * 默认关闭；env OPENINTJ_SELF_CONSISTENCY=N / OPENINTJ_SELF_CONSISTENCY_STRATEGY 也可启用。
   */
  selfConsistency?: { samples: number; strategy?: SelfConsistencyStrategy };
  /**
   * 前端可强化分类器：开启后每次 run 先分类 → 注入 taskType + 记忆 label，高置信简单类
   * 路由单次 LLM 降 token，收尾用 outcome 强化。默认关（env OPENINTJ_CLASSIFIER=1 也可开）。
   */
  enableClassifier?: boolean;
}

export interface AssembledAgent {
  hooks: HookBus;
  llm: LlmClient;
  control: ControlPlane;
  execution: Executor;
  memory: MemoryPlane;
  governance: GovernancePlane;
  contextEngine: ContextEngine;
  /** session 级增量混合检索索引（订阅 event.MEMORY_WRITTEN 自动维护）。 */
  hybridIndex: MemoryHybridIndex;
  /** 前端可强化分类器；仅 enableClassifier 时存在。 */
  classifier?: ReinforcingClassifier;
  tao: TaoLoop;
  run(query: string): Promise<TaoResult>;
}

const pickLlm = (provider: LlmProvider): LlmClient => {
  switch (provider) {
    case "hunyuan":
      return HunyuanClient.fromEnv();
    case "ollama":
      return OllamaClient.fromEnv();
    case "mock":
      return new HunyuanClient({ apiKey: "" });
    default: {
      // auto: 优先 hunyuan（如有 key），否则 ollama
      if (process.env["HUNYUAN_API_KEY"]) {
        return HunyuanClient.fromEnv();
      }
      return OllamaClient.fromEnv();
    }
  }
};

export const assembleAgent = (opts: AgentOptions = {}): AssembledAgent => {
  const hooks = new HookBus();
  const llm = pickLlm(opts.llmProvider ?? "auto");

  // 治理 → 执行 → 记忆 → 控制
  const governance = new GovernancePlane({ hooks });
  const memory = new MemoryPlane({ hooks });
  const control = new ControlPlane();

  // session 级共享 HybridRetriever：CLI 为内存态，开局 seed 空集，之后订阅 change-feed 增量维护。
  const hybridIndex = new MemoryHybridIndex();
  hybridIndex.seed(memory.store.all);
  hybridIndex.subscribe(hooks);

  // A1.3 opt-in：OPENINTJ_LOOP_HYBRID=1 时主循环检索改走 session 级增量 HybridRetriever。
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

  const toolHub = new ToolHub({ hooks });
  const handlers = opts.toolHandlers ?? {};

  // 默认 handler：真实工作区工具（沙箱限定在 workspace 根内，命令默认禁用）。
  const wsTools = createWorkspaceTools(resolveWorkspaceConfig(opts, process.cwd()));
  const defaultSearch = (params: Record<string, unknown>) => ({
    note: "[mock search]",
    query: params["query"],
    hits: [],
  });

  // search 默认优先级：外部 Web Search（Tavily/Brave）> 混元内建联网搜索（仅旧平台有效）> 占位。
  const webSearchCfg = resolveWebSearchConfig();
  const defaultSearchHandler = webSearchCfg
    ? createWebSearchTool(webSearchCfg)
    : llm instanceof HunyuanClient
      ? createHunyuanSearchTool(llm)
      : defaultSearch;
  toolHub.registerBuiltinTools({
    readFile: handlers.readFile ?? wsTools.readFile,
    writeFile: handlers.writeFile ?? wsTools.writeFile,
    executeCommand: handlers.executeCommand ?? wsTools.executeCommand,
    search: handlers.search ?? defaultSearchHandler,
  });
  const execution = new Executor({ toolHub, hooks, registerBuiltins: false });

  // TAO ←→ ReAct 装配
  const react = new ReactStateMachine({
    config: DEFAULT_REACT_CONFIG,
    hooks,
    llm,
    toolRunner: (
      name: string,
      params: Record<string, unknown>,
      callOpts?: { traceId?: string; timeoutMs?: number },
    ) => toolHub.call(name, params, callOpts ?? {}),
  });
  const baseSystemPrompt = opts.systemPrompt ?? DEFAULT_AGENT_SYSTEM_PROMPT;
  const tao = new TaoLoop({
    config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: opts.maxTaoIterations ?? 1 },
    hooks,
    react,
    availableTools: () => toolHub.list(),
    systemPrompt: baseSystemPrompt,
    // 每轮从记忆里检索相关片段注入 system prompt（多轮/编程式调用时也能引用历史）。
    contextProvider: async ({ query, history, taskType, topK, traceId }) => {
      const snap = await contextEngine.build({
        query,
        history,
        taskType,
        systemPrompt: baseSystemPrompt,
        topK: topK ?? 6,
        ...(traceId ? { traceId } : {}),
      });
      return snap.systemPrompt;
    },
  });

  const selfConsistency = resolveSelfConsistency(opts.selfConsistency);

  // 前端可强化分类器（opt-in）。CLI 为内存态，无持久化 store；首次 run 懒加载种子。
  const enableClassifier = opts.enableClassifier ?? process.env["OPENINTJ_CLASSIFIER"] === "1";
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
    tao,
    async run(query: string) {
      // 先跑（contextProvider 检索此前记忆），再记录本轮 → 避免检索命中当前输入本身。
      // 前端分类器：预分类 → taskType + 降 token 路由（高置信简单类走单次 LLM）。
      let cls: Awaited<ReturnType<ReinforcingClassifier["classify"]>> | undefined;
      let route: ReturnType<typeof decideRoute> | undefined;
      if (classifier) {
        await ensureClassifier();
        cls = await classifier.classify(query);
        route = decideRoute(cls);
      }
      const taoOpts = (traceId?: string) => ({
        ...(cls ? { taskType: cls.label } : {}),
        ...(route?.single ? { enableReact: false } : {}),
        ...(route ? { topK: route.topK } : {}),
        ...(traceId ? { traceId } : {}),
      });
      let result: TaoResult;
      if (selfConsistency) {
        // 方向一/二：并行多采样 + 投票。forkJoin 会发 forkjoin.* 事件 → OTel span/metric。
        const { fulfilled } = await forkJoin(
          Array.from({ length: selfConsistency.samples }, (_, i) => i),
          (i) => tao.run(query, taoOpts(`${randomUUID()}-sc${i}`)),
          { hooks, group: "self-consistency", minSuccess: 1 },
        );
        result = selectConsistentAnswer(fulfilled, selfConsistency.strategy) ?? fulfilled[0]!;
      } else {
        result = await tao.run(query, taoOpts());
      }
      result.finalAnswer = appendSourcesFooter(result.finalAnswer, result.trajectory);
      // 记忆带上分类 label（与 retriever taskType ×1.3 加成叠加，随使用复利）。
      const labelTags = cls ? [cls.label] : [];
      memory.recordUserInput(query, labelTags);
      memory.recordAssistantOutput(result.finalAnswer, labelTags);
      // 收尾反馈：用 outcome 强化分类器（与记忆写入同一收尾点）。
      if (classifier && cls) {
        await classifier.reinforce(query, cls.label, { signal: outcomeSignal(result.status) });
      }
      return result;
    },
  };
};
