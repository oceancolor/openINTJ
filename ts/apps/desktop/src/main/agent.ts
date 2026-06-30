import { randomUUID } from "node:crypto";
import { type RateLimitOpts, RateLimitedLlmClient, forkJoin } from "@openintj/concurrency";
import {
  DEFAULT_REACT_CONFIG,
  DEFAULT_TAO_CONFIG,
  HookBus,
  type LlmClient,
  ReactStateMachine,
  TaoLoop,
  type TaoResult,
} from "@openintj/core";
import {
  type DormantPersistenceAdapter,
  DormantRuntime,
  type DormantRuntimeOpts,
} from "@openintj/dormant";
import { HunyuanClient, createHunyuanSearchTool } from "@openintj/llm-hunyuan";
import { OllamaClient } from "@openintj/llm-ollama";
import { ControlPlane } from "@openintj/plane-control";
import {
  Executor,
  ToolHub,
  type WorkspaceTools,
  createWorkspaceTools,
} from "@openintj/plane-execution";
import { GovernancePlane } from "@openintj/plane-governance";
import {
  ContextEngine,
  MemoryPlane,
  type PersistenceMode,
  type PersistentMemoryStore,
  createPersistentMemoryStore,
} from "@openintj/plane-memory";
import {
  DEFAULT_AGENT_SYSTEM_PROMPT,
  type ResolvedWorkspaceConfig,
  type SelfConsistencyStrategy,
  appendSourcesFooter,
  resolveSelfConsistency,
  resolveWorkspaceConfig,
  selectConsistentAnswer,
} from "@openintj/shared";
import { createSqliteDormantStore } from "@openintj/storage-sqlite";
import {
  type HybridConfig,
  type HybridDoc,
  HybridRetriever,
  type HybridScored,
} from "@openintj/taskpool";
import {
  type AttachOtelOpts,
  type AttachedOtel,
  attachOtelToHooks,
} from "@openintj/telemetry-otel";

export type LlmProvider = "ollama" | "hunyuan" | "mock";

export interface DesktopAgentOpts {
  llmProvider?: LlmProvider;
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
   * Dormant 持久化策略（仅 enableDormant=true 时生效）。
   * - 'auto'（默认）：跟随主持久化（real → SqliteDormantStore，memory → 不挂 adapter）
   * - 'memory'：强制不挂 adapter
   * - 'real'：强制用 SqliteDormantStore（缺 dbPath 报错）
   */
  dormantPersistence?: "auto" | "memory" | "real";
  /** 自定义 dormant SQLite 文件路径；缺省 `${dataDir}/dormant.sqlite`。 */
  dormantDbPath?: string;
  /** RFC-003 方向 2：默认检索模式。env OPENINTJ_RETRIEVAL_MODE=hybrid 也启用。 */
  retrievalMode?: "vector" | "hybrid";
  /** RFC-003 方向 1：LLM 速率限制。env OPENINTJ_RATE_LIMIT_QPS 也启用。 */
  rateLimit?: RateLimitOpts;
  /**
   * RFC-003 方向一/二接入：opt-in 自一致性（并行多采样 + 投票）。
   * samples>1 时每次 run 用 forkJoin 并行跑 N 个 tao.run，再按 strategy 选最终答案。
   * 默认关闭；env OPENINTJ_SELF_CONSISTENCY=N / OPENINTJ_SELF_CONSISTENCY_STRATEGY 也可启用。
   */
  selfConsistency?: { samples: number; strategy?: SelfConsistencyStrategy };
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
}

const buildLlm = (provider: LlmProvider): LlmClient => {
  if (provider === "hunyuan") return new HunyuanClient();
  if (provider === "ollama") return new OllamaClient();
  return new HunyuanClient({ apiKey: "" });
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

/** 桌面端的 hybrid 检索辅助 —— 与 server 端实现保持等价。 */
export type DesktopHybridHit = HybridScored<
  HybridDoc & {
    metadata: {
      memoryType: string;
      taskTags: readonly string[];
      importance: number;
    };
  }
>;

export interface DesktopRetrieveHybridOpts {
  topK?: number;
  config?: Partial<HybridConfig>;
  memoryTypes?: readonly string[];
  taskTags?: readonly string[];
  queryEmbedding?: readonly number[];
}

const buildHybridRetrieve = (
  store: PersistentMemoryStore,
): ((q: string, opts?: DesktopRetrieveHybridOpts) => Promise<DesktopHybridHit[]>) => {
  return async (query, opts = {}) => {
    let fragments = store.all;
    if (opts.memoryTypes && opts.memoryTypes.length > 0) {
      const set = new Set(opts.memoryTypes);
      fragments = fragments.filter((f) => set.has(f.memoryType));
    }
    if (opts.taskTags && opts.taskTags.length > 0) {
      const tagSet = new Set(opts.taskTags);
      fragments = fragments.filter((f) => f.taskTags.some((t) => tagSet.has(t)));
    }
    if (fragments.length === 0) return [];
    const docs: DesktopHybridHit["doc"][] = fragments.map((f) => ({
      id: f.fragmentId,
      text: f.content,
      vector: f.embedding,
      metadata: {
        memoryType: f.memoryType,
        taskTags: f.taskTags,
        importance: f.importance,
      },
    }));
    let qVec: readonly number[] | undefined = opts.queryEmbedding;
    if (qVec === undefined) {
      const r = store.embedder.embed(query);
      qVec = r instanceof Promise ? await r : r;
    }
    const retriever = new HybridRetriever<DesktopHybridHit["doc"]>({
      ...(opts.config ? { config: opts.config } : {}),
    });
    retriever.index(docs);
    return retriever.search(query, qVec, opts.topK ?? 10);
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
  persistenceInfo: { mode: PersistenceMode; dataDir?: string };
  retrievalMode: "vector" | "hybrid";
  /** RFC-003 方向 3 蛰伏记忆学习；仅 opts.enableDormant=true 时存在。 */
  dormant?: DormantRuntime;
  /** Dormant 子系统的持久化信息（dormant 启用且挂了 adapter 时存在）。 */
  dormantPersistenceInfo?: { adapter: string; dbPath?: string };
  /** OpenTelemetry 接线状态（enableOtel 真值时存在；含 dispose 钩子）。 */
  otel?: AttachedOtel;
  /**
   * 工作区系统能力面（RFC-004 §8）：与 Agent 的 read_file/write_file 工具**共用同一沙箱**，
   * 供 IPC handler 直接复用，保证 UI 直接读写与 Agent 工具读写遵循完全相同的边界。
   */
  workspace: { config: ResolvedWorkspaceConfig; tools: WorkspaceTools };
  /** 基于 HybridRetriever 的检索辅助；无论 retrievalMode 都可用。 */
  retrieveHybrid(query: string, opts?: DesktopRetrieveHybridOpts): Promise<DesktopHybridHit[]>;
  run(query: string): Promise<TaoResult>;
  status(): {
    llm: ReturnType<LlmClient["getStatus"]>;
    memory: ReturnType<MemoryPlane["getStats"]>;
    governance: ReturnType<GovernancePlane["getStats"]>;
    tools: string[];
    persistence: { mode: PersistenceMode; dataDir?: string };
    retrievalMode: "vector" | "hybrid";
    dormant?: {
      enabled: true;
      passiveSize: number;
      pendingProposals: number;
      persistence?: { adapter: string; dbPath?: string };
    };
  };
  close(): Promise<void>;
}

const resolveDesktopOtel = (opts: DesktopAgentOpts): AttachOtelOpts | undefined => {
  if (opts.enableOtel === true) return {};
  if (opts.enableOtel && typeof opts.enableOtel === "object") return opts.enableOtel;
  if (opts.enableOtel === false) return undefined;
  if (process.env["OPENINTJ_OTEL"] === "1") return {};
  return undefined;
};

export const assembleDesktopAgent = async (opts: DesktopAgentOpts = {}): Promise<DesktopAgent> => {
  const hooks = new HookBus();
  const otelOpts = resolveDesktopOtel(opts);
  const otel = otelOpts ? attachOtelToHooks(hooks, otelOpts) : undefined;
  const rawLlm = buildLlm(opts.llmProvider ?? "mock");
  const rateLimit = resolveRateLimit(opts);
  const llm: LlmClient = rateLimit ? new RateLimitedLlmClient(rawLlm, rateLimit) : rawLlm;
  const persistence = resolvePersistence(opts);
  const embeddingDim = opts.embeddingDim ?? 64;
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
  const dormant = dormantEnabled
    ? new DormantRuntime({
        eventIdPrefix: "desktop",
        // 默认给磁盘事件一个 LRU 上限，防 dormant_events 无限增长；显式 dormantOpts 可覆盖。
        maxDiskEvents: 50_000,
        ...(opts.dormantOpts ?? {}),
        ...(dormantAdapter ? { adapter: dormantAdapter } : {}),
      })
    : undefined;
  if (dormant) await dormant.hydrate();
  const retrievalMode = resolveRetrievalMode(opts);

  const governance = new GovernancePlane({ hooks });
  const persistentStore = await createPersistentMemoryStore({
    ...(persistence.dataDir ? { dataDir: persistence.dataDir } : {}),
    mode: persistence.mode,
    embeddingDim,
    storeConfig: { embeddingDim },
  });

  const memory = new MemoryPlane({ hooks });
  Object.defineProperty(memory, "store", {
    value: persistentStore,
    writable: false,
  });
  Object.defineProperty(memory.retriever, "store", {
    value: persistentStore,
    writable: false,
  });

  const control = new ControlPlane();
  const contextEngine = new ContextEngine({
    store: persistentStore,
    hooks,
  });
  const toolHub = new ToolHub({ hooks });
  const noop = () => ({ note: "[mock]" });
  // search 工具优先接混元联网搜索（按 rawLlm，避开速率限制包装层）；非混元则保持占位。
  const searchHandler = rawLlm instanceof HunyuanClient ? createHunyuanSearchTool(rawLlm) : noop;
  // 真实工作区工具：read_file / write_file 沙箱限定在 workspace 根内，execute_command 默认禁用。
  const wsConfig = resolveWorkspaceConfig(opts, process.cwd());
  const wsTools = createWorkspaceTools(wsConfig);
  toolHub.registerBuiltinTools({
    readFile: wsTools.readFile,
    writeFile: wsTools.writeFile,
    executeCommand: wsTools.executeCommand,
    search: searchHandler,
  });
  const execution = new Executor({ toolHub, hooks, registerBuiltins: false });

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
    config: {
      ...DEFAULT_TAO_CONFIG,
      maxTaoIterations: opts.maxTaoIterations ?? 2,
    },
    hooks,
    react,
    availableTools: () => toolHub.list(),
    systemPrompt: baseSystemPrompt,
    // 每轮注入：①已批准的钝化记忆 persona（无需检索）②检索到的 [记忆参考]。
    contextProvider: async ({ query, history, taskType, traceId }) => {
      const persona = dormant?.personaSystemPrompt() ?? "";
      const snap = await contextEngine.build({
        query,
        history,
        taskType,
        systemPrompt: persona ? `${baseSystemPrompt}\n\n${persona}` : baseSystemPrompt,
        topK: 6,
        ...(traceId ? { traceId } : {}),
      });
      return snap.systemPrompt;
    },
  });

  const retrieveHybrid = buildHybridRetrieve(persistentStore);
  const selfConsistency = resolveSelfConsistency(opts.selfConsistency);

  return {
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
    persistenceInfo: persistence,
    retrievalMode,
    ...(dormant ? { dormant } : {}),
    ...(dormantPersistenceInfo ? { dormantPersistenceInfo } : {}),
    ...(otel ? { otel } : {}),
    workspace: { config: wsConfig, tools: wsTools },
    retrieveHybrid,
    async run(query: string) {
      if (dormant) dormant.record(query, "user", { stage: "run.input" });
      // 先跑（contextProvider 会检索此前已落盘的记忆），再记录本轮 → 避免检索命中当前输入本身。
      let result: TaoResult;
      if (selfConsistency) {
        // 方向一/二：并行多采样 + 投票。forkJoin 会发 forkjoin.* 事件 → OTel span/metric。
        const { fulfilled } = await forkJoin(
          Array.from({ length: selfConsistency.samples }, (_, i) => i),
          (i) => tao.run(query, { traceId: `${randomUUID()}-sc${i}` }),
          { hooks, group: "self-consistency", minSuccess: 1 },
        );
        result = selectConsistentAnswer(fulfilled, selfConsistency.strategy) ?? fulfilled[0]!;
      } else {
        result = await tao.run(query);
      }
      // 把 search 工具命中的联网来源追加到答案末尾 → 随记忆/dormant 一起入库。
      result.finalAnswer = appendSourcesFooter(result.finalAnswer, result.trajectory);
      memory.recordUserInput(query);
      memory.recordAssistantOutput(result.finalAnswer);
      if (dormant)
        dormant.record(result.finalAnswer, "agent", {
          stage: "run.output",
          iterations: result.iterations,
        });
      await persistentStore.awaitPendingWrites();
      return result;
    },
    status() {
      return {
        llm: llm.getStatus(),
        memory: memory.getStats(),
        governance: governance.getStats(),
        tools: toolHub.list().map((t) => t.name),
        persistence,
        retrievalMode,
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
      };
    },
    async close() {
      if (otel) otel.dispose();
      if (dormant) await dormant.close();
      await persistentStore.close();
    },
  };
};
