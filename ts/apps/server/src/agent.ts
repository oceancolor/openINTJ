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
import { HunyuanClient } from "@openintj/llm-hunyuan";
import { OllamaClient } from "@openintj/llm-ollama";
import { ControlPlane } from "@openintj/plane-control";
import { Executor, ToolHub } from "@openintj/plane-execution";
import { GovernancePlane } from "@openintj/plane-governance";
import {
  ContextEngine,
  MemoryPlane,
  type PersistenceMode,
  type PersistentMemoryStore,
  createPersistentMemoryStore,
} from "@openintj/plane-memory";
import { createSqliteDormantStore } from "@openintj/storage-sqlite";
import {
  type AttachOtelOpts,
  type AttachedOtel,
  attachOtelToHooks,
} from "@openintj/telemetry-otel";
import { type RateLimitOpts, RateLimitedLlmClient } from "./rate-limited-llm.js";

export interface ServerAgentOpts {
  llmProvider?: "ollama" | "hunyuan" | "mock";
  systemPrompt?: string;
  maxTaoIterations?: number;
  /**
   * 持久化数据目录。
   * - 优先级：opts.dataDir > 环境变量 OPENINTJ_DATA_DIR > 不持久化（in-memory）
   * - 当显式传入或 env 提供时，模式 = "real"，使用 LanceDB + SQLite 写盘
   */
  dataDir?: string;
  /** 显式锁定模式（覆盖 dataDir 推断）。 */
  persistenceMode?: PersistenceMode;
  /** 向量维度（默认 64，对齐 SimpleEmbedder）。 */
  embeddingDim?: number;
  /**
   * 启用 RFC-003 方向 3 - Dormant Memory Learning。
   * - true：构造 DormantRuntime，agent.run() 自动把 user query / final answer 记入 PassiveStore
   * - env OPENINTJ_DORMANT=1 也会启用
   * 默认关闭以保证零侵入。
   */
  enableDormant?: boolean;
  /** DormantRuntime 配置（仅 enableDormant=true 时生效）。 */
  dormantOpts?: DormantRuntimeOpts;
  /**
   * Dormant 持久化策略（仅 enableDormant=true 时生效）。
   * - 'auto'（默认）：跟随主持久化（real → 用 SqliteDormantStore，memory → 不挂 adapter）
   * - 'memory'：强制不挂 adapter（即使 dataDir 存在）
   * - 'real'：强制用 SqliteDormantStore，缺 dbPath 会显式报错
   * 同时 env `OPENINTJ_DORMANT_DB_PATH` 可覆盖默认 SQLite 路径。
   */
  dormantPersistence?: "auto" | "memory" | "real";
  /** 自定义 dormant SQLite 文件路径；缺省 `${dataDir}/dormant.sqlite`。 */
  dormantDbPath?: string;
  /**
   * 默认检索路径：
   *  - 'vector'（默认）：MemoryPlane.retrieve（cosine + 朴素 keyword + recency 衰减）
   *  - 'hybrid'：HybridRetriever（cosine + BM25 + 可选 RRF）
   * 路由 /api/memory 不带 mode 参数时按这里决定；可被 ?mode= 显式覆盖。
   * env OPENINTJ_RETRIEVAL_MODE=hybrid 也会切换默认值。
   */
  retrievalMode?: "vector" | "hybrid";
  /**
   * 启用 LLM 速率限制（TokenBucket 装饰器）。
   * - opts.rateLimit.qps：每秒平均请求数
   * - opts.rateLimit.burst：突发上限（默认 = qps）
   * - env OPENINTJ_RATE_LIMIT_QPS=数字 也会启用；OPENINTJ_RATE_LIMIT_BURST 可选
   * 默认关闭；零开销路径。
   */
  rateLimit?: RateLimitOpts;
  /**
   * RFC-003 衍生 / Phase 3.8 ：把 HookBus 接到 OpenTelemetry。
   * - true：attachOtelToHooks(hooks)，使用默认 scope（@openintj/telemetry-otel）
   * - AttachOtelOpts：透传给 attachOtelToHooks
   * - env OPENINTJ_OTEL=1 也会启用
   * 默认关闭。未注册 TracerProvider/MeterProvider 时 attach 也是零开销（OTel API 走 no-op）。
   * 真正想 export trace/metric 时，调用方需自己先调 `bootstrapNodeOtel(...)` 或注册自家 SDK。
   */
  enableOtel?: boolean | AttachOtelOpts;
}

export interface ServerAgent {
  hooks: HookBus;
  llm: LlmClient;
  control: ControlPlane;
  execution: Executor;
  memory: MemoryPlane;
  persistentStore: PersistentMemoryStore;
  governance: GovernancePlane;
  contextEngine: ContextEngine;
  tao: TaoLoop;
  /** 当前生效的持久化模式 + 数据目录（如果是 real 模式）。 */
  persistenceInfo: { mode: PersistenceMode; dataDir?: string };
  /** RFC-003 方向 3 蛰伏记忆学习；仅 opts.enableDormant=true 时存在。 */
  dormant?: DormantRuntime;
  /** Dormant 子系统的持久化信息（dormant 启用且挂了 adapter 时存在）。 */
  dormantPersistenceInfo?: { adapter: string; dbPath?: string };
  /** 当前默认检索模式（由 opts / env 决定）。 */
  retrievalMode: "vector" | "hybrid";
  /** OpenTelemetry 接线状态（enableOtel 真值时存在；含 dispose 钩子）。 */
  otel?: AttachedOtel;
  run(query: string): Promise<TaoResult>;
  status(): Promise<{
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
  }>;
  close(): Promise<void>;
}

const buildLlm = (provider: ServerAgentOpts["llmProvider"]): LlmClient => {
  if (provider === "hunyuan") return new HunyuanClient();
  if (provider === "ollama") return new OllamaClient();
  return new HunyuanClient({ apiKey: "" });
};

const resolvePersistence = (opts: ServerAgentOpts): { mode: PersistenceMode; dataDir?: string } => {
  const explicitMode = opts.persistenceMode;
  const dataDir = opts.dataDir ?? process.env["OPENINTJ_DATA_DIR"];
  if (explicitMode === "memory") return { mode: "memory" };
  if (explicitMode === "real") {
    if (!dataDir) {
      throw new Error(
        "ServerAgent: persistenceMode='real' requires dataDir or OPENINTJ_DATA_DIR env",
      );
    }
    return { mode: "real", dataDir };
  }
  if (dataDir) return { mode: "real", dataDir };
  return { mode: "memory" };
};

const resolveDormantEnabled = (opts: ServerAgentOpts): boolean => {
  if (opts.enableDormant !== undefined) return opts.enableDormant;
  return process.env["OPENINTJ_DORMANT"] === "1";
};

const resolveRetrievalMode = (opts: ServerAgentOpts): "vector" | "hybrid" => {
  if (opts.retrievalMode) return opts.retrievalMode;
  if (process.env["OPENINTJ_RETRIEVAL_MODE"] === "hybrid") return "hybrid";
  return "vector";
};

type DormantPersistenceResolution = { kind: "memory" } | { kind: "real"; dbPath: string };

const resolveDormantPersistence = (
  opts: ServerAgentOpts,
  persistence: { mode: PersistenceMode; dataDir?: string },
): DormantPersistenceResolution => {
  const strategy = opts.dormantPersistence ?? "auto";
  const dbPathOverride = opts.dormantDbPath ?? process.env["OPENINTJ_DORMANT_DB_PATH"];
  if (strategy === "memory") return { kind: "memory" };
  if (strategy === "real") {
    const dbPath =
      dbPathOverride ?? (persistence.dataDir ? `${persistence.dataDir}/dormant.sqlite` : undefined);
    if (!dbPath) {
      throw new Error(
        "ServerAgent: dormantPersistence='real' requires dataDir, dormantDbPath, or OPENINTJ_DORMANT_DB_PATH",
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

const resolveOtel = (opts: ServerAgentOpts): AttachOtelOpts | undefined => {
  if (opts.enableOtel === true) return {};
  if (opts.enableOtel && typeof opts.enableOtel === "object") return opts.enableOtel;
  if (opts.enableOtel === false) return undefined;
  if (process.env["OPENINTJ_OTEL"] === "1") return {};
  return undefined;
};

const resolveRateLimit = (opts: ServerAgentOpts): RateLimitOpts | undefined => {
  if (opts.rateLimit) return opts.rateLimit;
  const qpsRaw = process.env["OPENINTJ_RATE_LIMIT_QPS"];
  if (!qpsRaw) return undefined;
  const qps = Number(qpsRaw);
  if (!Number.isFinite(qps) || qps <= 0) return undefined;
  const burstRaw = process.env["OPENINTJ_RATE_LIMIT_BURST"];
  const burst = burstRaw ? Number(burstRaw) : undefined;
  return burst && Number.isFinite(burst) && burst > 0 ? { qps, burst } : { qps };
};

export const assembleServerAgent = async (opts: ServerAgentOpts = {}): Promise<ServerAgent> => {
  const hooks = new HookBus();
  const otelOpts = resolveOtel(opts);
  const otel = otelOpts ? attachOtelToHooks(hooks, otelOpts) : undefined;
  const rawLlm = buildLlm(opts.llmProvider);
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
      const dormantPersistence = resolveDormantPersistence(opts, persistence);
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
        eventIdPrefix: "server",
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
  toolHub.registerBuiltinTools({
    readFile: noop,
    writeFile: noop,
    executeCommand: noop,
    search: noop,
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
  const tao = new TaoLoop({
    config: {
      ...DEFAULT_TAO_CONFIG,
      maxTaoIterations: opts.maxTaoIterations ?? 1,
    },
    hooks,
    react,
    availableTools: () => toolHub.list(),
    ...(opts.systemPrompt ? { systemPrompt: opts.systemPrompt } : {}),
  });

  return {
    hooks,
    llm,
    control,
    execution,
    memory,
    persistentStore,
    governance,
    contextEngine,
    tao,
    persistenceInfo: persistence,
    ...(dormant ? { dormant } : {}),
    ...(dormantPersistenceInfo ? { dormantPersistenceInfo } : {}),
    retrievalMode,
    ...(otel ? { otel } : {}),
    async run(query: string) {
      memory.recordUserInput(query);
      if (dormant) dormant.record(query, "user", { stage: "run.input" });
      const result = await tao.run(query);
      memory.recordAssistantOutput(result.finalAnswer);
      if (dormant)
        dormant.record(result.finalAnswer, "agent", {
          stage: "run.output",
          iterations: result.iterations,
        });
      await persistentStore.awaitPendingWrites();
      return result;
    },
    async status() {
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
