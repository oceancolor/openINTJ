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
} from "@openintj/core";
import { randomUUID } from "node:crypto";
import { forkJoin } from "@openintj/concurrency";
import { HunyuanClient, createHunyuanSearchTool } from "@openintj/llm-hunyuan";
import { OllamaClient } from "@openintj/llm-ollama";
import { ControlPlane } from "@openintj/plane-control";
import { Executor, ToolHub, createWorkspaceTools } from "@openintj/plane-execution";
import { GovernancePlane } from "@openintj/plane-governance";
import { ContextEngine, MemoryPlane } from "@openintj/plane-memory";
import {
  DEFAULT_AGENT_SYSTEM_PROMPT,
  type SelfConsistencyStrategy,
  appendSourcesFooter,
  resolveSelfConsistency,
  resolveWorkspaceConfig,
  selectConsistentAnswer,
} from "@openintj/shared";

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
}

export interface AssembledAgent {
  hooks: HookBus;
  llm: LlmClient;
  control: ControlPlane;
  execution: Executor;
  memory: MemoryPlane;
  governance: GovernancePlane;
  contextEngine: ContextEngine;
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
  const contextEngine = new ContextEngine({ store: memory.store, hooks });

  const toolHub = new ToolHub({ hooks });
  const handlers = opts.toolHandlers ?? {};

  // 默认 handler：真实工作区工具（沙箱限定在 workspace 根内，命令默认禁用）。
  const wsTools = createWorkspaceTools(resolveWorkspaceConfig(opts, process.cwd()));
  const defaultSearch = (params: Record<string, unknown>) => ({
    note: "[mock search]",
    query: params["query"],
    hits: [],
  });

  // search 默认接混元联网搜索（llm 为混元时）；否则退回无副作用占位。
  const defaultSearchHandler =
    llm instanceof HunyuanClient ? createHunyuanSearchTool(llm) : defaultSearch;
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
    contextProvider: async ({ query, history, taskType, traceId }) => {
      const snap = await contextEngine.build({
        query,
        history,
        taskType,
        systemPrompt: baseSystemPrompt,
        topK: 6,
        ...(traceId ? { traceId } : {}),
      });
      return snap.systemPrompt;
    },
  });

  const selfConsistency = resolveSelfConsistency(opts.selfConsistency);

  return {
    hooks,
    llm,
    control,
    execution,
    memory,
    governance,
    contextEngine,
    tao,
    async run(query: string) {
      // 先跑（contextProvider 检索此前记忆），再记录本轮 → 避免检索命中当前输入本身。
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
      result.finalAnswer = appendSourcesFooter(result.finalAnswer, result.trajectory);
      memory.recordUserInput(query);
      memory.recordAssistantOutput(result.finalAnswer);
      return result;
    },
  };
};
