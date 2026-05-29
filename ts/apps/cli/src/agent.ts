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
import { HunyuanClient, createHunyuanSearchTool } from "@openintj/llm-hunyuan";
import { OllamaClient } from "@openintj/llm-ollama";
import { ControlPlane } from "@openintj/plane-control";
import { Executor, ToolHub } from "@openintj/plane-execution";
import { GovernancePlane } from "@openintj/plane-governance";
import { ContextEngine, MemoryPlane } from "@openintj/plane-memory";
import { DEFAULT_AGENT_SYSTEM_PROMPT, appendSourcesFooter } from "@openintj/shared";

export type LlmProvider = "auto" | "hunyuan" | "ollama" | "mock";

export interface AgentOptions {
  llmProvider?: LlmProvider;
  systemPrompt?: string;
  /** 注入自定义工具实现（默认是 4 个内置无副作用占位 handler）。 */
  toolHandlers?: {
    readFile?: (params: Record<string, unknown>) => unknown;
    writeFile?: (params: Record<string, unknown>) => unknown;
    executeCommand?: (params: Record<string, unknown>) => unknown;
    search?: (params: Record<string, unknown>) => unknown;
  };
  /** TAO 多轮上限（默认 1 = 单轮兼容 v2.0）。 */
  maxTaoIterations?: number;
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

  // 默认 handler：placeholder（不读真实磁盘 / 不执行命令；CLI demo 安全）
  const defaultReadFile = (params: Record<string, unknown>) => ({
    note: "[mock read_file]",
    path: params["path"],
  });
  const defaultWriteFile = (params: Record<string, unknown>) => ({
    note: "[mock write_file]",
    path: params["path"],
    bytes: typeof params["content"] === "string" ? (params["content"] as string).length : 0,
  });
  const defaultExecuteCommand = (params: Record<string, unknown>) => ({
    note: "[mock execute_command]",
    command: params["command"],
  });
  const defaultSearch = (params: Record<string, unknown>) => ({
    note: "[mock search]",
    query: params["query"],
    hits: [],
  });

  // search 默认接混元联网搜索（llm 为混元时）；否则退回无副作用占位。
  const defaultSearchHandler =
    llm instanceof HunyuanClient ? createHunyuanSearchTool(llm) : defaultSearch;
  toolHub.registerBuiltinTools({
    readFile: handlers.readFile ?? defaultReadFile,
    writeFile: handlers.writeFile ?? defaultWriteFile,
    executeCommand: handlers.executeCommand ?? defaultExecuteCommand,
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
  const tao = new TaoLoop({
    config: { ...DEFAULT_TAO_CONFIG, maxTaoIterations: opts.maxTaoIterations ?? 1 },
    hooks,
    react,
    availableTools: () => toolHub.list(),
    systemPrompt: opts.systemPrompt ?? DEFAULT_AGENT_SYSTEM_PROMPT,
  });

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
      memory.recordUserInput(query);
      const result = await tao.run(query);
      result.finalAnswer = appendSourcesFooter(result.finalAnswer, result.trajectory);
      memory.recordAssistantOutput(result.finalAnswer);
      return result;
    },
  };
};
