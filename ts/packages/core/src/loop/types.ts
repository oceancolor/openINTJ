import { z } from "zod";
import type { ShaderModeType, TaskTypeType } from "../types/shader.js";
import type { ToolCallResult, ToolDescriptor } from "../types/tool.js";

/** 用户输入图片数据。 */
export const ImagePayloadSchema = z.object({
  base64: z.string(),
  mimeType: z.enum(["image/jpeg", "image/png", "image/gif", "image/webp"]),
  sizeBytes: z.number().int().positive(),
});
export type ImagePayload = z.infer<typeof ImagePayloadSchema>;

// =====================================================
// ReAct 微循环
// =====================================================

export type ReactState =
  | { type: "thought"; content: string; iteration: number }
  | { type: "action"; tool: string; params: unknown; iteration: number }
  | { type: "observation"; toolResult: ToolCallResult; iteration: number }
  | { type: "final"; answer: string };

export type ReactStopCondition =
  | { kind: "explicitFinal" }
  | { kind: "duplicateToolCall"; threshold: number }
  | { kind: "failFast" }
  | { kind: "tokenBudgetExceeded"; maxTokens: number };

export interface ReactConfig {
  maxIterations: number;
  stepTimeoutMs: number;
  stopConditions: ReactStopCondition[];
  observationMaxChars: number;
}

export const DEFAULT_REACT_CONFIG: ReactConfig = {
  maxIterations: 8,
  stepTimeoutMs: 60_000,
  observationMaxChars: 4000,
  stopConditions: [
    { kind: "explicitFinal" },
    { kind: "duplicateToolCall", threshold: 2 },
    { kind: "failFast" },
    { kind: "tokenBudgetExceeded", maxTokens: 16_000 },
  ],
};

export interface TrajectoryEntry {
  timestamp: number;
  state: ReactState;
  durationMs: number;
}

export type ReactStatus = "ok" | "duplicate_loop" | "max_iter" | "fail_fast" | "token_overflow";

export interface ReactInput {
  /** 完整 prompt 消息列表（来自 ContextEngine.window.toPromptMessages）。 */
  messages: ChatMessage[];
  availableTools: ToolDescriptor[];
  taoIteration: number;
  /** 系统提示（可包含记忆注入）。 */
  systemPrompt: string;
}

export interface ReactOutput {
  finalAnswer: string;
  trajectory: TrajectoryEntry[];
  iterations: number;
  status: ReactStatus;
  failedTool?: { tool: string; error: string };
  totalTokensSpent: number;
}

// =====================================================
// TAO 宏循环
// =====================================================

export interface TaoConfig {
  /** TAO 宏循环最大轮数。1 = v2.0 行为；>=2 启用多轮思考。 */
  maxTaoIterations: number;
  /** 单次 TAO run 总超时（毫秒）。 */
  taoTimeoutMs: number;
  /** 是否启用 ReAct 微循环。false 时退化为单次 LLM 调用。 */
  enableReact: boolean;
  /** 内嵌 ReAct 配置。 */
  react: ReactConfig;
}

export const DEFAULT_TAO_CONFIG: TaoConfig = {
  maxTaoIterations: 1,
  taoTimeoutMs: 5 * 60 * 1000,
  enableReact: true,
  react: DEFAULT_REACT_CONFIG,
};

export type TaoStatus = "completed" | "failed" | "timeout" | "max_iter_reached";

export interface TaoResult {
  traceId: string;
  status: TaoStatus;
  finalAnswer: string;
  iterations: number;
  reactTotalSteps: number;
  durationMs: number;
  trajectory: TrajectoryEntry[];
  taskType: TaskTypeType;
  shaderMode: ShaderModeType;
  metrics: Record<string, number>;
  failureReason?: string;
}

export interface TaoContext {
  readonly traceId: string;
  readonly query: string;
  readonly imageData?: ImagePayload;
  iteration: number;
  trajectory: TrajectoryEntry[];
  finalAnswer?: string;
  metrics: Record<string, number>;
}

// =====================================================
// 通用：聊天消息（兼容 OpenAI 格式）
// =====================================================

export type ChatMessageContent =
  | string
  | Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }>;

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: ChatMessageContent;
  /** 关联的工具调用 id（OpenAI tool_calls 协议）。 */
  toolCallId?: string;
  /** 关联的工具名（assistant 消息中的 tool_call 引用）。 */
  name?: string;
  /** 元数据（不发送到 LLM）。 */
  metadata?: Record<string, unknown>;
}

// =====================================================
// LLM 适配器接口（实现见 @openintj/llm-*）
// =====================================================

export interface LlmStatus {
  provider: string;
  model: string;
  visionModel?: string;
  baseUrl?: string;
  available: boolean;
  mode: "live" | "mock" | "unauthorized";
  status: "connected" | "degraded" | "unauthorized" | "missing_api_key";
  lastError?: string;
  lastErrorCode?: string;
  lastErrorType?: string;
  visionSupported: boolean;
}

export interface LlmClient {
  chat(messages: ChatMessage[], opts?: ChatOptions): Promise<string>;
  visionChat(
    messages: ChatMessage[],
    image: { base64: string; mimeType: string },
    opts?: ChatOptions,
  ): Promise<string>;
  getStatus(): LlmStatus;
}

export interface ChatOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  /** 工具调用支持（OpenAI function calling）。 */
  tools?: ToolDescriptor[];
  /** 强制返回 final answer（用于 ReAct 收敛）。 */
  forceFinal?: boolean;
}
