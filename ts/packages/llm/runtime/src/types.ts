import type { EmbeddingProvider, HookBus, LlmClient } from "@openintj/core";
import type { ModelRuntimeErrorCode } from "./errors.js";

/** LLM provider 标识（requested / actual）。 */
export type LlmProviderId = "auto" | "ollama" | "hunyuan" | "mock";

/** Embedding provider 标识。首期无 hunyuan/xenova 在 auto 链中（xenova 可显式选）。 */
export type EmbedProviderId = "auto" | "simple" | "ollama" | "xenova" | "mock";

export type ProviderMode = "live" | "mock" | "fallback";

export interface ProviderAttempt {
  provider: string;
  outcome: "selected" | "unhealthy" | "model_missing" | "ineligible" | "failed";
  durationMs: number;
  errorCode?: ModelRuntimeErrorCode;
  errorMessage?: string;
  /** @deprecated Use outcome. Retained for status wire compatibility. */
  ok: boolean;
  /** @deprecated Use errorMessage. Retained for status wire compatibility. */
  reason?: string;
}

export interface RuntimeErrorInfo {
  code: ModelRuntimeErrorCode;
  message: string;
  retriable: boolean;
  at: number;
}

export interface LlmRuntimeStatus {
  requestedProvider: LlmProviderId;
  provider: string;
  model: string;
  mode: ProviderMode;
  status: "connected" | "degraded" | "unauthorized" | "missing_api_key";
  fallbackFrom?: string;
  lastError?: RuntimeErrorInfo;
  attempts: ProviderAttempt[];
}

export interface EmbedRuntimeStatus {
  requestedProvider: EmbedProviderId;
  provider: string;
  model: string;
  dimension: number;
  mode: ProviderMode;
  fallbackFrom?: string;
  lastError?: RuntimeErrorInfo;
  attempts: ProviderAttempt[];
}

export interface ModelRuntimeStatus {
  llm: LlmRuntimeStatus;
  embed: EmbedRuntimeStatus;
}

export interface ResolveLlmOpts {
  provider?: LlmProviderId;
  env?: NodeJS.ProcessEnv;
  fetch?: typeof globalThis.fetch;
  hooks?: HookBus;
  now?: () => number;
}

export interface ResolveEmbedOpts {
  /** Embedding provider（与 LLM provider 独立）。 */
  embedProvider?: EmbedProviderId;
  /** @deprecated 使用 embedProvider */
  provider?: EmbedProviderId;
  env?: NodeJS.ProcessEnv;
  fetch?: typeof globalThis.fetch;
  now?: () => number;
}

export interface ResolveModelRuntimeOpts extends ResolveLlmOpts {
  embedProvider?: EmbedProviderId;
  /** Minimum interval between network health refreshes. */
  healthRefreshIntervalMs?: number;
}

export interface ResolvedLlm {
  client: LlmClient;
  status: LlmRuntimeStatus;
}

export interface ResolvedEmbed {
  embedder: EmbeddingProvider;
  status: EmbedRuntimeStatus;
  dimension: number;
}

/** 持久化向量空间的身份指纹（provider + model + dimension）。 */
export interface EmbeddingFingerprint {
  schemaVersion: 1;
  provider: string;
  model: string;
  dimension: number;
}

export interface ModelRuntime {
  readonly llm: ResolvedLlm;
  readonly embed: ResolvedEmbed;
  readonly embeddingFingerprint: EmbeddingFingerprint;
  readonly status: ModelRuntimeStatus;
  getStatus(): ModelRuntimeStatus;
  refreshHealth(): Promise<ModelRuntimeStatus>;
  close(): Promise<void>;
}

export const EMBEDDING_FINGERPRINT_FILENAME = "embedding-fingerprint.json";
