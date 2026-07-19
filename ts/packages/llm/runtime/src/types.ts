import type { EmbeddingProvider, LlmClient } from "@openintj/core";

/** LLM provider 标识（requested / actual）。 */
export type LlmProviderId = "auto" | "ollama" | "hunyuan" | "mock";

/** Embedding provider 标识。首期无 hunyuan/xenova 在 auto 链中（xenova 可显式选）。 */
export type EmbedProviderId = "auto" | "simple" | "ollama" | "xenova" | "mock";

export type ProviderMode = "live" | "mock" | "fallback";

export interface ProviderAttempt {
  provider: string;
  ok: boolean;
  reason?: string;
}

export interface LlmRuntimeStatus {
  requestedProvider: LlmProviderId;
  provider: string;
  model: string;
  mode: ProviderMode;
  status: "connected" | "degraded" | "unauthorized" | "missing_api_key";
  fallbackFrom?: string;
  lastError?: string;
  attempts: ProviderAttempt[];
}

export interface EmbedRuntimeStatus {
  requestedProvider: EmbedProviderId;
  provider: string;
  model: string;
  dimension: number;
  mode: ProviderMode;
  fallbackFrom?: string;
  lastError?: string;
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
}

export interface ResolveEmbedOpts {
  /** Embedding provider（与 LLM provider 独立）。 */
  embedProvider?: EmbedProviderId;
  /** @deprecated 使用 embedProvider */
  provider?: EmbedProviderId;
  env?: NodeJS.ProcessEnv;
  fetch?: typeof globalThis.fetch;
}

export interface ResolveModelRuntimeOpts extends ResolveLlmOpts {
  embedProvider?: EmbedProviderId;
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

export const EMBEDDING_FINGERPRINT_FILENAME = "embedding-fingerprint.json";
