import { OpenAICompatibleClient, type OpenAICompatibleConfig } from "@openintj/llm-openai-compat";
import type { LlmProviderId } from "./types.js";

export type OpenAICloudProviderId = Extract<LlmProviderId, "kimi" | "minimax" | "glm">;

export interface OpenAIProviderProfile {
  id: OpenAICloudProviderId;
  label: string;
  apiKeyEnv: string;
  apiKeyAliases: readonly string[];
  baseUrlEnv: string;
  modelEnv: string;
  defaultBaseUrl: string;
  defaultModel: string;
}

export const OPENAI_PROVIDER_PROFILES: Record<OpenAICloudProviderId, OpenAIProviderProfile> = {
  kimi: {
    id: "kimi",
    label: "Kimi",
    apiKeyEnv: "KIMI_API_KEY",
    apiKeyAliases: ["MOONSHOT_API_KEY"],
    baseUrlEnv: "KIMI_BASE_URL",
    modelEnv: "KIMI_MODEL",
    defaultBaseUrl: "https://api.moonshot.ai/v1",
    defaultModel: "kimi-k3",
  },
  minimax: {
    id: "minimax",
    label: "MiniMax",
    apiKeyEnv: "MINIMAX_API_KEY",
    apiKeyAliases: [],
    baseUrlEnv: "MINIMAX_BASE_URL",
    modelEnv: "MINIMAX_MODEL",
    defaultBaseUrl: "https://api.minimax.io/v1",
    defaultModel: "MiniMax-M3",
  },
  glm: {
    id: "glm",
    label: "GLM",
    apiKeyEnv: "GLM_API_KEY",
    apiKeyAliases: ["ZAI_API_KEY", "ZHIPUAI_API_KEY"],
    baseUrlEnv: "GLM_BASE_URL",
    modelEnv: "GLM_MODEL",
    defaultBaseUrl: "https://api.z.ai/api/paas/v4",
    defaultModel: "glm-5.2",
  },
};

const firstEnv = (env: NodeJS.ProcessEnv, names: readonly string[]): string =>
  names.map((name) => env[name]?.trim()).find(Boolean) ?? "";

export const loadOpenAIProviderConfig = (
  provider: OpenAICloudProviderId,
  env: NodeJS.ProcessEnv = process.env,
  fetch?: typeof globalThis.fetch,
): OpenAICompatibleConfig => {
  const profile = OPENAI_PROVIDER_PROFILES[provider];
  return {
    provider,
    apiKey: firstEnv(env, [profile.apiKeyEnv, ...profile.apiKeyAliases]),
    baseUrl: env[profile.baseUrlEnv]?.trim() || profile.defaultBaseUrl,
    model: env[profile.modelEnv]?.trim() || profile.defaultModel,
    timeoutMs: env[`${provider.toUpperCase()}_TIMEOUT_MS`]
      ? Number.parseInt(env[`${provider.toUpperCase()}_TIMEOUT_MS`] ?? "", 10)
      : 60_000,
    ...(fetch ? { fetch } : {}),
  };
};

export const hasOpenAIProviderCredentials = (
  provider: OpenAICloudProviderId,
  env: NodeJS.ProcessEnv,
): boolean => Boolean(loadOpenAIProviderConfig(provider, env).apiKey);

export const createOpenAIProviderClient = (
  provider: OpenAICloudProviderId,
  env: NodeJS.ProcessEnv = process.env,
  fetch?: typeof globalThis.fetch,
): OpenAICompatibleClient =>
  new OpenAICompatibleClient(loadOpenAIProviderConfig(provider, env, fetch));
