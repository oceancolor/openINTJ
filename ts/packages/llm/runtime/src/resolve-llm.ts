import { HunyuanClient } from "@openintj/llm-hunyuan";
import { OllamaClient } from "@openintj/llm-ollama";
import {
  ModelRuntimeError,
  type ModelRuntimeErrorCode,
  runtimeErrorInfo,
  sanitizeModelRuntimeErrorMessage,
} from "./errors.js";
import { hasHunyuanCredentials, probeOllama } from "./health.js";
import { MockLlmClient } from "./mock-llm.js";
import {
  OPENAI_PROVIDER_PROFILES,
  type OpenAICloudProviderId,
  createOpenAIProviderClient,
  hasOpenAIProviderCredentials,
  loadOpenAIProviderConfig,
} from "./openai-providers.js";
import { StrictLlmWrapper } from "./strict-wrapper.js";
import type {
  LlmProviderId,
  LlmRuntimeStatus,
  ProviderAttempt,
  ResolveLlmOpts,
  ResolvedLlm,
} from "./types.js";

const parseLlmProvider = (env: NodeJS.ProcessEnv, opts?: LlmProviderId): LlmProviderId => {
  const raw = (opts ?? env["LLM_PROVIDER"] ?? "auto").trim().toLowerCase();
  if (
    raw === "ollama" ||
    raw === "hunyuan" ||
    raw === "kimi" ||
    raw === "minimax" ||
    raw === "glm" ||
    raw === "mock" ||
    raw === "auto"
  ) {
    return raw;
  }
  return "auto";
};

const ollamaBase = (env: NodeJS.ProcessEnv): string =>
  env["OLLAMA_BASE_URL"]?.trim() || "http://127.0.0.1:11434";

const ollamaModel = (env: NodeJS.ProcessEnv): string => env["OLLAMA_MODEL"]?.trim() || "qwen2.5:7b";

const probeFailureCode = (reason?: string): ModelRuntimeErrorCode =>
  reason?.startsWith("model_not_installed:") ? "MODEL_NOT_INSTALLED" : "MODEL_PROVIDER_UNAVAILABLE";

const probeOutcome = (reason?: string): ProviderAttempt["outcome"] =>
  reason?.startsWith("model_not_installed:") ? "model_missing" : "unhealthy";

const isOpenAICloudProvider = (provider: LlmProviderId): provider is OpenAICloudProviderId =>
  provider === "kimi" || provider === "minimax" || provider === "glm";

/**
 * 统一 LLM 解析（ADR-002）：auto 本地优先；显式 ollama/hunyuan fail closed（StrictLlmWrapper）。
 */
export const resolveLlmClient = async (opts: ResolveLlmOpts = {}): Promise<ResolvedLlm> => {
  const env = opts.env ?? process.env;
  const now = opts.now ?? Date.now;
  const requested = parseLlmProvider(env, opts.provider);
  const attempts: ProviderAttempt[] = [];

  const finish = (
    client: import("@openintj/core").LlmClient,
    status: Omit<LlmRuntimeStatus, "requestedProvider" | "attempts">,
  ): ResolvedLlm => ({
    client,
    status: { requestedProvider: requested, attempts, ...status },
  });

  if (requested === "mock") {
    attempts.push({ provider: "mock", outcome: "selected", durationMs: 0, ok: true });
    return finish(new MockLlmClient(), {
      provider: "mock",
      model: "mock-template",
      mode: "mock",
      status: "connected",
    });
  }

  if (requested === "ollama") {
    const startedAt = now();
    const probe = await probeOllama(ollamaBase(env), 3000, ollamaModel(env), opts.fetch);
    const errorCode = probe.ok ? undefined : probeFailureCode(probe.reason);
    const errorMessage = probe.reason ? sanitizeModelRuntimeErrorMessage(probe.reason) : undefined;
    attempts.push({
      provider: "ollama",
      outcome: probe.ok ? "selected" : probeOutcome(probe.reason),
      durationMs: Math.max(0, now() - startedAt),
      ok: probe.ok,
      ...(probe.reason ? { reason: probe.reason } : {}),
      ...(errorCode ? { errorCode } : {}),
      ...(errorMessage ? { errorMessage } : {}),
    });
    if (!probe.ok) {
      throw new ModelRuntimeError({
        code: probeFailureCode(probe.reason),
        message: `LLM provider 'ollama' 不可用 (${probe.reason ?? "unknown"})。请启动 ollama serve、安装所需模型或改用 LLM_PROVIDER=auto。`,
        retriable: true,
        provider: "ollama",
      });
    }
    const inner = OllamaClient.fromEnv(env);
    return finish(new StrictLlmWrapper(inner, "Ollama"), {
      provider: "ollama",
      model: ollamaModel(env),
      mode: "live",
      status: "connected",
    });
  }

  if (requested === "hunyuan") {
    if (!hasHunyuanCredentials(env)) {
      attempts.push({
        provider: "hunyuan",
        outcome: "ineligible",
        durationMs: 0,
        errorCode: "MODEL_CREDENTIAL_MISSING",
        errorMessage: "missing_api_key",
        ok: false,
        reason: "missing_api_key",
      });
      throw new ModelRuntimeError({
        code: "MODEL_CREDENTIAL_MISSING",
        message: "LLM provider 'hunyuan' 需要 HUNYUAN_API_KEY。",
        retriable: false,
        provider: "hunyuan",
      });
    }
    attempts.push({ provider: "hunyuan", outcome: "selected", durationMs: 0, ok: true });
    const inner = HunyuanClient.fromEnv(env);
    return finish(new StrictLlmWrapper(inner, "Hunyuan"), {
      provider: "hunyuan",
      model: inner.config.model,
      mode: "live",
      status: "connected",
    });
  }

  if (isOpenAICloudProvider(requested)) {
    const profile = OPENAI_PROVIDER_PROFILES[requested];
    if (!hasOpenAIProviderCredentials(requested, env)) {
      attempts.push({
        provider: requested,
        outcome: "ineligible",
        durationMs: 0,
        errorCode: "MODEL_CREDENTIAL_MISSING",
        errorMessage: "missing_api_key",
        ok: false,
        reason: "missing_api_key",
      });
      throw new ModelRuntimeError({
        code: "MODEL_CREDENTIAL_MISSING",
        message: `LLM provider '${requested}' 需要 ${profile.apiKeyEnv}。`,
        retriable: false,
        provider: requested,
      });
    }
    const config = loadOpenAIProviderConfig(requested, env, opts.fetch);
    attempts.push({ provider: requested, outcome: "selected", durationMs: 0, ok: true });
    return finish(
      new StrictLlmWrapper(createOpenAIProviderClient(requested, env, opts.fetch), profile.label),
      {
        provider: requested,
        model: config.model,
        mode: "live",
        status: "connected",
      },
    );
  }

  // auto: Ollama 健康 → Hunyuan 有 key → 可见 mock
  const startedAt = now();
  const probe = await probeOllama(ollamaBase(env), 3000, ollamaModel(env), opts.fetch);
  const probeCode = probe.ok ? undefined : probeFailureCode(probe.reason);
  const probeMessage = probe.reason ? sanitizeModelRuntimeErrorMessage(probe.reason) : undefined;
  attempts.push({
    provider: "ollama",
    outcome: probe.ok ? "selected" : probeOutcome(probe.reason),
    durationMs: Math.max(0, now() - startedAt),
    ok: probe.ok,
    ...(probe.reason ? { reason: probe.reason } : {}),
    ...(probeCode ? { errorCode: probeCode } : {}),
    ...(probeMessage ? { errorMessage: probeMessage } : {}),
  });
  if (probe.ok) {
    const inner = OllamaClient.fromEnv(env);
    return finish(new StrictLlmWrapper(inner, "Ollama"), {
      provider: "ollama",
      model: ollamaModel(env),
      mode: "live",
      status: "connected",
    });
  }

  if (hasHunyuanCredentials(env)) {
    attempts.push({ provider: "hunyuan", outcome: "selected", durationMs: 0, ok: true });
    const inner = HunyuanClient.fromEnv(env);
    return finish(new StrictLlmWrapper(inner, "Hunyuan"), {
      provider: "hunyuan",
      model: inner.config.model,
      mode: "live",
      status: "connected",
      fallbackFrom: "ollama",
      ...(probe.reason
        ? {
            lastError: runtimeErrorInfo(
              new ModelRuntimeError({
                code: probeFailureCode(probe.reason),
                message: probe.reason,
                retriable: true,
                provider: "ollama",
              }),
              now(),
            ),
          }
        : {}),
    });
  }

  attempts.push({
    provider: "mock",
    outcome: "selected",
    durationMs: 0,
    ok: true,
    reason: "no_healthy_provider",
  });
  const fallbackError = new ModelRuntimeError({
    code: "MODEL_PROVIDER_UNAVAILABLE",
    message: probe.reason ?? "no cloud credentials",
    retriable: true,
    provider: "ollama",
  });
  return finish(new MockLlmClient(), {
    provider: "mock",
    model: "mock-template",
    mode: "mock",
    status: "degraded",
    fallbackFrom: "ollama",
    lastError: runtimeErrorInfo(fallbackError, now()),
  });
};

/** 同步快捷路径：仅 mock / 不探测（单测用）。生产路径请用 resolveLlmClient。 */
export const resolveLlmClientSync = (opts: ResolveLlmOpts = {}): ResolvedLlm => {
  const env = opts.env ?? process.env;
  const requested = parseLlmProvider(env, opts.provider);
  if (requested === "mock" || requested === "auto") {
    return {
      client: new MockLlmClient(),
      status: {
        requestedProvider: requested,
        provider: "mock",
        model: "mock-template",
        mode: requested === "mock" ? "mock" : "fallback",
        status: "connected",
        attempts: [{ provider: "mock", outcome: "selected", durationMs: 0, ok: true }],
      },
    };
  }
  if (requested === "hunyuan") {
    const inner = HunyuanClient.fromEnv(env);
    return {
      client: new StrictLlmWrapper(inner, "Hunyuan"),
      status: {
        requestedProvider: requested,
        provider: "hunyuan",
        model: inner.config.model,
        mode: "live",
        status: inner.getStatus().status,
        attempts: [{ provider: "hunyuan", outcome: "selected", durationMs: 0, ok: true }],
      },
    };
  }
  if (isOpenAICloudProvider(requested)) {
    const profile = OPENAI_PROVIDER_PROFILES[requested];
    const inner = createOpenAIProviderClient(requested, env, opts.fetch);
    return {
      client: new StrictLlmWrapper(inner, profile.label),
      status: {
        requestedProvider: requested,
        provider: requested,
        model: loadOpenAIProviderConfig(requested, env).model,
        mode: "live",
        status: inner.getStatus().status,
        attempts: [{ provider: requested, outcome: "selected", durationMs: 0, ok: true }],
      },
    };
  }
  const inner = OllamaClient.fromEnv(env);
  return {
    client: new StrictLlmWrapper(inner, "Ollama"),
    status: {
      requestedProvider: requested,
      provider: "ollama",
      model: ollamaModel(env),
      mode: "live",
      status: "connected",
      attempts: [{ provider: "ollama", outcome: "selected", durationMs: 0, ok: true }],
    },
  };
};

export { parseLlmProvider };
