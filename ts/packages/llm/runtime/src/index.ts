export * from "./types.js";
export * from "./mock-llm.js";
export * from "./strict-wrapper.js";
export * from "./health.js";
export * from "./resolve-llm.js";
export * from "./resolve-embedder.js";
export * from "./embedding-fingerprint.js";
export * from "./embedding-persistence.js";
export * from "./errors.js";

import { loadOllamaEmbedderConfigFromEnv } from "@openintj/embed-ollama";
import { ModelRuntimeError, runtimeErrorInfo } from "./errors.js";
import { hasHunyuanCredentials, probeOllama } from "./health.js";
import { resolveEmbedder } from "./resolve-embedder.js";
import { resolveLlmClient } from "./resolve-llm.js";
import type {
  EmbeddingFingerprint,
  ModelRuntime,
  ModelRuntimeStatus,
  ProviderAttempt,
  ResolveModelRuntimeOpts,
} from "./types.js";

const pushAttempt = (attempts: ProviderAttempt[], attempt: ProviderAttempt): void => {
  attempts.push(attempt);
  if (attempts.length > 20) attempts.splice(0, attempts.length - 20);
};

const asRuntimeError = (
  error: unknown,
  provider: string,
  code: "MODEL_PROVIDER_UNAVAILABLE" | "MODEL_REQUEST_FAILED" = "MODEL_REQUEST_FAILED",
): ModelRuntimeError =>
  error instanceof ModelRuntimeError
    ? error
    : new ModelRuntimeError({
        code,
        message: error instanceof Error ? error.message : String(error),
        retriable: true,
        provider,
        cause: error,
      });

/** 同时解析 LLM 与 embedding（三端装配入口）。 */
export const resolveModelRuntime = async (
  opts: ResolveModelRuntimeOpts = {},
): Promise<ModelRuntime> => {
  const env = opts.env ?? process.env;
  const now = opts.now ?? Date.now;
  const hooks = opts.hooks;
  const emitError = async (
    channel: "llm" | "embedding",
    error: ModelRuntimeError,
  ): Promise<void> => {
    await hooks?.emit("model.provider.error", {
      channel,
      provider: error.provider ?? "unknown",
      code: error.code,
      message: error.message,
      retriable: error.retriable,
    });
  };

  let llm: Awaited<ReturnType<typeof resolveLlmClient>>;
  try {
    llm = await resolveLlmClient({
      ...(opts.provider !== undefined ? { provider: opts.provider } : {}),
      env,
      ...(opts.fetch !== undefined ? { fetch: opts.fetch } : {}),
      now,
    });
  } catch (error) {
    const structured = asRuntimeError(error, opts.provider ?? env["LLM_PROVIDER"] ?? "auto");
    if (structured.provider === "ollama") {
      await hooks?.emit("model.provider.probe", {
        channel: "llm",
        provider: "ollama",
        model: env["OLLAMA_MODEL"]?.trim() || "qwen2.5:7b",
        ok: false,
        durationMs: 0,
        errorCode: structured.code,
      });
    }
    await emitError("llm", structured);
    throw structured;
  }

  let embed: Awaited<ReturnType<typeof resolveEmbedder>>;
  try {
    embed = await resolveEmbedder({
      ...(opts.embedProvider !== undefined ? { embedProvider: opts.embedProvider } : {}),
      env,
      ...(opts.fetch !== undefined ? { fetch: opts.fetch } : {}),
      now,
    });
  } catch (error) {
    const structured = asRuntimeError(error, opts.embedProvider ?? env["EMBED_PROVIDER"] ?? "auto");
    if (structured.provider === "ollama") {
      await hooks?.emit("model.provider.probe", {
        channel: "embedding",
        provider: "ollama",
        model: env["OLLAMA_EMBED_MODEL"]?.trim() || "nomic-embed-text",
        ok: false,
        durationMs: 0,
        errorCode: structured.code,
      });
    }
    await emitError("embedding", structured);
    throw structured;
  }

  const embeddingFingerprint: EmbeddingFingerprint = {
    schemaVersion: 1,
    provider: embed.status.provider,
    model: embed.status.model,
    dimension: embed.dimension,
  };
  const status: ModelRuntimeStatus = { llm: llm.status, embed: embed.status };

  const emitResolution = async (
    channel: "llm" | "embedding",
    channelStatus: typeof status.llm | typeof status.embed,
  ): Promise<void> => {
    for (const attempt of channelStatus.attempts) {
      if (attempt.provider === "ollama" || attempt.provider === "xenova") {
        await hooks?.emit("model.provider.probe", {
          channel,
          provider: attempt.provider,
          model: channelStatus.model,
          ok: attempt.ok,
          durationMs: attempt.durationMs,
          ...(attempt.errorCode ? { errorCode: attempt.errorCode } : {}),
        });
      }
    }
    await hooks?.emit("model.provider.selected", {
      channel,
      requestedProvider: channelStatus.requestedProvider,
      provider: channelStatus.provider,
      model: channelStatus.model,
      mode: channelStatus.mode,
    });
    if (channelStatus.fallbackFrom) {
      await hooks?.emit("model.provider.fallback", {
        channel,
        from: channelStatus.fallbackFrom,
        to: channelStatus.provider,
        errorCode: channelStatus.lastError?.code ?? "MODEL_PROVIDER_UNAVAILABLE",
      });
    }
  };
  await emitResolution("llm", status.llm);
  await emitResolution("embedding", status.embed);

  const refreshIntervalMs = Math.max(0, opts.healthRefreshIntervalMs ?? 30_000);
  let lastRefreshAt = now();
  let refreshInFlight: Promise<ModelRuntimeStatus> | undefined;

  const refreshOllama = async (
    channel: "llm" | "embedding",
    channelStatus: typeof status.llm | typeof status.embed,
    endpoint: string,
    model: string,
  ): Promise<void> => {
    const startedAt = now();
    const probe = await probeOllama(endpoint, 3000, model, opts.fetch);
    const durationMs = Math.max(0, now() - startedAt);
    const code = probe.reason?.startsWith("model_not_installed:")
      ? "MODEL_NOT_INSTALLED"
      : "MODEL_PROVIDER_UNAVAILABLE";
    pushAttempt(channelStatus.attempts, {
      provider: "ollama",
      outcome: probe.ok
        ? "selected"
        : code === "MODEL_NOT_INSTALLED"
          ? "model_missing"
          : "unhealthy",
      durationMs,
      ok: probe.ok,
      ...(probe.reason ? { reason: probe.reason, errorMessage: probe.reason } : {}),
      ...(!probe.ok ? { errorCode: code } : {}),
    });
    await hooks?.emit("model.provider.probe", {
      channel,
      provider: "ollama",
      model,
      ok: probe.ok,
      durationMs,
      ...(!probe.ok ? { errorCode: code } : {}),
    });
    if (probe.ok) {
      channelStatus.status = "connected";
      delete channelStatus.lastError;
      return;
    }
    channelStatus.status = "degraded";
    const error = new ModelRuntimeError({
      code,
      message: probe.reason ?? "Ollama health probe failed",
      retriable: true,
      provider: "ollama",
    });
    channelStatus.lastError = runtimeErrorInfo(error, now());
    await emitError(channel, error);
  };

  const runtime: ModelRuntime = {
    llm,
    embed,
    embeddingFingerprint,
    status,
    getStatus: () => status,
    refreshHealth: async () => {
      if (refreshInFlight) return refreshInFlight;
      if (now() - lastRefreshAt < refreshIntervalMs) return status;
      refreshInFlight = (async () => {
        if (status.llm.provider === "ollama") {
          await refreshOllama(
            "llm",
            status.llm,
            env["OLLAMA_BASE_URL"]?.trim() || "http://127.0.0.1:11434",
            status.llm.model,
          );
        } else if (status.llm.provider === "hunyuan") {
          const providerStatus = llm.client.getStatus();
          if (!hasHunyuanCredentials(env) || !providerStatus.available) {
            const error = new ModelRuntimeError({
              code: hasHunyuanCredentials(env)
                ? "MODEL_PROVIDER_UNAVAILABLE"
                : "MODEL_CREDENTIAL_MISSING",
              message: providerStatus.lastError ?? "Hunyuan provider unavailable",
              retriable: hasHunyuanCredentials(env),
              provider: "hunyuan",
            });
            status.llm.status = providerStatus.status;
            status.llm.lastError = runtimeErrorInfo(error, now());
            await emitError("llm", error);
          } else {
            status.llm.status = "connected";
            delete status.llm.lastError;
          }
        }
        if (status.embed.provider === "ollama") {
          const cfg = loadOllamaEmbedderConfigFromEnv(env);
          await refreshOllama("embedding", status.embed, cfg.endpoint, status.embed.model);
        }
        lastRefreshAt = now();
        return status;
      })().finally(() => {
        refreshInFlight = undefined;
      });
      return refreshInFlight;
    },
    close: async () => {},
  };
  return runtime;
};
