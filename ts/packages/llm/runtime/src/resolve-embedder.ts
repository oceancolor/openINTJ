import { SimpleEmbedder } from "@openintj/core";
import { OllamaEmbedder, loadOllamaEmbedderConfigFromEnv } from "@openintj/embed-ollama";
import { ModelRuntimeError, runtimeErrorInfo, sanitizeModelRuntimeErrorMessage } from "./errors.js";
import { probeOllama } from "./health.js";
import type {
  EmbedProviderId,
  EmbedRuntimeStatus,
  ProviderAttempt,
  ResolveEmbedOpts,
  ResolvedEmbed,
} from "./types.js";

const parseEmbedProvider = (env: NodeJS.ProcessEnv, opts?: ResolveEmbedOpts): EmbedProviderId => {
  const raw = (
    opts?.embedProvider ??
    opts?.provider ??
    env["EMBEDDING_PROVIDER"] ??
    env["EMBED_PROVIDER"] ??
    "auto"
  )
    .trim()
    .toLowerCase();
  if (
    raw === "simple" ||
    raw === "ollama" ||
    raw === "xenova" ||
    raw === "mock" ||
    raw === "auto"
  ) {
    return raw as EmbedProviderId;
  }
  return "auto";
};

const simpleDim = (env: NodeJS.ProcessEnv): number => {
  const n = Number(env["OPENINTJ_EMBED_DIM"] ?? env["EMBEDDING_DIM"] ?? "64");
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 64;
};

/**
 * 统一 embedding 解析。auto：Ollama 健康 → simple（可见 mock 语义）。
 * xenova 需显式选择（可选 peer dep，动态 import）。
 */
export const resolveEmbedder = async (opts: ResolveEmbedOpts = {}): Promise<ResolvedEmbed> => {
  const env = opts.env ?? process.env;
  const now = opts.now ?? Date.now;
  const requested = parseEmbedProvider(env, opts);
  const attempts: ProviderAttempt[] = [];

  const finish = (
    embedder: import("@openintj/core").EmbeddingProvider,
    status: Omit<EmbedRuntimeStatus, "requestedProvider" | "attempts">,
  ): ResolvedEmbed => ({
    embedder,
    dimension: status.dimension,
    status: { requestedProvider: requested, attempts, ...status },
  });

  if (requested === "simple" || requested === "mock") {
    const dim = simpleDim(env);
    attempts.push({ provider: "simple", outcome: "selected", durationMs: 0, ok: true });
    return finish(new SimpleEmbedder(dim), {
      provider: "simple-sha256",
      model: `dim${dim}`,
      dimension: dim,
      mode: requested === "mock" ? "mock" : "live",
      status: "connected",
    });
  }

  if (requested === "xenova") {
    const startedAt = now();
    try {
      const { XenovaEmbedder } = (await import("@openintj/embed-xenova")) as {
        XenovaEmbedder: new () => import("@openintj/core").EmbeddingProvider;
      };
      const embedder = new XenovaEmbedder();
      await embedder.embed("warmup");
      const dim = embedder.dimension;
      attempts.push({
        provider: "xenova",
        outcome: "selected",
        durationMs: Math.max(0, now() - startedAt),
        ok: true,
      });
      return finish(embedder, {
        provider: embedder.name,
        model: "xenova",
        dimension: dim,
        mode: "live",
        status: "connected",
      });
    } catch (e) {
      const message = sanitizeModelRuntimeErrorMessage(e);
      attempts.push({
        provider: "xenova",
        outcome: "failed",
        durationMs: Math.max(0, now() - startedAt),
        errorCode: "MODEL_PROVIDER_UNAVAILABLE",
        errorMessage: message,
        ok: false,
        reason: message,
      });
      throw new ModelRuntimeError({
        code: "MODEL_PROVIDER_UNAVAILABLE",
        message: `EMBED_PROVIDER=xenova 失败: ${message}`,
        retriable: true,
        provider: "xenova",
        cause: e,
      });
    }
  }

  if (requested === "ollama") {
    const cfg = loadOllamaEmbedderConfigFromEnv(env);
    const startedAt = now();
    const probe = await probeOllama(cfg.endpoint, 3000, cfg.model, opts.fetch);
    const code = probe.reason?.startsWith("model_not_installed:")
      ? "MODEL_NOT_INSTALLED"
      : "MODEL_PROVIDER_UNAVAILABLE";
    attempts.push({
      provider: "ollama",
      outcome: probe.ok
        ? "selected"
        : probe.reason?.startsWith("model_not_installed:")
          ? "model_missing"
          : "unhealthy",
      durationMs: Math.max(0, now() - startedAt),
      ...(!probe.ok ? { errorCode: code } : {}),
      ...(probe.reason ? { errorMessage: sanitizeModelRuntimeErrorMessage(probe.reason) } : {}),
      ok: probe.ok,
      ...(probe.reason ? { reason: probe.reason } : {}),
    });
    if (!probe.ok) {
      throw new ModelRuntimeError({
        code,
        message: `Embed provider 'ollama' 不可用: ${probe.reason ?? "unknown"}`,
        retriable: true,
        provider: "ollama",
      });
    }
    const embedder = new OllamaEmbedder(cfg);
    await embedder.embed("warmup");
    const dim = embedder.dimension;
    if (dim <= 0) {
      throw new ModelRuntimeError({
        code: "EMBEDDING_DIMENSION_UNKNOWN",
        message: "OllamaEmbedder: could not infer dimension",
        retriable: false,
        provider: "ollama",
      });
    }
    return finish(embedder, {
      provider: "ollama",
      model: cfg.model,
      dimension: dim,
      mode: "live",
      status: "connected",
    });
  }

  // auto
  const cfg = loadOllamaEmbedderConfigFromEnv(env);
  const startedAt = now();
  const probe = await probeOllama(cfg.endpoint, 3000, cfg.model, opts.fetch);
  const probeCode = probe.reason?.startsWith("model_not_installed:")
    ? "MODEL_NOT_INSTALLED"
    : "MODEL_PROVIDER_UNAVAILABLE";
  attempts.push({
    provider: "ollama",
    outcome: probe.ok
      ? "selected"
      : probe.reason?.startsWith("model_not_installed:")
        ? "model_missing"
        : "unhealthy",
    durationMs: Math.max(0, now() - startedAt),
    ...(!probe.ok ? { errorCode: probeCode } : {}),
    ...(probe.reason ? { errorMessage: sanitizeModelRuntimeErrorMessage(probe.reason) } : {}),
    ok: probe.ok,
    ...(probe.reason ? { reason: probe.reason } : {}),
  });
  if (probe.ok) {
    try {
      const embedder = new OllamaEmbedder(cfg);
      await embedder.embed("warmup");
      const dim = embedder.dimension;
      return finish(embedder, {
        provider: "ollama",
        model: cfg.model,
        dimension: dim,
        mode: "live",
        status: "connected",
      });
    } catch (e) {
      const message = sanitizeModelRuntimeErrorMessage(e);
      attempts.push({
        provider: "ollama-embed",
        outcome: "failed",
        durationMs: 0,
        errorCode: "MODEL_REQUEST_FAILED",
        errorMessage: message,
        ok: false,
        reason: message,
      });
    }
  }

  const dim = simpleDim(env);
  attempts.push({
    provider: "simple",
    outcome: "selected",
    durationMs: 0,
    ok: true,
    reason: "ollama_unavailable",
  });
  const fallbackError = new ModelRuntimeError({
    code: probeCode,
    message: probe.reason ?? "ollama embedding unavailable",
    retriable: true,
    provider: "ollama",
  });
  return finish(new SimpleEmbedder(dim), {
    provider: "simple-sha256",
    model: `dim${dim}`,
    dimension: dim,
    mode: "mock",
    status: "degraded",
    fallbackFrom: "ollama",
    lastError: runtimeErrorInfo(fallbackError, now()),
  });
};

export { parseEmbedProvider };
