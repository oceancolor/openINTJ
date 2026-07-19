import { SimpleEmbedder } from "@openintj/core";
import { OllamaEmbedder, loadOllamaEmbedderConfigFromEnv } from "@openintj/embed-ollama";
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
    attempts.push({ provider: "simple", ok: true });
    return finish(new SimpleEmbedder(dim), {
      provider: "simple-sha256",
      model: `dim${dim}`,
      dimension: dim,
      mode: requested === "mock" ? "mock" : "live",
    });
  }

  if (requested === "xenova") {
    try {
      const { XenovaEmbedder } = (await import("@openintj/embed-xenova")) as {
        XenovaEmbedder: new () => import("@openintj/core").EmbeddingProvider;
      };
      const embedder = new XenovaEmbedder();
      await embedder.embed("warmup");
      const dim = embedder.dimension;
      attempts.push({ provider: "xenova", ok: true });
      return finish(embedder, {
        provider: embedder.name,
        model: "xenova",
        dimension: dim,
        mode: "live",
      });
    } catch (e) {
      attempts.push({
        provider: "xenova",
        ok: false,
        reason: e instanceof Error ? e.message : String(e),
      });
      throw new Error(`EMBED_PROVIDER=xenova 失败: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  if (requested === "ollama") {
    const cfg = loadOllamaEmbedderConfigFromEnv(env);
    const probe = await probeOllama(cfg.endpoint, 3000, cfg.model, opts.fetch);
    attempts.push({
      provider: "ollama",
      ok: probe.ok,
      ...(probe.reason ? { reason: probe.reason } : {}),
    });
    if (!probe.ok) {
      throw new Error(`Embed provider 'ollama' 不可达: ${probe.reason ?? "unknown"}`);
    }
    const embedder = new OllamaEmbedder(cfg);
    await embedder.embed("warmup");
    const dim = embedder.dimension;
    if (dim <= 0) throw new Error("OllamaEmbedder: could not infer dimension");
    attempts.push({ provider: "ollama-embed", ok: true });
    return finish(embedder, {
      provider: "ollama",
      model: cfg.model,
      dimension: dim,
      mode: "live",
    });
  }

  // auto
  const cfg = loadOllamaEmbedderConfigFromEnv(env);
  const probe = await probeOllama(cfg.endpoint, 3000, cfg.model, opts.fetch);
  attempts.push({
    provider: "ollama",
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
      });
    } catch (e) {
      attempts.push({
        provider: "ollama-embed",
        ok: false,
        reason: e instanceof Error ? e.message : String(e),
      });
    }
  }

  const dim = simpleDim(env);
  attempts.push({ provider: "simple", ok: true, reason: "ollama_unavailable" });
  return finish(new SimpleEmbedder(dim), {
    provider: "simple-sha256",
    model: `dim${dim}`,
    dimension: dim,
    mode: "mock",
    fallbackFrom: "ollama",
    ...(probe.reason ? { lastError: probe.reason } : {}),
  });
};

export { parseEmbedProvider };
