import { HunyuanClient } from "@openintj/llm-hunyuan";
import { OllamaClient } from "@openintj/llm-ollama";
import { hasHunyuanCredentials, probeOllama } from "./health.js";
import { MockLlmClient } from "./mock-llm.js";
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
  if (raw === "ollama" || raw === "hunyuan" || raw === "mock" || raw === "auto") return raw;
  return "auto";
};

const ollamaBase = (env: NodeJS.ProcessEnv): string =>
  env["OLLAMA_BASE_URL"]?.trim() || "http://127.0.0.1:11434";

const ollamaModel = (env: NodeJS.ProcessEnv): string => env["OLLAMA_MODEL"]?.trim() || "qwen2.5:7b";

/**
 * 统一 LLM 解析（ADR-002）：auto 本地优先；显式 ollama/hunyuan fail closed（StrictLlmWrapper）。
 */
export const resolveLlmClient = async (opts: ResolveLlmOpts = {}): Promise<ResolvedLlm> => {
  const env = opts.env ?? process.env;
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
    attempts.push({ provider: "mock", ok: true });
    return finish(new MockLlmClient(), {
      provider: "mock",
      model: "mock-template",
      mode: "mock",
      status: "connected",
    });
  }

  if (requested === "ollama") {
    const probe = await probeOllama(ollamaBase(env), 3000, ollamaModel(env), opts.fetch);
    attempts.push({
      provider: "ollama",
      ok: probe.ok,
      ...(probe.reason ? { reason: probe.reason } : {}),
    });
    if (!probe.ok) {
      throw new Error(
        `LLM provider 'ollama' 不可达 (${probe.reason ?? "unknown"})。请启动 ollama serve 或改用 LLM_PROVIDER=auto。`,
      );
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
      attempts.push({ provider: "hunyuan", ok: false, reason: "missing_api_key" });
      throw new Error("LLM provider 'hunyuan' 需要 HUNYUAN_API_KEY。");
    }
    attempts.push({ provider: "hunyuan", ok: true });
    const inner = HunyuanClient.fromEnv(env);
    return finish(new StrictLlmWrapper(inner, "Hunyuan"), {
      provider: "hunyuan",
      model: env["HUNYUAN_MODEL"]?.trim() || "hunyuan-turbos-latest",
      mode: "live",
      status: "connected",
    });
  }

  // auto: Ollama 健康 → Hunyuan 有 key → 可见 mock
  const probe = await probeOllama(ollamaBase(env), 3000, ollamaModel(env), opts.fetch);
  attempts.push({
    provider: "ollama",
    ok: probe.ok,
    ...(probe.reason ? { reason: probe.reason } : {}),
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
    attempts.push({ provider: "hunyuan", ok: true });
    const inner = HunyuanClient.fromEnv(env);
    return finish(new StrictLlmWrapper(inner, "Hunyuan"), {
      provider: "hunyuan",
      model: env["HUNYUAN_MODEL"]?.trim() || "hunyuan-turbos-latest",
      mode: "live",
      status: "connected",
      fallbackFrom: "ollama",
      ...(probe.reason ? { lastError: probe.reason } : {}),
    });
  }

  attempts.push({ provider: "mock", ok: true, reason: "no_healthy_provider" });
  return finish(new MockLlmClient(), {
    provider: "mock",
    model: "mock-template",
    mode: "mock",
    status: "degraded",
    fallbackFrom: "ollama",
    lastError: probe.reason ?? "no cloud credentials",
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
        attempts: [{ provider: "mock", ok: true }],
      },
    };
  }
  if (requested === "hunyuan") {
    const inner = new HunyuanClient();
    return {
      client: new StrictLlmWrapper(inner, "Hunyuan"),
      status: {
        requestedProvider: requested,
        provider: "hunyuan",
        model: env["HUNYUAN_MODEL"]?.trim() || "hunyuan-turbos-latest",
        mode: "live",
        status: inner.getStatus().status,
        attempts: [{ provider: "hunyuan", ok: true }],
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
      attempts: [{ provider: "ollama", ok: true }],
    },
  };
};

export { parseLlmProvider };
