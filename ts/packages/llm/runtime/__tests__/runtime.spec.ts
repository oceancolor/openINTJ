import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { HookBus } from "@openintj/core";
import { describe, expect, it } from "vitest";
import {
  assertEmbeddingFingerprint,
  readEmbeddingFingerprint,
  writeEmbeddingFingerprint,
} from "../src/embedding-fingerprint.js";
import { validateEmbeddingFingerprintForDataDir } from "../src/embedding-persistence.js";
import { ModelRuntimeError, sanitizeModelRuntimeErrorMessage } from "../src/errors.js";
import { probeOllama } from "../src/health.js";
import { resolveModelRuntime } from "../src/index.js";
import { MockLlmClient } from "../src/mock-llm.js";
import { OPENAI_PROVIDER_PROFILES } from "../src/openai-providers.js";
import { parseEmbedProvider, resolveEmbedder } from "../src/resolve-embedder.js";
import { parseLlmProvider, resolveLlmClient, resolveLlmClientSync } from "../src/resolve-llm.js";

const simpleFingerprint = {
  schemaVersion: 1 as const,
  provider: "simple-sha256",
  model: "dim64",
  dimension: 64,
};

describe("parseLlmProvider", () => {
  it("默认 auto", () => {
    expect(parseLlmProvider({})).toBe("auto");
  });
  it("读取 LLM_PROVIDER", () => {
    expect(parseLlmProvider({ LLM_PROVIDER: "ollama" })).toBe("ollama");
  });
  it.each(["kimi", "minimax", "glm"] as const)("recognizes %s", (provider) => {
    expect(parseLlmProvider({ LLM_PROVIDER: provider })).toBe(provider);
  });
});

describe("explicit OpenAI-compatible providers", () => {
  const cases = [
    ["kimi", "KIMI_API_KEY", "kimi-k2.5", "https://api.moonshot.cn/v1"],
    ["minimax", "MINIMAX_API_KEY", "MiniMax-M2.1", "https://api.minimaxi.com/v1"],
    ["glm", "GLM_API_KEY", "glm-4.7", "https://open.bigmodel.cn/api/paas/v4"],
  ] as const;

  it.each(cases)(
    "%s uses official defaults and calls the compatible endpoint",
    async (provider, keyName, model, baseUrl) => {
      const fetchMock: typeof fetch = async (url, init) => {
        expect(String(url)).toBe(`${baseUrl}/chat/completions`);
        expect(JSON.parse(String(init?.body))).toMatchObject({ model });
        return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }));
      };
      const resolved = await resolveLlmClient({
        provider,
        env: { [keyName]: "key" },
        fetch: fetchMock,
      });
      expect(resolved.status).toMatchObject({
        requestedProvider: provider,
        provider,
        model,
        mode: "live",
        status: "connected",
      });
      await expect(resolved.client.chat([{ role: "user", content: "hi" }])).resolves.toBe("ok");
    },
  );

  it.each(Object.keys(OPENAI_PROVIDER_PROFILES) as Array<keyof typeof OPENAI_PROVIDER_PROFILES>)(
    "%s fails closed without credentials",
    async (provider) => {
      await expect(resolveLlmClient({ provider, env: {} })).rejects.toMatchObject({
        name: "ModelRuntimeError",
        code: "MODEL_CREDENTIAL_MISSING",
        retriable: false,
        provider,
      });
    },
  );

  it("preserves explicit endpoint and model overrides", async () => {
    const resolved = await resolveLlmClient({
      provider: "kimi",
      env: {
        KIMI_API_KEY: "key",
        KIMI_BASE_URL: "https://gateway.example.test/v1",
        KIMI_MODEL: "custom-kimi",
      },
      fetch: async (url, init) => {
        expect(String(url)).toBe("https://gateway.example.test/v1/chat/completions");
        expect(JSON.parse(String(init?.body))).toMatchObject({ model: "custom-kimi" });
        return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }));
      },
    });
    expect(resolved.status.model).toBe("custom-kimi");
    await resolved.client.chat([{ role: "user", content: "hi" }]);
  });

  it("keeps auto order limited to Ollama, Hunyuan, then mock", async () => {
    const resolved = await resolveLlmClient({
      provider: "auto",
      env: { KIMI_API_KEY: "key", MINIMAX_API_KEY: "key", GLM_API_KEY: "key" },
      fetch: async () => {
        throw new Error("ollama offline");
      },
    });
    expect(resolved.status.provider).toBe("mock");
    expect(resolved.status.attempts.map((attempt) => attempt.provider)).toEqual(["ollama", "mock"]);
  });

  it("surfaces authorization failure in runtime health", async () => {
    const runtime = await resolveModelRuntime({
      provider: "glm",
      embedProvider: "simple",
      env: { GLM_API_KEY: "bad" },
      fetch: async () =>
        new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401 }),
      healthRefreshIntervalMs: 0,
    });
    await expect(runtime.llm.client.chat([{ role: "user", content: "hi" }])).rejects.toMatchObject({
      code: "CONFIG_MISSING",
    });
    const status = await runtime.refreshHealth();
    expect(status.llm).toMatchObject({
      provider: "glm",
      status: "unauthorized",
      lastError: { code: "MODEL_AUTH_FAILED", retriable: false },
    });
  });
});

describe("resolveLlmClientSync mock", () => {
  it("mock provider 返回 MockLlmClient", async () => {
    const { client, status } = resolveLlmClientSync({ provider: "mock" });
    expect(client).toBeInstanceOf(MockLlmClient);
    expect(status.provider).toBe("mock");
    expect(status.mode).toBe("mock");
    const out = await client.chat([{ role: "user", content: "hi" }]);
    expect(out).toContain("[mock]");
  });
});

describe("ModelRuntime lifecycle and structured errors", () => {
  it("throws a structured error for an ineligible explicit provider", async () => {
    await expect(resolveLlmClient({ provider: "hunyuan", env: {} })).rejects.toMatchObject({
      name: "ModelRuntimeError",
      code: "MODEL_CREDENTIAL_MISSING",
      retriable: false,
      provider: "hunyuan",
    });
  });

  it("redacts and bounds error text exposed to status and telemetry", () => {
    const secret = `api_key=secret-value ${"x".repeat(400)}`;
    const message = sanitizeModelRuntimeErrorMessage(secret);
    expect(message).not.toContain("secret-value");
    expect(message.length).toBeLessThanOrEqual(256);
  });

  it("refreshes selected provider health with throttling and emits hooks", async () => {
    const hooks = new HookBus();
    const probes: Array<{ ok: boolean; errorCode?: string }> = [];
    const errors: string[] = [];
    hooks.on("model.provider.probe", (ctx) => void probes.push(ctx.payload));
    hooks.on("model.provider.error", (ctx) => void errors.push(ctx.payload.code));
    let now = 0;
    let fetchCalls = 0;
    const fetchMock: typeof fetch = async () => {
      fetchCalls++;
      if (fetchCalls === 1) {
        return new Response(JSON.stringify({ models: [{ name: "qwen2.5:7b" }] }), {
          status: 200,
        });
      }
      throw new Error("offline");
    };
    const runtime = await resolveModelRuntime({
      provider: "ollama",
      embedProvider: "simple",
      env: { OLLAMA_MODEL: "qwen2.5:7b" },
      fetch: fetchMock,
      hooks,
      now: () => now,
      healthRefreshIntervalMs: 10,
    });

    await runtime.refreshHealth();
    expect(fetchCalls).toBe(1);
    now = 11;
    const refreshed = await runtime.refreshHealth();

    expect(fetchCalls).toBe(2);
    expect(refreshed.llm.status).toBe("degraded");
    expect(refreshed.llm.lastError).toMatchObject({
      code: "MODEL_PROVIDER_UNAVAILABLE",
      retriable: true,
      at: 11,
    });
    expect(probes.map((probe) => probe.ok)).toEqual([true, false]);
    expect(errors).toContain("MODEL_PROVIDER_UNAVAILABLE");
    await runtime.close();
  });
});

describe("resolveEmbedder simple", () => {
  it("EMBED_PROVIDER=simple", async () => {
    const r = await resolveEmbedder({ provider: "simple", env: {} });
    expect(r.dimension).toBe(64);
    expect(r.status.provider).toContain("simple");
  });
});

describe("embedding fingerprint", () => {
  it("读写与 mismatch 检测", async () => {
    const dir = await mkdtemp(join(tmpdir(), "openintj-fp-"));
    try {
      await writeEmbeddingFingerprint(dir, {
        schemaVersion: 1,
        provider: "simple-sha256",
        model: "dim64",
        dimension: 64,
      });
      const stored = await readEmbeddingFingerprint(dir);
      expect(stored?.dimension).toBe(64);
      assertEmbeddingFingerprint(simpleFingerprint, stored);
      expect(() =>
        assertEmbeddingFingerprint(
          { schemaVersion: 1, provider: "ollama", model: "nomic", dimension: 768 },
          stored,
        ),
      ).toThrow(/EMBEDDING_FINGERPRINT_MISMATCH/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("model-aware Ollama probe", () => {
  it("rejects a healthy daemon when the configured model is absent", async () => {
    const fetchMock: typeof fetch = async () =>
      new Response(JSON.stringify({ models: [{ name: "other:latest" }] }), { status: 200 });
    await expect(
      probeOllama("http://ollama.test", 100, "qwen2.5:7b", fetchMock),
    ).resolves.toMatchObject({
      ok: false,
      reason: "model_not_installed:qwen2.5:7b",
    });
  });

  it("accepts the configured model", async () => {
    const fetchMock: typeof fetch = async () =>
      new Response(JSON.stringify({ models: [{ name: "qwen2.5:7b" }] }), { status: 200 });
    await expect(
      probeOllama("http://ollama.test", 100, "qwen2.5:7b", fetchMock),
    ).resolves.toMatchObject({
      ok: true,
    });
  });

  it("treats an omitted tag as latest", async () => {
    const fetchMock: typeof fetch = async () =>
      new Response(JSON.stringify({ models: [{ name: "nomic-embed-text:latest" }] }), {
        status: 200,
      });
    await expect(
      probeOllama("http://ollama.test", 100, "nomic-embed-text", fetchMock),
    ).resolves.toMatchObject({ ok: true });
  });
});

describe("embedding persistence ordering", () => {
  it("writes an empty directory fingerprint and accepts the same restart", async () => {
    const root = await mkdtemp(join(tmpdir(), "openintj-fp-empty-"));
    const dir = join(root, "new-store");
    try {
      await validateEmbeddingFingerprintForDataDir(dir, simpleFingerprint);
      expect(await readEmbeddingFingerprint(dir)).toEqual(simpleFingerprint);
      await validateEmbeddingFingerprintForDataDir(dir, simpleFingerprint);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("fails closed on mismatch before reuse", async () => {
    const dir = await mkdtemp(join(tmpdir(), "openintj-fp-mismatch-"));
    try {
      await writeEmbeddingFingerprint(dir, simpleFingerprint);
      await expect(
        validateEmbeddingFingerprintForDataDir(dir, {
          schemaVersion: 1,
          provider: "ollama",
          model: "nomic-embed-text",
          dimension: 64,
        }),
      ).rejects.toThrow(/EMBEDDING_FINGERPRINT_MISMATCH/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("emits checked and rejected fingerprint lifecycle events", async () => {
    const root = await mkdtemp(join(tmpdir(), "openintj-fp-hooks-"));
    const hooks = new HookBus();
    const events: string[] = [];
    hooks.on("model.embedding.fingerprint.checked", () => void events.push("checked"));
    hooks.on("model.embedding.fingerprint.rejected", () => void events.push("rejected"));
    try {
      await validateEmbeddingFingerprintForDataDir(root, simpleFingerprint, undefined, { hooks });
      await expect(
        validateEmbeddingFingerprintForDataDir(
          root,
          { ...simpleFingerprint, model: "other" },
          undefined,
          { hooks },
        ),
      ).rejects.toBeInstanceOf(ModelRuntimeError);
      expect(events).toEqual(["checked", "rejected"]);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("rejects legacy data with no fingerprint", async () => {
    const dir = await mkdtemp(join(tmpdir(), "openintj-fp-legacy-"));
    try {
      await mkdir(join(dir, "lancedb"), { recursive: true });
      await writeFile(join(dir, "lancedb", "legacy-data"), "present");
      await expect(validateEmbeddingFingerprintForDataDir(dir, simpleFingerprint)).rejects.toThrow(
        /EMBEDDING_FINGERPRINT_MISSING/,
      );
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe("parseEmbedProvider", () => {
  it("默认 auto", () => {
    expect(parseEmbedProvider({})).toBe("auto");
  });
});
