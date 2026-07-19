import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  assertEmbeddingFingerprint,
  readEmbeddingFingerprint,
  writeEmbeddingFingerprint,
} from "../src/embedding-fingerprint.js";
import { validateEmbeddingFingerprintForDataDir } from "../src/embedding-persistence.js";
import { probeOllama } from "../src/health.js";
import { MockLlmClient } from "../src/mock-llm.js";
import { parseEmbedProvider, resolveEmbedder } from "../src/resolve-embedder.js";
import { parseLlmProvider, resolveLlmClientSync } from "../src/resolve-llm.js";

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
