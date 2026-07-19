import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { validateEmbeddingFingerprintForDataDir } from "../src/embedding-persistence.js";
import { probeOllama } from "../src/health.js";
import { resolveModelRuntime } from "../src/index.js";
import { resolveLlmClient } from "../src/resolve-llm.js";

const describeOllama = process.env["RUN_OLLAMA_E2E"] === "1" ? describe : describe.skip;
const baseUrl = process.env["OLLAMA_BASE_URL"] ?? "http://127.0.0.1:11434";
const chatModel = process.env["OLLAMA_MODEL"] ?? "qwen2.5:7b";
const embedModel = process.env["OLLAMA_EMBED_MODEL"] ?? "nomic-embed-text";

describeOllama("Ollama runtime e2e", () => {
  it("uses explicit Ollama chat without mock output", async ({ skip }) => {
    if (!(await probeOllama(baseUrl, 3000, chatModel)).ok) skip();
    const runtime = await resolveLlmClient({
      provider: "ollama",
      env: { ...process.env, OLLAMA_BASE_URL: baseUrl, OLLAMA_MODEL: chatModel },
    });
    const output = await runtime.client.chat([{ role: "user", content: "Reply with exactly: OK" }]);
    expect(runtime.status.provider).toBe("ollama");
    expect(runtime.status.mode).toBe("live");
    expect(output).not.toContain("[mock]");
  }, 120_000);

  it("fails closed for an explicit dead endpoint", async () => {
    await expect(
      resolveLlmClient({
        provider: "ollama",
        env: {
          ...process.env,
          OLLAMA_BASE_URL: "http://127.0.0.1:1",
          OLLAMA_MODEL: chatModel,
        },
      }),
    ).rejects.toThrow(/不可达/);
  });

  it("auto visibly falls back when Ollama is unavailable", async () => {
    const runtime = await resolveModelRuntime({
      provider: "auto",
      embedProvider: "auto",
      env: {
        OLLAMA_BASE_URL: "http://127.0.0.1:1",
        OLLAMA_EMBED_ENDPOINT: "http://127.0.0.1:1",
      },
    });
    expect(runtime.status.llm.provider).toBe("mock");
    expect(runtime.status.llm.fallbackFrom).toBe("ollama");
    expect(runtime.status.embed.fallbackFrom).toBe("ollama");
  });

  it("restarts with the same embedding fingerprint and searches persisted vectors", async ({
    skip,
  }) => {
    if (!(await probeOllama(baseUrl, 3000, embedModel)).ok) skip();
    const dir = await mkdtemp(join(tmpdir(), "openintj-ollama-e2e-"));
    const env = {
      ...process.env,
      OLLAMA_BASE_URL: baseUrl,
      OLLAMA_EMBED_ENDPOINT: baseUrl,
      OLLAMA_EMBED_MODEL: embedModel,
    };
    try {
      const first = await resolveModelRuntime({
        provider: "mock",
        embedProvider: "ollama",
        env,
      });
      await validateEmbeddingFingerprintForDataDir(dir, first.embeddingFingerprint);
      const vector = await first.embed.embedder.embed("persistent semantic marker");
      await writeFile(join(dir, "vectors.json"), JSON.stringify([{ id: "marker", vector }]));

      const second = await resolveModelRuntime({
        provider: "mock",
        embedProvider: "ollama",
        env,
      });
      await validateEmbeddingFingerprintForDataDir(dir, second.embeddingFingerprint);
      const query = await second.embed.embedder.embed("persistent semantic marker");
      const rows = JSON.parse(await readFile(join(dir, "vectors.json"), "utf8")) as Array<{
        id: string;
        vector: number[];
      }>;
      const dot = rows[0]!.vector.reduce((sum, value, i) => sum + value * query[i]!, 0);
      expect(rows[0]!.id).toBe("marker");
      expect(dot).toBeGreaterThan(0);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }, 120_000);
});
