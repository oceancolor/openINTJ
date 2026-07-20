import { describe, expect, it, vi } from "vitest";
import { OpenAICompatibleClient } from "../src/index.js";

const client = (fetchImpl: typeof fetch, apiKey = "test-key") =>
  new OpenAICompatibleClient({
    provider: "test-provider",
    apiKey,
    baseUrl: "https://api.example.test/v1/",
    model: "test-model",
    fetch: fetchImpl,
  });

describe("OpenAICompatibleClient", () => {
  it("posts a Chat Completions request and returns content", async () => {
    const fetchMock = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
      expect(String(url)).toBe("https://api.example.test/v1/chat/completions");
      expect((init?.headers as Record<string, string>)["Authorization"]).toBe("Bearer test-key");
      expect(JSON.parse(String(init?.body))).toMatchObject({
        model: "test-model",
        messages: [{ role: "user", content: "hello" }],
        stream: false,
      });
      return new Response(
        JSON.stringify({ choices: [{ message: { role: "assistant", content: "world" } }] }),
        { status: 200 },
      );
    });
    await expect(client(fetchMock).chat([{ role: "user", content: "hello" }])).resolves.toBe(
      "world",
    );
  });

  it("fails closed when the API key is missing", async () => {
    const fetchMock = vi.fn();
    const instance = client(fetchMock as unknown as typeof fetch, "");
    await expect(instance.chat([{ role: "user", content: "hello" }])).rejects.toMatchObject({
      code: "CONFIG_MISSING",
      retriable: false,
      details: { provider: "test-provider", reason: "missing API key" },
    });
    expect(instance.getStatus()).toMatchObject({
      available: false,
      mode: "unauthorized",
      status: "missing_api_key",
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("reports structured authentication failure and updates status", async () => {
    const instance = client(
      async () =>
        new Response(
          JSON.stringify({
            error: { message: "bad key", code: "invalid_api_key", type: "auth_error" },
          }),
          { status: 401 },
        ),
    );
    await expect(instance.chat([{ role: "user", content: "hello" }])).rejects.toMatchObject({
      code: "CONFIG_MISSING",
      retriable: false,
      details: {
        provider: "test-provider",
        status: 401,
        errorCode: "invalid_api_key",
      },
    });
    expect(instance.getStatus()).toMatchObject({
      available: false,
      status: "unauthorized",
      lastErrorCode: "invalid_api_key",
      lastErrorType: "auth_error",
    });
  });

  it("marks 429 and 5xx errors retriable", async () => {
    for (const status of [429, 503]) {
      const instance = client(async () => new Response("busy", { status }));
      await expect(instance.chat([{ role: "user", content: "hello" }])).rejects.toMatchObject({
        code: "INTERNAL_ERROR",
        retriable: true,
        details: { status },
      });
      expect(instance.getStatus().status).toBe("degraded");
    }
  });

  it("rejects malformed successful responses", async () => {
    const instance = client(async () => new Response(JSON.stringify({ choices: [] })));
    await expect(instance.chat([{ role: "user", content: "hello" }])).rejects.toMatchObject({
      code: "INTERNAL_ERROR",
      retriable: true,
    });
    expect(instance.getStatus().lastErrorType).toBe("invalid_response");
  });

  it("propagates caller cancellation and aborts fetch", async () => {
    let fetchSignal: AbortSignal | undefined;
    const instance = client((_url, init) => {
      fetchSignal = init?.signal as AbortSignal;
      return new Promise<Response>((_resolve, reject) => {
        fetchSignal?.addEventListener("abort", () => reject(fetchSignal?.reason), { once: true });
      });
    });
    const controller = new AbortController();
    const pending = instance.chat([{ role: "user", content: "hello" }], {
      signal: controller.signal,
    });
    const reason = new Error("caller cancelled");
    controller.abort(reason);
    await expect(pending).rejects.toBe(reason);
    expect(fetchSignal?.aborted).toBe(true);
  });
});
