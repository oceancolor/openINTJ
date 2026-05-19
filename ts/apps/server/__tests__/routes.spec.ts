import { describe, expect, it } from "vitest";
import { assembleServerAgent } from "../src/agent.js";
import { buildApp } from "../src/routes.js";

describe("server routes (Hono fetch handler)", () => {
  it("/healthz responds ok", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/healthz");
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ok: boolean };
    expect(body.ok).toBe(true);
  });

  it("/api/status returns 4-plane snapshot", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/api/status");
    expect(res.status).toBe(200);
    const body = (await res.json()) as Record<string, unknown>;
    expect(body).toHaveProperty("llm");
    expect(body).toHaveProperty("memory");
    expect(body).toHaveProperty("governance");
    expect(body).toHaveProperty("tools");
    expect(Array.isArray(body["tools"])).toBe(true);
  });

  it("/api/chat (non-stream) returns finalAnswer in mock mode", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "你好", stream: false }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { finalAnswer: string; status: string };
    expect(typeof body.finalAnswer).toBe("string");
    expect(body.finalAnswer.length).toBeGreaterThan(0);
    expect(body.status).toMatch(/completed|failed|timeout|max_iter_reached/);
  });

  it("/api/chat rejects invalid body", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(res.status).toBe(400);
  });

  it("/api/memory returns recent fragments after a chat", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    await app.request("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "我喜欢绿茶", stream: false }),
    });
    const res = await app.request("/api/memory?topK=10");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      recent?: Array<{ fragmentId: string }>;
    };
    expect(Array.isArray(body.recent)).toBe(true);
    expect(body.recent!.length).toBeGreaterThan(0);
  });

  it("/api/memory with q does retrieval", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    await app.request("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "我家里养了一只橘猫", stream: false }),
    });
    const res = await app.request("/api/memory?q=cat&topK=5");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      query: string;
      results: unknown[];
    };
    expect(body.query).toBe("cat");
    expect(Array.isArray(body.results)).toBe(true);
  });

  it("/api/audit returns stats + recent events", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/api/audit?limit=10");
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      stats: { totalEvents: number };
      recent: unknown[];
    };
    expect(body.stats).toBeDefined();
    expect(Array.isArray(body.recent)).toBe(true);
  });
});
