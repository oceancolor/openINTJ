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

  it("/api/chat intercepts /clear before the model", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    let thoughts = 0;
    agent.hooks.on("react.beforeThought", () => {
      thoughts += 1;
    });
    const app = buildApp(agent);
    const res = await app.request("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: "/clear", stream: false }),
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { finalAnswer: string; status: string };
    expect(body.finalAnswer).toBe("对话已清空。");
    expect(body.status).toBe("completed");
    expect(thoughts).toBe(0);
    expect(agent.journal.list("server").at(-1)?.type).toBe("session.cleared");
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

  it("/api/profile supports CRUD + export (RFC-014)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const created = await app.request("/api/profile", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ section: "preferences", key: "editor", value: "VS Code" }),
    });
    expect(created.status).toBe(200);
    const entry = (await created.json()) as { id: string; value: string };
    expect(entry.value).toBe("VS Code");

    const listed = await app.request("/api/profile?section=preferences");
    const listBody = (await listed.json()) as { entries: unknown[] };
    expect(listBody.entries).toHaveLength(1);

    const exported = await app.request("/api/profile/export");
    const exportBody = (await exported.json()) as { profile: string };
    expect(exportBody.profile).toContain("VS Code");

    const invalid = await app.request("/api/profile", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ section: "bogus", key: "x", value: "y" }),
    });
    expect(invalid.status).toBe(400);

    const removed = await app.request(`/api/profile/${entry.id}`, { method: "DELETE" });
    expect(removed.status).toBe(200);
    const after = await app.request("/api/profile");
    const afterBody = (await after.json()) as { entries: unknown[] };
    expect(afterBody.entries).toEqual([]);
  });

  it("/api/identity asserts, resolves offline for phone seeds, confirms and unlinks (RFC-020)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const asserted = await app.request("/api/identity/assert", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ platform: "wechat", kind: "wechat", value: "ben_dev_01" }),
    });
    expect(asserted.status).toBe(200);
    const person = (await asserted.json()) as {
      id: string;
      accounts: Array<{ id: string; source: string }>;
    };
    expect(person.accounts[0]?.source).toBe("manual");

    const listed = await app.request("/api/identity/persons");
    const listBody = (await listed.json()) as { persons: unknown[] };
    expect(listBody.persons).toHaveLength(1);

    // Phone seeds never hit the network (Gravatar is email-only); empty corpus.
    const resolved = await app.request("/api/identity/resolve", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind: "phone", value: "13800138000" }),
    });
    expect(resolved.status).toBe(200);
    const resolvedBody = (await resolved.json()) as { proposals: unknown[] };
    expect(resolvedBody.proposals).toEqual([]);

    // Confirming a value already owned by another person conflicts (409).
    const conflict = await app.request("/api/identity/confirm", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        proposals: [
          {
            id: "p1",
            personId: null,
            displayName: "Someone Else",
            account: {
              platform: "wechat",
              kind: "wechat",
              value: "ben_dev_01",
              source: "manual",
              evidence: "test",
              confidence: 0.5,
              verifiedAt: null,
            },
          },
        ],
      }),
    });
    expect(conflict.status).toBe(409);

    const unlinked = await app.request("/api/identity/unlink", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ accountId: person.accounts[0]!.id }),
    });
    expect(unlinked.status).toBe(200);
    const after = await app.request("/api/identity/persons");
    expect(((await after.json()) as { persons: unknown[] }).persons).toEqual([]);
    await agent.close();
  });

  it("/api/identity mirrors the primary email into the profile identity section (RFC-020 writeback)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    await app.request("/api/identity/assert", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        displayName: "Ben",
        platform: "gravatar",
        kind: "email",
        value: "ben@example.com",
      }),
    });
    const listed = await app.request("/api/profile?section=identity");
    const body = (await listed.json()) as {
      entries: Array<{ key: string; value: string; provenance?: string }>;
    };
    expect(body.entries).toContainEqual(
      expect.objectContaining({
        key: "email",
        value: "ben@example.com",
        provenance: "identity-graph",
      }),
    );
    expect(body.entries).toContainEqual(expect.objectContaining({ key: "name", value: "Ben" }));
    await agent.close();
  });

  it("/api/connectors reports connector statuses (RFC-016)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/api/connectors");
    expect(res.status).toBe(200);
    const body = (await res.json()) as { connectors: Array<{ id: string; ok: boolean }> };
    expect(Array.isArray(body.connectors)).toBe(true);
    await agent.close();
  });

  it("/api/taskpool/checkpoints lists checkpoints and rejects unknown run actions (RFC-012)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const listed = await app.request("/api/taskpool/checkpoints");
    expect(listed.status).toBe(200);
    expect(await listed.json()).toEqual({ checkpoints: [] });

    const resumed = await app.request("/api/taskpool/runs/missing-run/resume", {
      method: "POST",
    });
    expect(resumed.status).toBe(404);
    const cancelled = await app.request("/api/taskpool/runs/missing-run/cancel", {
      method: "POST",
    });
    expect(cancelled.status).toBe(404);
    await agent.close();
  });

  it("/api/subagents lists subagent records (RFC-012 §3)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/api/subagents");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ subagents: [] });
    await agent.close();
  });

  it("/api/channels lists channels and 404s unknown test targets (RFC-018)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const res = await app.request("/api/channels");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ channels: [] });
    const tested = await app.request("/api/channels/nope/test", { method: "POST" });
    expect(tested.status).toBe(404);
    await agent.close();
  });

  it("/api/grants creates, lists and revokes explicit grants (RFC-013)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const created = await app.request("/api/grants", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tool: "write_file", risk: "write", scope: "notes/" }),
    });
    expect(created.status).toBe(200);
    const grant = (await created.json()) as { id: string };
    const listed = await app.request("/api/grants");
    const body = (await listed.json()) as { grants: Array<{ id: string }> };
    expect(body.grants.map((g) => g.id)).toEqual([grant.id]);
    const revoked = await app.request(`/api/grants/${grant.id}`, { method: "DELETE" });
    expect(revoked.status).toBe(200);
    expect(agent.grants.list()).toEqual([]);
    await agent.close();
  });

  it("/api/sync exports and re-imports a bundle; /api/voice 503s unconfigured (RFC-019)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    agent.profile.upsert({ section: "identity", key: "name", value: "Ben" });
    const exported = await app.request("/api/sync/export");
    expect(exported.status).toBe(200);
    const bundle = (await exported.json()) as { version: number };
    expect(bundle.version).toBe(1);
    const imported = await app.request("/api/sync/import", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(bundle),
    });
    expect(imported.status).toBe(200);
    const bad = await app.request("/api/sync/import", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ version: 9 }),
    });
    expect(bad.status).toBe(400);
    const voice = await app.request("/api/voice/transcribe", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ audioBase64: "AA==", mimeType: "audio/wav" }),
    });
    expect(voice.status).toBe(503);
    await agent.close();
  });

  it("/api/schedules supports CRUD (RFC-015)", async () => {
    const agent = await assembleServerAgent({ llmProvider: "mock" });
    const app = buildApp(agent);
    const created = await app.request("/api/schedules", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        title: "晨报",
        prompt: "汇总昨夜动态",
        trigger: { kind: "daily", time: "09:00" },
      }),
    });
    expect(created.status).toBe(200);
    const job = (await created.json()) as { id: string; nextRunAt?: number };
    expect(job.nextRunAt).toBeGreaterThan(Date.now());

    const listed = await app.request("/api/schedules");
    const listBody = (await listed.json()) as { jobs: unknown[] };
    expect(listBody.jobs).toHaveLength(1);

    const patched = await app.request(`/api/schedules/${job.id}`, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ enabled: false }),
    });
    expect(patched.status).toBe(200);
    expect(((await patched.json()) as { enabled: boolean }).enabled).toBe(false);

    const invalid = await app.request("/api/schedules", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ title: "x", prompt: "y", trigger: { kind: "daily", time: "25:00" } }),
    });
    expect(invalid.status).toBe(400);

    const removed = await app.request(`/api/schedules/${job.id}`, { method: "DELETE" });
    expect(removed.status).toBe(200);
    await agent.close();
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
