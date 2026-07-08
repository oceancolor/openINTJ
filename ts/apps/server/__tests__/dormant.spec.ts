/**
 * Dormant Memory Learning 装配测试（RFC-003 方向 3 + Phase 3.3.A）。
 *
 * 覆盖：
 *  1. 未启用时 /api/dormant/* 路由全部 503，agent.dormant 是 undefined
 *  2. 启用后 agent.run() 自动喂 PassiveStore
 *  3. mine → proposals → approve → persona 的完整 HTTP 链路
 *  4. /api/status 暴露 dormant 子项
 *  5. OPENINTJ_DORMANT=1 环境变量也能启用
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type ServerAgent, assembleServerAgent } from "../src/agent.js";
import { buildApp } from "../src/routes.js";

describe("dormant memory learning (disabled by default)", () => {
  let agent: ServerAgent;
  beforeEach(async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
  });
  afterEach(async () => {
    await agent.close();
  });

  it("agent.dormant 未启用时是 undefined", () => {
    expect(agent.dormant).toBeUndefined();
  });

  it("/api/dormant/mine 返回 503", async () => {
    const app = buildApp(agent);
    const res = await app.request("/api/dormant/mine", { method: "POST" });
    expect(res.status).toBe(503);
    const body = (await res.json()) as { error: string };
    expect(body.error).toBe("dormant_not_enabled");
  });

  it("/api/dormant/proposals 返回 503", async () => {
    const app = buildApp(agent);
    const res = await app.request("/api/dormant/proposals");
    expect(res.status).toBe(503);
  });

  it("/api/status 不带 dormant 字段", async () => {
    const app = buildApp(agent);
    const res = await app.request("/api/status");
    const body = (await res.json()) as Record<string, unknown>;
    expect(body["dormant"]).toBeUndefined();
  });
});

describe("dormant memory learning (enabled)", () => {
  let agent: ServerAgent;

  beforeEach(async () => {
    agent = await assembleServerAgent({
      llmProvider: "mock",
      enableDormant: true,
      dormantOpts: {
        minerOpts: {
          ngramSize: 2,
          minFrequency: 2,
          minConfidence: 0.3,
          // 注入式 LLM 抽取，跳过真实 LLM；给 ngram 打 preference 类别
          llmExtract: async (ngram) => ({
            description: `用户偏好（来自 mock）: ${ngram}`,
            category: "preference",
          }),
        },
      },
    });
  });
  afterEach(async () => {
    await agent.close();
  });

  it("agent.run() 自动把用户输入 + final answer 喂进 PassiveStore", async () => {
    expect(agent.dormant).toBeDefined();
    expect(agent.dormant!.passiveSize()).toBe(0);
    await agent.run("你好 世界");
    // 1 user input + 1 assistant output = 2
    expect(agent.dormant!.passiveSize()).toBe(2);
  });

  it("/api/status 暴露 dormant 子项", async () => {
    const app = buildApp(agent);
    const res = await app.request("/api/status");
    const body = (await res.json()) as {
      dormant?: { enabled: boolean; passiveSize: number; pendingProposals: number };
    };
    expect(body.dormant).toBeDefined();
    expect(body.dormant!.enabled).toBe(true);
    expect(body.dormant!.passiveSize).toBe(0);
    expect(body.dormant!.pendingProposals).toBe(0);
  });

  it("mine → proposals → approve → persona 完整 HTTP 链路", async () => {
    const app = buildApp(agent);

    // 1) 直接往 dormant 灌一批近似事件（绕开 LLM，确定性更强）
    for (let i = 0; i < 5; i++) {
      agent.dormant!.record("绿 茶 健 康", "user", { iter: i });
    }

    // 2) mine
    const mineRes = await app.request("/api/dormant/mine", { method: "POST" });
    expect(mineRes.status).toBe(200);
    const mineBody = (await mineRes.json()) as {
      scannedEvents: number;
      patterns: Array<{ patternId: string; description: string }>;
      proposals: Array<{ proposalId: string; status: string }>;
    };
    expect(mineBody.scannedEvents).toBe(5);
    expect(mineBody.patterns.length).toBeGreaterThan(0);
    expect(mineBody.proposals.length).toBeGreaterThan(0);

    // 3) list pending
    const listRes = await app.request("/api/dormant/proposals?status=pending");
    const listBody = (await listRes.json()) as {
      total: number;
      proposals: Array<{ proposalId: string; status: string }>;
    };
    expect(listBody.total).toBeGreaterThan(0);
    expect(listBody.proposals.every((p) => p.status === "pending")).toBe(true);

    // 4) approve 第一条
    const first = listBody.proposals[0]!;
    const approveRes = await app.request(`/api/dormant/proposals/${first.proposalId}/approve`, {
      method: "POST",
    });
    expect(approveRes.status).toBe(200);
    const approveBody = (await approveRes.json()) as { status: string };
    expect(approveBody.status).toBe("applied");

    // 5) persona 已经记录了这条偏好
    const personaRes = await app.request("/api/dormant/persona");
    expect(personaRes.status).toBe(200);
    const persona = (await personaRes.json()) as {
      preferences: Record<string, unknown>;
      meta: { version: number };
    };
    expect(Object.keys(persona.preferences).length).toBeGreaterThan(0);
    expect(persona.meta.version).toBe(1);
  });

  it("approve 不存在 / 已决策的 proposalId 返回 404", async () => {
    const app = buildApp(agent);
    const r1 = await app.request("/api/dormant/proposals/non-existent-id/approve", {
      method: "POST",
    });
    expect(r1.status).toBe(404);
  });

  it("revoke 已 applied 的条目：从 persona 删字段 + version++ + 状态转 revoked（§3.6 #4）", async () => {
    const app = buildApp(agent);
    for (let i = 0; i < 5; i++) {
      agent.dormant!.record("绿 茶 健 康", "user", { iter: i });
    }
    await app.request("/api/dormant/mine", { method: "POST" });
    const listBody = (await (
      await app.request("/api/dormant/proposals?status=pending")
    ).json()) as { proposals: Array<{ proposalId: string }> };
    const id = listBody.proposals[0]!.proposalId;

    // approve → persona 有内容、version=1
    await app.request(`/api/dormant/proposals/${id}/approve`, { method: "POST" });
    const afterApprove = (await (await app.request("/api/dormant/persona")).json()) as {
      preferences: Record<string, unknown>;
      meta: { version: number };
    };
    expect(Object.keys(afterApprove.preferences).length).toBeGreaterThan(0);

    // revoke → 200 + status=revoked
    const revokeRes = await app.request(`/api/dormant/proposals/${id}/revoke`, { method: "POST" });
    expect(revokeRes.status).toBe(200);
    expect(((await revokeRes.json()) as { status: string }).status).toBe("revoked");

    // persona 字段被删除、version 再 +1
    const afterRevoke = (await (await app.request("/api/dormant/persona")).json()) as {
      preferences: Record<string, unknown>;
      meta: { version: number };
    };
    expect(Object.keys(afterRevoke.preferences).length).toBe(0);
    expect(afterRevoke.meta.version).toBe(afterApprove.meta.version + 1);

    // revoked 状态可被 list 过滤出来
    const revokedList = (await (
      await app.request("/api/dormant/proposals?status=revoked")
    ).json()) as { proposals: Array<{ proposalId: string; status: string }> };
    expect(revokedList.proposals.some((p) => p.proposalId === id)).toBe(true);
  });

  it("revoke 不存在 / 非 applied 的 proposalId 返回 404", async () => {
    const app = buildApp(agent);
    const r = await app.request("/api/dormant/proposals/nope/revoke", { method: "POST" });
    expect(r.status).toBe(404);
    expect(((await r.json()) as { error: string }).error).toBe("not_found_or_not_applied");
  });

  it("reject 流程不污染 persona", async () => {
    const app = buildApp(agent);
    for (let i = 0; i < 3; i++) {
      agent.dormant!.record("讨 厌", "user");
    }
    const mineRes = await app.request("/api/dormant/mine", { method: "POST" });
    const mineBody = (await mineRes.json()) as {
      proposals: Array<{ proposalId: string }>;
    };
    expect(mineBody.proposals.length).toBeGreaterThan(0);

    const id = mineBody.proposals[0]!.proposalId;
    const r = await app.request(`/api/dormant/proposals/${id}/reject`, {
      method: "POST",
    });
    expect(r.status).toBe(200);

    const persona = (await (await app.request("/api/dormant/persona")).json()) as {
      preferences: Record<string, unknown>;
    };
    expect(Object.keys(persona.preferences).length).toBe(0);
  });
});

describe("dormant via OPENINTJ_DORMANT env", () => {
  it("env=1 显式启用", async () => {
    const prev = process.env["OPENINTJ_DORMANT"];
    process.env["OPENINTJ_DORMANT"] = "1";
    try {
      const agent = await assembleServerAgent({ llmProvider: "mock" });
      expect(agent.dormant).toBeDefined();
      await agent.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_DORMANT"];
      else process.env["OPENINTJ_DORMANT"] = prev;
    }
  });

  it("env=0 / 未设置 → 不启用（即使 enableDormant 缺省）", async () => {
    const prev = process.env["OPENINTJ_DORMANT"];
    delete process.env["OPENINTJ_DORMANT"];
    try {
      const agent = await assembleServerAgent({ llmProvider: "mock" });
      expect(agent.dormant).toBeUndefined();
      await agent.close();
    } finally {
      if (prev !== undefined) process.env["OPENINTJ_DORMANT"] = prev;
    }
  });

  it("opts.enableDormant=false 覆盖 env=1", async () => {
    const prev = process.env["OPENINTJ_DORMANT"];
    process.env["OPENINTJ_DORMANT"] = "1";
    try {
      const agent = await assembleServerAgent({
        llmProvider: "mock",
        enableDormant: false,
      });
      expect(agent.dormant).toBeUndefined();
      await agent.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_DORMANT"];
      else process.env["OPENINTJ_DORMANT"] = prev;
    }
  });
});
