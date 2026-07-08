/**
 * 技能自学习（Phase 2）装配测试。
 *
 * 覆盖：
 *  1. 默认关：agent.skillLearning undefined，/api/skills/* 全 503
 *  2. opt 开启：runtime 存在，蒸馏→审批→生效 的 HTTP 链路（喂 outcome 走 runtime API，脱 LLM 确定性）
 *  3. OPENINTJ_SKILLS_LEARN 环境变量启用 + opts=false 覆盖 env
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type ServerAgent, assembleServerAgent } from "../src/agent.js";
import { buildApp } from "../src/routes.js";

describe("skills learning (disabled by default)", () => {
  let agent: ServerAgent;
  beforeEach(async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
  });
  afterEach(async () => {
    await agent.close();
  });

  it("agent.skillLearning 未启用时是 undefined", () => {
    expect(agent.skillLearning).toBeUndefined();
  });

  it("/api/skills/* 全部 503 skills_learning_not_enabled", async () => {
    const app = buildApp(agent);
    for (const path of [
      "/api/skills",
      "/api/skills/proposals",
      "/api/skills/distill",
      "/api/skills/proposals/x/approve",
    ]) {
      const method = path.endsWith("distill") || path.endsWith("approve") ? "POST" : "GET";
      const res = await app.request(path, { method });
      expect(res.status).toBe(503);
      const body = (await res.json()) as { error: string };
      expect(body.error).toBe("skills_learning_not_enabled");
    }
  });
});

describe("skills learning (enabled)", () => {
  let agent: ServerAgent;
  beforeEach(async () => {
    agent = await assembleServerAgent({ llmProvider: "mock", enableSkillLearning: true });
  });
  afterEach(async () => {
    await agent.close();
  });

  it("runtime 存在；初始无提案无生效技能", async () => {
    expect(agent.skillLearning).toBeDefined();
    const app = buildApp(agent);
    const proposals = (await (await app.request("/api/skills/proposals")).json()) as {
      total: number;
    };
    expect(proposals.total).toBe(0);
    const active = (await (await app.request("/api/skills")).json()) as { total: number };
    expect(active.total).toBe(0);
  });

  it("distill 空 buffer → produced 0", async () => {
    const app = buildApp(agent);
    const res = await app.request("/api/skills/distill", { method: "POST" });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { produced: number };
    expect(body.produced).toBe(0);
  });

  it("喂成功 outcome → distill → approve → 生效技能出现在 /api/skills", async () => {
    const app = buildApp(agent);
    // 直接走 runtime API 喂成功轨迹（脱 LLM，确定性）。
    for (let i = 0; i < 3; i++) {
      agent.skillLearning!.recordOutcome(`write a unit test case ${i}`, undefined, "completed", {
        toolsUsed: ["read_file"],
      });
    }

    const distillRes = await app.request("/api/skills/distill", { method: "POST" });
    const distillBody = (await distillRes.json()) as {
      produced: number;
      proposals: Array<{ proposalId: string; status: string }>;
    };
    expect(distillBody.produced).toBeGreaterThan(0);
    const proposalId = distillBody.proposals[0]!.proposalId;

    const approveRes = await app.request(`/api/skills/proposals/${proposalId}/approve`, {
      method: "POST",
    });
    expect(approveRes.status).toBe(200);
    expect(((await approveRes.json()) as { status: string }).status).toBe("approved");

    const active = (await (await app.request("/api/skills")).json()) as {
      total: number;
      skills: Array<{ id: string; source: string }>;
    };
    expect(active.total).toBe(1);
    expect(active.skills[0]!.source).toBe("learned:db");

    // 撤销后回到 0
    const revokeRes = await app.request(`/api/skills/proposals/${proposalId}/revoke`, {
      method: "POST",
    });
    expect(revokeRes.status).toBe(200);
    const active2 = (await (await app.request("/api/skills")).json()) as { total: number };
    expect(active2.total).toBe(0);
  });

  it("approve 不存在的 id → 404", async () => {
    const app = buildApp(agent);
    const res = await app.request("/api/skills/proposals/nope/approve", { method: "POST" });
    expect(res.status).toBe(404);
  });
});

describe("skills learning via OPENINTJ_SKILLS_LEARN env", () => {
  it("env=1 启用；opts=false 覆盖 env", async () => {
    const prev = process.env["OPENINTJ_SKILLS_LEARN"];
    process.env["OPENINTJ_SKILLS_LEARN"] = "1";
    try {
      const a = await assembleServerAgent({ llmProvider: "mock" });
      expect(a.skillLearning).toBeDefined();
      await a.close();

      const b = await assembleServerAgent({ llmProvider: "mock", enableSkillLearning: false });
      expect(b.skillLearning).toBeUndefined();
      await b.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_SKILLS_LEARN"];
      else process.env["OPENINTJ_SKILLS_LEARN"] = prev;
    }
  });
});
