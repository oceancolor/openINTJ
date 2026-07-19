/**
 * 真实持久化 e2e —— 验证：写入 → 关闭 → 重启装配 → 读回一致。
 *
 * 走真实 LanceDB + SQLite（peer deps 必须装好）。
 * 由 OPENINTJ_E2E=1 控制是否运行；CI 默认跳过以保留对 peer dep 的可选性。
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SqliteTaskStore } from "@openintj/storage-sqlite";
import type { StoredTaskRun } from "@openintj/taskpool";
import { afterAll, describe, expect, it } from "vitest";
import { assembleServerAgent } from "../src/agent.js";
import { buildApp } from "../src/routes.js";

const RUN_E2E = process.env["OPENINTJ_E2E"] === "1";
const describeE2E = RUN_E2E ? describe : describe.skip;

const fixtures: string[] = [];
const makeDir = (label: string): string => {
  const d = mkdtempSync(join(tmpdir(), `openintj-e2e-${label}-`));
  fixtures.push(d);
  return d;
};

afterAll(() => {
  for (const d of fixtures) {
    try {
      rmSync(d, { recursive: true, force: true });
    } catch {
      // 忽略 LanceDB 可能持有的句柄释放延迟
    }
  }
});

describeE2E("server persistence e2e (LanceDB + SQLite)", { timeout: 30_000 }, () => {
  it("explicitly resumes an incomplete TaskPool run during startup", async () => {
    const dir = makeDir("taskpool-resume");
    const dbPath = join(dir, "taskpool.sqlite");
    const stored: StoredTaskRun = {
      runId: "server-recovery-run",
      planId: "server-recovery-plan",
      status: "running",
      graph: {
        planId: "server-recovery-plan",
        goalIntent: "plan",
        goalInput: "规划服务迁移",
        nodes: [{ id: "a", deps: [], action: "respond", description: "输出计划" }],
      },
      nodes: [{ taskId: "a", state: "running", attempt: 1, updatedAt: 2 }],
      createdAt: 1,
      updatedAt: 2,
    };
    const seed = new SqliteTaskStore(dbPath, false);
    await seed.init();
    await seed.saveRun(stored);
    await seed.close();

    const agent = await assembleServerAgent({
      llmProvider: "mock",
      embedProvider: "simple",
      dataDir: dir,
      enableTaskPool: true,
      taskPoolRecoveryPolicy: "resume",
    });

    expect(agent.taskPoolRecovery).toEqual({
      policy: "resume",
      found: 1,
      resumed: 1,
      completed: 1,
      cancelled: 0,
      failed: 0,
    });
    await agent.close();

    const verify = new SqliteTaskStore(dbPath, false);
    await verify.init();
    expect((await verify.loadRun(stored.runId))?.status).toBe("completed");
    await verify.close();
  });

  it("write → close → reassemble → memory + audit + vector search 一致", async () => {
    const dir = makeDir("roundtrip");

    // ===== 第一次会话：写入 =====
    const a1 = await assembleServerAgent({
      llmProvider: "mock",
      dataDir: dir,
    });
    expect(a1.persistenceInfo.mode).toBe("real");
    expect(a1.persistenceInfo.dataDir).toBe(dir);

    a1.persistentStore.addLongTerm("我喜欢喝绿茶", {
      taskTags: ["preference"],
      importance: 0.9,
      metadata: { topic: "tea" },
    });
    a1.persistentStore.addWorking("今天讨论的项目代号 Atlas", {
      taskTags: ["project"],
      importance: 0.7,
    });
    a1.persistentStore.addShortTerm("用户名是 ben", {
      importance: 0.5,
    });

    a1.governance.auditTrail.record({
      eventType: "policy.checked",
      command: "read_file",
      riskLevel: "low",
      approved: true,
      reason: null,
      metadata: { path: "/tmp/x" },
    });

    await a1.persistentStore.awaitPendingWrites();
    // governance.auditTrail 默认是内存版，需要手动同步到 metadataStore
    // 此处我们直接写一笔 audit 进 metadataStore 验证 SQLite 持久化
    await a1.persistentStore.metadataStore.recordAudit({
      eventId: "audit-e1",
      eventType: "policy.checked",
      command: "read_file",
      riskLevel: "low",
      approved: 1,
      reason: null,
      metadataJson: JSON.stringify({ path: "/tmp/x" }),
      timestamp: Date.now(),
    });
    await a1.close();

    // ===== 第二次装配：在同一 dataDir 上重启 =====
    const a2 = await assembleServerAgent({
      llmProvider: "mock",
      dataDir: dir,
    });
    expect(a2.persistenceInfo.mode).toBe("real");

    // 验证 hydrate 后内存三层有数据
    const stats = a2.memory.getStats();
    expect(stats.total).toBeGreaterThanOrEqual(3);

    // 验证元数据可读
    const allMeta = await a2.persistentStore.metadataStore.listFragmentMeta({
      limit: 100,
    });
    expect(allMeta.length).toBeGreaterThanOrEqual(3);
    const tea = allMeta.find((m) => m.taskTagsCsv === "preference");
    expect(tea).toBeDefined();
    expect(tea!.memoryType).toBe("long_term");
    expect(tea!.importance).toBeCloseTo(0.9, 5);

    // 验证向量检索能从持久化层取回结果
    const searchEmbed = a2.persistentStore.embedder.embed("绿茶");
    const vec = Array.isArray(searchEmbed) ? searchEmbed : await searchEmbed;
    const hits = await a2.persistentStore.vectorSearch(vec, { topK: 5 });
    expect(hits.length).toBeGreaterThan(0);
    const teaHit = hits.find((h) => h.row.content.includes("绿茶"));
    expect(teaHit).toBeDefined();

    // 验证 audit 持久化读回
    const auditRows = await a2.persistentStore.metadataStore.queryAudit({
      eventType: "policy.checked",
    });
    expect(auditRows.length).toBeGreaterThanOrEqual(1);
    expect(auditRows[0]!.command).toBe("read_file");

    // 验证 HTTP 路由也能在持久化模式下跑通
    const app = buildApp(a2);
    const status = await app.request("/api/status");
    const body = (await status.json()) as {
      persistence: { mode: string };
    };
    expect(body.persistence.mode).toBe("real");

    const memRes = await app.request("/api/memory?topK=10");
    const memBody = (await memRes.json()) as {
      recent: Array<{ fragmentId: string }>;
    };
    expect(memBody.recent.length).toBeGreaterThanOrEqual(3);

    await a2.close();
  });

  it("memory 模式：dataDir 缺省 → 不写盘", async () => {
    const a = await assembleServerAgent({ llmProvider: "mock" });
    expect(a.persistenceInfo.mode).toBe("memory");
    expect(a.persistenceInfo.dataDir).toBeUndefined();
    a.persistentStore.addLongTerm("ephemeral", { importance: 0.5 });
    await a.persistentStore.awaitPendingWrites();
    const all = await a.persistentStore.metadataStore.listFragmentMeta({});
    expect(all.length).toBe(1);
    await a.close();
  });

  it("OPENINTJ_DATA_DIR env 推断 dataDir", async () => {
    const dir = makeDir("envvar");
    const prev = process.env["OPENINTJ_DATA_DIR"];
    process.env["OPENINTJ_DATA_DIR"] = dir;
    try {
      const a = await assembleServerAgent({ llmProvider: "mock" });
      expect(a.persistenceInfo.mode).toBe("real");
      expect(a.persistenceInfo.dataDir).toBe(dir);
      await a.close();
    } finally {
      if (prev === undefined) delete process.env["OPENINTJ_DATA_DIR"];
      else process.env["OPENINTJ_DATA_DIR"] = prev;
    }
  });

  it("persistenceMode='real' 但缺 dataDir → 抛错", async () => {
    await expect(
      assembleServerAgent({
        llmProvider: "mock",
        persistenceMode: "real",
      }),
    ).rejects.toThrow(/dataDir/);
  });
});
