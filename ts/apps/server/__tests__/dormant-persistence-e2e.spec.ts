/**
 * Dormant 持久化端到端（Phase 3.4 #9）。
 *
 * 装配 → record / mine / approve / reject → close → 重新装配 → hydrate → 状态恢复
 *
 * 默认 skip；需要 OPENINTJ_E2E=1 + better-sqlite3 已装。
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { assembleServerAgent } from "../src/agent.js";

const RUN_E2E = process.env["OPENINTJ_E2E"] === "1";
const describeE2E = RUN_E2E ? describe : describe.skip;

const fixtures: string[] = [];
const makeDir = (label: string): string => {
  const d = mkdtempSync(join(tmpdir(), `openintj-srv-dormant-${label}-`));
  fixtures.push(d);
  return d;
};
afterAll(() => {
  for (const d of fixtures) {
    try {
      rmSync(d, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
});

describeE2E("server agent dormant persistence (real)", { timeout: 30_000 }, () => {
  it("auto 模式：dataDir + enableDormant=true → 自动挂 SqliteDormantStore", async () => {
    const dir = makeDir("auto");
    const a = await assembleServerAgent({
      llmProvider: "mock",
      dataDir: dir,
      enableDormant: true,
      dormantOpts: {
        minerOpts: {
          ngramSize: 2,
          minFrequency: 2,
          minConfidence: 0.3,
          llmExtract: async (ng) => ({ description: ng, category: "preference" }),
        },
      },
    });
    expect(a.dormant).toBeDefined();
    expect(a.dormantPersistenceInfo).toBeDefined();
    expect(a.dormantPersistenceInfo!.adapter).toMatch(/^sqlite-dormant:/);
    expect(a.dormantPersistenceInfo!.dbPath).toBe(`${dir}/dormant.sqlite`);

    const s = await a.status();
    expect(s.dormant?.persistence?.adapter).toBe(a.dormantPersistenceInfo!.adapter);
    await a.close();
  });

  it("dormantPersistence='memory' → 不挂 adapter（即使有 dataDir）", async () => {
    const dir = makeDir("force-memory");
    const a = await assembleServerAgent({
      llmProvider: "mock",
      dataDir: dir,
      enableDormant: true,
      dormantPersistence: "memory",
    });
    expect(a.dormant).toBeDefined();
    expect(a.dormantPersistenceInfo).toBeUndefined();
    await a.close();
  });

  it("record + mine + approve → close → 重新装配 + hydrate → 状态恢复", async () => {
    const dir = makeDir("roundtrip");
    const llmExtract = async (ng: string) => ({ description: ng, category: "preference" as const });

    const a1 = await assembleServerAgent({
      llmProvider: "mock",
      dataDir: dir,
      enableDormant: true,
      dormantOpts: {
        minerOpts: { ngramSize: 2, minFrequency: 2, minConfidence: 0.3, llmExtract },
      },
    });
    expect(a1.dormant).toBeDefined();

    for (let i = 0; i < 5; i++) {
      a1.dormant!.record("绿 茶", "user", { iter: i });
    }
    expect(a1.dormant!.passiveSize()).toBe(5);

    const mineResult = await a1.dormant!.mine();
    expect(mineResult.proposals.length).toBeGreaterThan(0);

    const first = mineResult.proposals[0]!;
    const approved = a1.dormant!.approve(first.proposalId);
    expect(approved?.status).toBe("applied");

    const personaBefore = a1.dormant!.snapshot();
    const preferencesBefore = { ...personaBefore.preferences };
    const versionBefore = personaBefore.meta.version;

    await a1.close();

    const a2 = await assembleServerAgent({
      llmProvider: "mock",
      dataDir: dir,
      enableDormant: true,
      dormantOpts: {
        minerOpts: { ngramSize: 2, minFrequency: 2, minConfidence: 0.3, llmExtract },
      },
    });
    expect(a2.dormant).toBeDefined();
    expect(a2.dormant!.passiveSize()).toBe(5);

    const restoredProposals = a2.dormant!.listProposals();
    expect(restoredProposals.length).toBe(mineResult.proposals.length);
    expect(restoredProposals.find((p) => p.proposalId === first.proposalId)?.status).toBe(
      "applied",
    );

    const personaAfter = a2.dormant!.snapshot();
    expect(personaAfter.preferences).toEqual(preferencesBefore);
    expect(personaAfter.meta.version).toBe(versionBefore);

    await a2.close();
  });

  it("dormantDbPath 覆盖：dormant 库可与主 dataDir 分离", async () => {
    const dir = makeDir("custom-path");
    const dormantDir = makeDir("dormant-custom");
    const a = await assembleServerAgent({
      llmProvider: "mock",
      dataDir: dir,
      enableDormant: true,
      dormantDbPath: `${dormantDir}/my-dormant.sqlite`,
    });
    expect(a.dormantPersistenceInfo!.dbPath).toBe(`${dormantDir}/my-dormant.sqlite`);
    await a.close();
  });
});

describe("server agent dormant persistence (memory mode)", () => {
  it("不传 dataDir + enableDormant=true → dormantPersistenceInfo undefined", async () => {
    const a = await assembleServerAgent({
      llmProvider: "mock",
      enableDormant: true,
    });
    expect(a.dormant).toBeDefined();
    expect(a.dormantPersistenceInfo).toBeUndefined();
    const s = await a.status();
    expect(s.dormant?.persistence).toBeUndefined();
    await a.close();
  });

  it("dormantPersistence='real' 缺 dataDir 时立刻抛错", async () => {
    await expect(
      assembleServerAgent({
        llmProvider: "mock",
        enableDormant: true,
        dormantPersistence: "real",
      }),
    ).rejects.toThrow(/dormantPersistence='real' requires/);
  });
});
