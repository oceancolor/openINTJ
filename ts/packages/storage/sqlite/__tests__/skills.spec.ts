import type { Skill, SkillProposal, SkillWeight } from "@openintj/skills";
/**
 * SqliteSkillStore 单测（技能自学习 Phase 2）。
 * 走 `:memory:`（不触盘）；better-sqlite3 peer 未装则整段 skip。
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SqliteSkillStore, createSqliteSkillStore } from "../src/skills.js";

const HAS_PEER = await (async () => {
  try {
    await import("better-sqlite3");
    return true;
  } catch {
    return false;
  }
})();

const describeIfPeer = HAS_PEER ? describe : describe.skip;

const mkSkill = (id: string): Skill => ({
  id,
  name: `Skill ${id}`,
  description: `desc ${id}`,
  triggers: ["t1", "t2"],
  taskTypes: [],
  priority: 1,
  version: "1.0.0",
  body: `body ${id}`,
  source: "learned:db",
});

const mkProposal = (id: string, status: SkillProposal["status"] = "pending"): SkillProposal => ({
  proposalId: id,
  candidate: mkSkill(`cand-${id}`),
  evidence: { queries: ["q1", "q2"], count: 2 },
  status,
  ts: 1000,
  ...(status !== "pending" ? { decidedAt: 2000 } : {}),
});

describeIfPeer("SqliteSkillStore (:memory:)", () => {
  let store: SqliteSkillStore;

  beforeEach(async () => {
    store = await createSqliteSkillStore({ dbPath: ":memory:" });
  });

  afterEach(async () => {
    await store.close();
  });

  it("approvedSkills / proposals / weights 往返", async () => {
    store.upsertApprovedSkill(mkSkill("a"));
    store.upsertProposal(mkProposal("p1"));
    const w: SkillWeight = { skillId: "a", weight: 2.5, lastUsed: 42 };
    store.saveWeight(w);

    const snap = await store.loadAll();
    expect(snap.approvedSkills).toHaveLength(1);
    expect(snap.approvedSkills[0]!.id).toBe("a");
    expect(snap.approvedSkills[0]!.body).toBe("body a");
    expect(snap.proposals).toHaveLength(1);
    expect(snap.proposals[0]!.candidate.id).toBe("cand-p1");
    expect(snap.weights).toEqual([{ skillId: "a", weight: 2.5, lastUsed: 42 }]);
  });

  it("upsert 覆盖同 id / proposal 状态迁移", async () => {
    store.upsertProposal(mkProposal("p1", "pending"));
    store.upsertProposal(mkProposal("p1", "approved"));
    const snap = await store.loadAll();
    expect(snap.proposals).toHaveLength(1);
    expect(snap.proposals[0]!.status).toBe("approved");
    expect(snap.proposals[0]!.decidedAt).toBe(2000);
  });

  it("removeApprovedSkill / clearAll", async () => {
    store.upsertApprovedSkill(mkSkill("a"));
    store.upsertApprovedSkill(mkSkill("b"));
    store.removeApprovedSkill("a");
    let snap = await store.loadAll();
    expect(snap.approvedSkills.map((s) => s.id)).toEqual(["b"]);

    store.upsertProposal(mkProposal("p1"));
    store.saveWeight({ skillId: "b", weight: 1, lastUsed: 1 });
    store.clearAll();
    snap = await store.loadAll();
    expect(snap.approvedSkills).toHaveLength(0);
    expect(snap.proposals).toHaveLength(0);
    expect(snap.weights).toHaveLength(0);
  });

  it("saveWeight 覆盖同 skillId", async () => {
    store.saveWeight({ skillId: "a", weight: 1, lastUsed: 1 });
    store.saveWeight({ skillId: "a", weight: 3, lastUsed: 9 });
    const snap = await store.loadAll();
    expect(snap.weights).toEqual([{ skillId: "a", weight: 3, lastUsed: 9 }]);
  });

  it("关闭后 loadAll 抛错", async () => {
    await store.close();
    await expect(store.loadAll()).rejects.toThrow(/not initialized/);
  });
});
