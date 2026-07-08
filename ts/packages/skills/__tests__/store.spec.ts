import { describe, expect, it } from "vitest";
import { InMemorySkillStore } from "../src/store.js";
import type { Skill, SkillProposal, SkillWeight } from "../src/types.js";

const mkSkill = (id: string): Skill => ({
  id,
  name: id,
  description: `${id} desc`,
  triggers: [],
  taskTypes: [],
  priority: 0,
  version: "1.0.0",
  tools: [],
  body: "body",
  source: "learned:db",
});

const mkProposal = (id: string, status: SkillProposal["status"] = "pending"): SkillProposal => ({
  proposalId: id,
  candidate: mkSkill(`cand-${id}`),
  evidence: { queries: ["q1", "q2"], count: 2 },
  status,
  ts: 1000,
});

describe("InMemorySkillStore", () => {
  it("往返：approvedSkills / proposals / weights，upsert 覆盖同 id", async () => {
    const store = new InMemorySkillStore();
    store.upsertApprovedSkill(mkSkill("a"));
    store.upsertProposal(mkProposal("p1"));
    store.upsertProposal(mkProposal("p1", "approved")); // 覆盖
    const w: SkillWeight = { skillId: "a", weight: 2.5, lastUsed: 42 };
    store.saveWeight(w);

    const snap = await store.loadAll();
    expect(snap.approvedSkills).toHaveLength(1);
    expect(snap.approvedSkills[0]!.id).toBe("a");
    expect(snap.proposals).toHaveLength(1);
    expect(snap.proposals[0]!.status).toBe("approved");
    expect(snap.weights).toEqual([{ skillId: "a", weight: 2.5, lastUsed: 42 }]);
  });

  it("removeApprovedSkill / clearAll", async () => {
    const store = new InMemorySkillStore();
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

  it("存的是副本（外部改动不影响已存）", async () => {
    const store = new InMemorySkillStore();
    const s = mkSkill("a");
    store.upsertApprovedSkill(s);
    s.body = "mutated";
    const snap = await store.loadAll();
    expect(snap.approvedSkills[0]!.body).toBe("body");
  });
});
