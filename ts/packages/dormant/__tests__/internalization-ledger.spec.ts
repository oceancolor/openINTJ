import { describe, expect, it } from "vitest";
import {
  type DormantPattern,
  type DormantProfileEntryLike,
  InternalizationManager,
  type PersonaLedgerPort,
  createProfileLedgerPort,
} from "../src/index.js";

interface FakeAssertion {
  assertionId: string;
  targetField: string;
  value: string;
  revoked: boolean;
}

const makeFakeLedger = () => {
  const assertions: FakeAssertion[] = [];
  const dismissed = new Set<string>();
  let seq = 0;
  const port: PersonaLedgerPort = {
    applyAssertion: ({ targetField, value }) => {
      seq += 1;
      const a: FakeAssertion = { assertionId: `a${seq}`, targetField, value, revoked: false };
      assertions.push(a);
      return a.assertionId;
    },
    revokeAssertion: (id) => {
      const a = assertions.find((x) => x.assertionId === id);
      if (!a || a.revoked) return false;
      a.revoked = true;
      return true;
    },
    dismissField: (field) => void dismissed.add(field),
    isFieldDismissed: (field) => dismissed.has(field),
    liveAssertions: () =>
      assertions
        .filter((a) => !a.revoked && !dismissed.has(a.targetField))
        .map(({ assertionId, targetField, value }) => ({ assertionId, targetField, value })),
  };
  return { port, assertions, dismissed };
};

const pattern = (over: Partial<DormantPattern> = {}): DormantPattern => ({
  patternId: "pat-tea",
  category: "preference",
  description: "喜欢喝茶",
  confidence: 0.9,
  frequency: 5,
  evidenceIds: ["evt-1"],
  ts: 2,
  ...over,
});

describe("G5.5 InternalizationManager × profile ledger", () => {
  it("approve 写穿账本，PersonaConfig 是账本 live 集的投影", () => {
    const fake = makeFakeLedger();
    const mgr = new InternalizationManager({}, { ledger: fake.port });
    const proposal = mgr.propose(pattern())!;
    expect(proposal.targetField).toBe("preferences.pat_tea");

    mgr.approve(proposal.proposalId);
    expect(fake.assertions).toHaveLength(1);
    expect(fake.assertions[0]).toMatchObject({ targetField: "preferences.pat_tea", value: "喜欢喝茶" });
    expect(mgr.snapshot().preferences).toEqual({ pat_tea: "喜欢喝茶" });
    expect(mgr.personaSystemPrompt()).toContain("喜欢喝茶");
  });

  it("revoke 只撤 dormant 层 + 登记抑制：同字段不再被提议（防白撤销）", () => {
    const fake = makeFakeLedger();
    const mgr = new InternalizationManager({}, { ledger: fake.port });
    const proposal = mgr.propose(pattern())!;
    mgr.approve(proposal.proposalId);

    mgr.revoke(proposal.proposalId);
    expect(fake.assertions[0]?.revoked).toBe(true);
    expect(fake.dismissed.has("preferences.pat_tea")).toBe(true);
    expect(mgr.snapshot().preferences).toEqual({});
    expect(mgr.personaSystemPrompt()).toBe("");
    // 同一 pattern 再次挖掘 → 抑制生效，不再产生提议。
    expect(mgr.propose(pattern())).toBeUndefined();
  });

  it("账本侧用户层遮蔽（applyAssertion 返回 undefined）时投影不显示该字段", () => {
    const fake = makeFakeLedger();
    const port: PersonaLedgerPort = {
      ...fake.port,
      applyAssertion: () => undefined,
      liveAssertions: () => [],
    };
    const mgr = new InternalizationManager({}, { ledger: port });
    const proposal = mgr.propose(pattern())!;
    const applied = mgr.approve(proposal.proposalId);
    expect(applied?.status).toBe("applied");
    // 用户显式值优先：persona 投影里不出现 dormant 的值。
    expect(mgr.snapshot().preferences).toEqual({});
  });

  it("重启恢复：新 manager 从账本 live 集重投影，撤销不依赖内存映射", () => {
    const fake = makeFakeLedger();
    const mgr1 = new InternalizationManager({}, { ledger: fake.port });
    const proposal = mgr1.propose(pattern())!;
    mgr1.approve(proposal.proposalId);

    // 模拟重启：新 manager 只拿到持久化的 proposals/persona，账本仍在。
    const mgr2 = new InternalizationManager({}, { ledger: fake.port });
    mgr2.restoreState(mgr1.listProposals(), { preferences: {}, phrases: {}, habits: {}, context: {} });
    expect(mgr2.snapshot().preferences).toEqual({ pat_tea: "喜欢喝茶" });

    mgr2.revoke(proposal.proposalId);
    expect(fake.assertions[0]?.revoked).toBe(true);
    expect(mgr2.snapshot().preferences).toEqual({});
  });

  it("无 ledger 时保持旧行为（本地 PersonaConfig 权威）", () => {
    const mgr = new InternalizationManager();
    const proposal = mgr.propose(pattern())!;
    mgr.approve(proposal.proposalId);
    expect(mgr.snapshot().preferences).toEqual({ pat_tea: "喜欢喝茶" });
    mgr.revoke(proposal.proposalId);
    expect(mgr.snapshot().preferences).toEqual({});
    // 旧行为：撤销后仍可重新提议（无抑制记忆）——与 ledger 模式的差异点。
    expect(mgr.propose(pattern())).toBeDefined();
  });
});

describe("G5.5 createProfileLedgerPort 字段映射", () => {
  it("phrases→habits、context→notes 映射且投影无损还原原始 targetField", () => {
    const entries: DormantProfileEntryLike[] = [];
    const dismissed = new Set<string>();
    let seq = 0;
    const port = createProfileLedgerPort({
      upsert: ({ section, key, value, provenance }) => {
        seq += 1;
        entries.push({ id: `e${seq}`, section, key, value, provenance });
        return `e${seq}`;
      },
      revokeAssertion: (id) => {
        const e = entries.find((x) => x.id === id);
        if (!e || e.revokedAt) return false;
        e.revokedAt = 1;
        return true;
      },
      dismiss: (section, key) => void dismissed.add(`${section}:${key}`),
      dismissedKeys: () => [...dismissed],
      dormantEntries: () => entries,
    });

    port.applyAssertion({ targetField: "phrases.morning", value: "早上好先喝水", provenance: "p1" });
    port.applyAssertion({ targetField: "context.city", value: "在成都", provenance: "p2" });
    expect(entries.map((e) => `${e.section}:${e.key}`)).toEqual(["habits:morning", "notes:city"]);
    // 投影还原用 provenance 里的原始 targetField，不受 section 映射影响。
    expect(port.liveAssertions().map((a) => a.targetField)).toEqual([
      "phrases.morning",
      "context.city",
    ]);

    port.dismissField("phrases.morning");
    expect(port.isFieldDismissed("phrases.morning")).toBe(true);
    expect(port.isFieldDismissed("habits.morning")).toBe(true); // 同账本键
  });
});
