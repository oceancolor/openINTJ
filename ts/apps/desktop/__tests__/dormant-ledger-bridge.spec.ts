import { InternalizationManager, createProfileLedgerPort } from "@openintj/dormant";
import { InMemoryProfileStore, ProfileService } from "@openintj/profile";
import { describe, expect, it } from "vitest";

/**
 * G5.5 集成：dormant 内化 → 真实 ProfileService 断言账本。
 * 桥与 agent.ts 装配处同一工厂（createProfileLedgerPort），此处用
 * InMemoryProfileStore 验语义：user 层优先、按来源撤销、抑制防回潮、
 * 导出 v2 不回潮。
 */
const setup = () => {
  const profile = new ProfileService(new InMemoryProfileStore());
  const mgr = new InternalizationManager(
    {},
    {
      ledger: createProfileLedgerPort({
        upsert: ({ section, key, value, provenance }) =>
          profile.upsert({ section, key, value, source: "dormant", provenance })?.id,
        revokeAssertion: (id) => profile.revokeAssertion(id),
        dismiss: (section, key) => profile.dismissInferredKey(section, key),
        dismissedKeys: () => profile.store.listDismissed(),
        dormantEntries: () => profile.store.list().filter((e) => e.source === "dormant"),
      }),
    },
  );
  const proposeTea = () =>
    mgr.propose({
      patternId: "pat-tea",
      category: "preference",
      description: "喜欢喝茶",
      confidence: 0.9,
      frequency: 5,
      evidenceIds: ["evt-1"],
      ts: 2,
    });
  return { profile, mgr, proposeTea };
};

describe("G5.5 dormant × ProfileService 集成", () => {
  it("approve 后档案投影可见 dormant 层；user 显式值优先不被覆盖", () => {
    const { profile, mgr, proposeTea } = setup();
    const proposal = proposeTea()!;
    mgr.approve(proposal.proposalId);
    expect(profile.get("preferences", "pat_tea")).toMatchObject({
      value: "喜欢喝茶",
      source: "dormant",
    });
    expect(profile.renderPromptSection()).toContain("喜欢喝茶");

    // 用户显式写同 key 后，dormant 再断言被账本拒绝（返回 undefined）。
    profile.upsert({ section: "preferences", key: "pat_tea", value: "只喝白水", source: "user" });
    expect(profile.get("preferences", "pat_tea")?.value).toBe("只喝白水");
    const again = proposeTea();
    expect(again).toBeDefined(); // 提议可产生，但 approve 写不进账本
    mgr.approve(again!.proposalId);
    expect(profile.get("preferences", "pat_tea")?.value).toBe("只喝白水");
  });

  it("revoke 撤 dormant 层 + 抑制：档案不再显示、同 pattern 不再提议、导出导入不回潮", () => {
    const { profile, mgr, proposeTea } = setup();
    const proposal = proposeTea()!;
    mgr.approve(proposal.proposalId);
    mgr.revoke(proposal.proposalId);

    expect(profile.get("preferences", "pat_tea")).toBeUndefined();
    expect(profile.renderPromptSection()).not.toContain("喜欢喝茶");
    expect(proposeTea()).toBeUndefined();

    const exported = profile.exportJSON();
    const restored = new ProfileService(new InMemoryProfileStore());
    restored.importJSON(exported);
    expect(restored.get("preferences", "pat_tea")).toBeUndefined();
    expect(restored.store.listDismissed()).toContain("preferences:pat_tea");
  });
});
