import { describe, expect, it } from "vitest";
import { DormantRuntime } from "../src/index.js";

describe("DormantRuntime", () => {
  it("record → mine → propose 全链路", async () => {
    const rt = new DormantRuntime({
      minerOpts: { ngramSize: 3, minFrequency: 4, minConfidence: 0.4 },
    });
    const variants = [
      "我喜欢喝绿茶",
      "今天我喜欢喝绿茶",
      "我喜欢喝绿茶啊",
      "晚饭后喜欢喝绿茶",
      "其实我喜欢喝绿茶",
      "你知道我喜欢喝绿茶",
    ];
    for (const t of variants) rt.record(t, "user");
    expect(rt.passiveSize()).toBe(6);

    const r = await rt.mine();
    expect(r.scannedEvents).toBe(6);
    expect(r.patterns.length).toBeGreaterThan(0);
    // 默认 mapToField 把 "other" 类别忽略，PatternMiner 无 llmExtract 时全部 "other"
    // 所以 proposals 可能为空。这是预期：需要 llmExtract 或显式 category 才落地
    expect(r.proposals).toBeDefined();
  });

  it("record 落盘前自动脱敏（默认 redactor）", async () => {
    const rt = new DormantRuntime();
    rt.record("我的邮箱是 bob@example.com", "user");
    const ev = rt.passive.exportAll()[0]!;
    expect(ev.text).toContain("[REDACTED_EMAIL]");
    expect(ev.text).not.toContain("bob@example.com");
  });

  it("redactor=null 显式关闭脱敏", () => {
    const rt = new DormantRuntime({ redactor: null });
    rt.record("邮箱 bob@example.com", "user");
    expect(rt.passive.exportAll()[0]!.text).toContain("bob@example.com");
  });

  it("revoke 撤销已批准条目 → 从 persona 删除并 version 自增", async () => {
    const rt = new DormantRuntime({
      minerOpts: {
        ngramSize: 2,
        minFrequency: 3,
        minConfidence: 0.4,
        llmExtract: async (ngram) => ({ description: `偏好: ${ngram}`, category: "preference" }),
      },
    });
    for (let i = 0; i < 5; i++) rt.record("绿 茶 健 康", "user");
    const r = await rt.mine();
    const first = r.proposals[0]!;
    rt.approve(first.proposalId);
    expect(Object.keys(rt.snapshot().preferences).length).toBeGreaterThan(0);
    const versionAfterApprove = rt.snapshot().meta.version;

    const revoked = rt.revoke(first.proposalId);
    expect(revoked?.status).toBe("revoked");
    const persona = rt.snapshot();
    expect(Object.keys(persona.preferences).length).toBe(0);
    expect(persona.meta.version).toBe(versionAfterApprove + 1);

    // 二次撤销无效（已非 applied）
    expect(rt.revoke(first.proposalId)).toBeUndefined();
  });

  it("revoke 仅对 applied 生效：pending / 不存在返回 undefined", async () => {
    const rt = new DormantRuntime({
      minerOpts: {
        ngramSize: 2,
        minFrequency: 3,
        minConfidence: 0.4,
        llmExtract: async (ngram) => ({ description: ngram, category: "preference" }),
      },
    });
    for (let i = 0; i < 5; i++) rt.record("a b c", "user");
    const r = await rt.mine();
    const pending = r.proposals[0]!;
    expect(rt.revoke(pending.proposalId)).toBeUndefined(); // 还是 pending
    expect(rt.revoke("ghost")).toBeUndefined();
  });

  it("注入 llmExtract → proposals 落地 → approve 写入 PersonaConfig", async () => {
    const rt = new DormantRuntime({
      minerOpts: {
        ngramSize: 2,
        minFrequency: 3,
        minConfidence: 0.4,
        llmExtract: async (ngram) => ({
          description: `用户偏好: ${ngram}`,
          category: "preference",
        }),
      },
    });
    for (let i = 0; i < 5; i++) rt.record("绿 茶 健 康", "user");
    const r = await rt.mine();
    expect(r.proposals.length).toBeGreaterThan(0);

    const first = r.proposals[0]!;
    const approved = rt.approve(first.proposalId);
    expect(approved?.status).toBe("applied");

    const persona = rt.snapshot();
    expect(Object.keys(persona.preferences).length).toBeGreaterThan(0);
    expect(persona.meta.version).toBe(1);
  });

  it("reject 不会污染 persona", async () => {
    const rt = new DormantRuntime({
      minerOpts: {
        ngramSize: 2,
        minFrequency: 2,
        minConfidence: 0.3,
        llmExtract: async (ngram) => ({
          description: ngram,
          category: "preference",
        }),
      },
    });
    rt.record("讨 厌", "user");
    rt.record("讨 厌", "user");
    rt.record("讨 厌", "user");
    const { proposals } = await rt.mine();
    expect(proposals.length).toBeGreaterThan(0);
    rt.reject(proposals[0]!.proposalId);
    expect(Object.keys(rt.snapshot().preferences).length).toBe(0);
    expect(rt.listProposals("rejected").length).toBe(1);
  });

  it("PassiveStore 容量上限生效（环形丢弃）", () => {
    const rt = new DormantRuntime({ maxPassiveEvents: 3 });
    for (let i = 0; i < 10; i++) rt.record(`msg ${i}`, "user");
    expect(rt.passiveSize()).toBe(3);
  });

  it("eventId 自动递增 + 不冲突", () => {
    const rt = new DormantRuntime({ eventIdPrefix: "test" });
    const a = rt.record("一", "user");
    const b = rt.record("二", "agent");
    expect(a.eventId).not.toBe(b.eventId);
    expect(a.eventId.startsWith("test_1_")).toBe(true);
    expect(b.eventId.startsWith("test_2_")).toBe(true);
    expect(b.source).toBe("agent");
  });

  it("reset 清空状态", async () => {
    const rt = new DormantRuntime({
      minerOpts: {
        ngramSize: 2,
        minFrequency: 2,
        minConfidence: 0.3,
        llmExtract: async (ng) => ({ description: ng, category: "preference" }),
      },
    });
    for (let i = 0; i < 3; i++) rt.record("a b", "user");
    const { proposals } = await rt.mine();
    rt.approve(proposals[0]!.proposalId);
    expect(rt.snapshot().meta.version).toBe(1);

    rt.reset();
    expect(rt.passiveSize()).toBe(0);
    expect(rt.listProposals().length).toBe(0);
    expect(rt.snapshot().meta.version).toBe(0);
  });
});
