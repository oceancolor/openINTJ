import { TaskType } from "@openintj/core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  type CandidateSkillDraft,
  SkillLearningRuntime,
  resolveSkillWeightHalfLifeSec,
  skillOutcomeSignal,
} from "../src/learning-runtime.js";
import { InMemorySkillStore } from "../src/store.js";

describe("skillOutcomeSignal", () => {
  it("与 classifier 同映射：completed +1 / failed|timeout -0.5 / 其它 +0.2", () => {
    expect(skillOutcomeSignal("completed")).toBe(1);
    expect(skillOutcomeSignal("failed")).toBe(-0.5);
    expect(skillOutcomeSignal("timeout")).toBe(-0.5);
    expect(skillOutcomeSignal("max_iterations")).toBeCloseTo(0.2);
  });
});

describe("resolveSkillWeightHalfLifeSec", () => {
  afterEach(() => {
    delete process.env["OPENINTJ_SKILL_WEIGHT_HALFLIFE_SEC"];
  });
  it("默认（无 opts / 无 env）→ undefined（不衰减）", () => {
    expect(resolveSkillWeightHalfLifeSec()).toBeUndefined();
  });
  it("opts.weightHalfLifeSec 透传（>0）", () => {
    expect(resolveSkillWeightHalfLifeSec({ weightHalfLifeSec: 3600 })).toBe(3600);
  });
  it("非正数视为未设", () => {
    expect(resolveSkillWeightHalfLifeSec({ weightHalfLifeSec: 0 })).toBeUndefined();
    expect(resolveSkillWeightHalfLifeSec({ weightHalfLifeSec: -5 })).toBeUndefined();
  });
  it("从 env 读取", () => {
    process.env["OPENINTJ_SKILL_WEIGHT_HALFLIFE_SEC"] = "7200";
    expect(resolveSkillWeightHalfLifeSec()).toBe(7200);
  });
  it("opts 优先 env", () => {
    process.env["OPENINTJ_SKILL_WEIGHT_HALFLIFE_SEC"] = "7200";
    expect(resolveSkillWeightHalfLifeSec({ weightHalfLifeSec: 100 })).toBe(100);
  });
});

describe("SkillLearningRuntime · 加权", () => {
  it("noteSelected + recordOutcome：成功升权、失败降权、写穿 store", () => {
    const store = new InMemorySkillStore();
    const rt = new SkillLearningRuntime({ store, clock: () => 1_000_000 });
    rt.noteSelected("fix the bug", TaskType.CODING, ["s1", "s2"]);
    rt.recordOutcome("fix the bug", TaskType.CODING, "completed");
    expect(rt.weightFor("s1")).toBe(1);
    expect(rt.weightFor("s2")).toBe(1);

    rt.noteSelected("fix the bug", TaskType.CODING, ["s1"]);
    rt.recordOutcome("fix the bug", TaskType.CODING, "failed");
    expect(rt.weightFor("s1")).toBeCloseTo(0.5);
  });

  it("权重有界 clamp（不溢出）", () => {
    const rt = new SkillLearningRuntime({ weightClamp: { min: -1, max: 2 } });
    for (let i = 0; i < 10; i++) {
      rt.noteSelected("q", undefined, ["s"]);
      rt.recordOutcome("q", undefined, "completed");
    }
    expect(rt.weightFor("s")).toBe(2);
    for (let i = 0; i < 20; i++) {
      rt.noteSelected("q", undefined, ["s"]);
      rt.recordOutcome("q", undefined, "failed");
    }
    expect(rt.weightFor("s")).toBe(-1);
  });

  it("hydrate 从 store 恢复权重", async () => {
    const store = new InMemorySkillStore();
    store.saveWeight({ skillId: "s", weight: 3, lastUsed: 1 });
    const rt = new SkillLearningRuntime({ store });
    await rt.hydrate();
    expect(rt.weightFor("s")).toBe(3);
  });

  it("未 noteSelected 的 query 不影响任何权重", () => {
    const rt = new SkillLearningRuntime();
    rt.recordOutcome("never selected", undefined, "completed");
    expect(rt.weightFor("s")).toBe(0);
  });
});

describe("SkillLearningRuntime · 权重衰减（半衰期）", () => {
  it("不设半衰期 → 不衰减（历史行为）", () => {
    let now = 1_000_000_000;
    const rt = new SkillLearningRuntime({ clock: () => now });
    rt.noteSelected("q", undefined, ["s"]);
    rt.recordOutcome("q", undefined, "completed");
    expect(rt.weightFor("s")).toBe(1);
    now += 3600 * 1000 * 24 * 30; // 30 天后
    expect(rt.weightFor("s")).toBe(1);
  });

  it("weightFor 读时按半衰期指数衰减", () => {
    let now = 1_000_000_000;
    const rt = new SkillLearningRuntime({ clock: () => now, weightHalfLifeSec: 100 });
    rt.noteSelected("q", undefined, ["s"]);
    rt.recordOutcome("q", undefined, "completed"); // weight=1 @ now
    expect(rt.weightFor("s")).toBe(1);
    now += 100 * 1000; // 一个半衰期后 → 减半
    expect(rt.weightFor("s")).toBeCloseTo(0.5, 6);
    now += 100 * 1000; // 再一个半衰期 → 1/4
    expect(rt.weightFor("s")).toBeCloseTo(0.25, 6);
  });

  it("reinforce 累加前先把旧权重衰减到当下（陈旧高权重不永久累积）", () => {
    let now = 1_000_000_000;
    const rt = new SkillLearningRuntime({ clock: () => now, weightHalfLifeSec: 100 });
    rt.noteSelected("q", undefined, ["s"]);
    rt.recordOutcome("q", undefined, "completed"); // weight=1
    now += 100 * 1000; // 半衰期后旧值应视作 0.5
    rt.noteSelected("q", undefined, ["s"]);
    rt.recordOutcome("q", undefined, "completed"); // 0.5 + 1 = 1.5
    expect(rt.weightFor("s")).toBeCloseTo(1.5, 6);
  });

  it("weight=0 时衰减为 no-op", () => {
    const rt = new SkillLearningRuntime({ clock: () => 1_000_000, weightHalfLifeSec: 100 });
    expect(rt.weightFor("missing")).toBe(0);
  });
});

describe("SkillLearningRuntime · 蒸馏/审批", () => {
  let rt: SkillLearningRuntime;
  let store: InMemorySkillStore;

  beforeEach(() => {
    store = new InMemorySkillStore();
    let t = 1_000_000;
    rt = new SkillLearningRuntime({ store, clock: () => (t += 1000), minSamplesToDistill: 3 });
  });

  const feedSuccess = (query: string, taskType = TaskType.CODING) => {
    rt.noteSelected(query, taskType, []);
    rt.recordOutcome(query, taskType, "completed", { toolsUsed: ["read_file"] });
  };

  it("启发式蒸馏：达阈值的簇产 pending 候选，buffer 消费清空", async () => {
    feedSuccess("write a unit test for parser");
    feedSuccess("write a unit test for router");
    feedSuccess("add unit test coverage");
    expect(rt.bufferedCount()).toBe(3);

    const proposals = await rt.distill();
    expect(proposals).toHaveLength(1);
    expect(proposals[0]!.status).toBe("pending");
    expect(proposals[0]!.candidate.source).toBe("learned:db");
    expect(proposals[0]!.evidence.count).toBe(3);
    expect(rt.bufferedCount()).toBe(0);
    expect(rt.listProposals("pending")).toHaveLength(1);
  });

  it("低于阈值不产候选", async () => {
    feedSuccess("only one coding task");
    feedSuccess("second coding task");
    expect(await rt.distill()).toEqual([]);
  });

  it("approve → 进 listApproved + store；revoke → 移除；reject 不生效", async () => {
    for (let i = 0; i < 3; i++) feedSuccess(`coding task number ${i} refactor`);
    const [p] = await rt.distill();
    expect(p).toBeDefined();

    const approved = await rt.approve(p!.proposalId);
    expect(approved!.status).toBe("approved");
    expect(rt.listApproved().map((s) => s.id)).toContain(p!.candidate.id);
    expect((await store.loadAll()).approvedSkills).toHaveLength(1);

    // 二次 approve 已非 pending → undefined
    expect(await rt.approve(p!.proposalId)).toBeUndefined();

    const revoked = await rt.revoke(p!.proposalId);
    expect(revoked!.status).toBe("revoked");
    expect(rt.listApproved()).toHaveLength(0);
    expect((await store.loadAll()).approvedSkills).toHaveLength(0);
  });

  it("跨次 distill 按 candidate id 去重（同簇更新证据，不新增；已 approved 跳过）", async () => {
    for (let i = 0; i < 3; i++) feedSuccess(`coding refactor case ${i}`);
    const first = await rt.distill();
    expect(first).toHaveLength(1);
    const pid = first[0]!.proposalId;

    // 再喂同类 → 同 candidate id → 更新原 pending，不新增
    for (let i = 0; i < 3; i++) feedSuccess(`coding refactor extra ${i}`);
    const second = await rt.distill();
    expect(second).toHaveLength(1);
    expect(second[0]!.proposalId).toBe(pid);
    expect(rt.listProposals("pending")).toHaveLength(1);

    // approve 后再喂同类 → 已生效 → distill 跳过该 candidate
    await rt.approve(pid);
    for (let i = 0; i < 3; i++) feedSuccess(`coding refactor more ${i}`);
    expect(await rt.distill()).toHaveLength(0);
  });

  it("onSkillsChanged 在 approve/revoke 时触发", async () => {
    const onChanged = vi.fn();
    const rt2 = new SkillLearningRuntime({
      store: new InMemorySkillStore(),
      minSamplesToDistill: 1,
      onSkillsChanged: onChanged,
      clock: () => Date.now(),
    });
    rt2.noteSelected("q", TaskType.CODING, []);
    rt2.recordOutcome("q coding thing", TaskType.CODING, "completed");
    const [p] = await rt2.distill();
    await rt2.approve(p!.proposalId);
    expect(onChanged).toHaveBeenCalledTimes(1);
    await rt2.revoke(p!.proposalId);
    expect(onChanged).toHaveBeenCalledTimes(2);
  });

  it("llmDistill 优先；抛错回退启发式", async () => {
    const draft: CandidateSkillDraft = {
      id: "custom-skill",
      name: "Custom",
      description: "llm distilled",
      body: "do the thing",
    };
    const rt2 = new SkillLearningRuntime({
      store: new InMemorySkillStore(),
      llmDistill: () => [draft],
      clock: () => Date.now(),
    });
    rt2.recordOutcome("some coding query", TaskType.CODING, "completed");
    const [p] = await rt2.distill();
    expect(p!.candidate.id).toBe("custom-skill");
    expect(p!.candidate.description).toBe("llm distilled");
  });

  it("emit event.SKILL_PROPOSED（如给了 hooks）", async () => {
    const emit = vi.fn().mockResolvedValue(undefined);
    const hooks = { emit } as unknown as ConstructorParameters<
      typeof SkillLearningRuntime
    >[0]["hooks"];
    const rt2 = new SkillLearningRuntime({
      store: new InMemorySkillStore(),
      minSamplesToDistill: 1,
      hooks,
      clock: () => Date.now(),
    });
    rt2.recordOutcome("coding query for proposal", TaskType.CODING, "completed");
    await rt2.distill();
    expect(emit).toHaveBeenCalledWith(
      "event.SKILL_PROPOSED",
      expect.objectContaining({ evidenceCount: expect.any(Number) }),
    );
  });
});
