import type { EmbeddingProvider } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { SkillRegistry } from "../src/registry.js";
import { SkillSelector, renderSkillPrompt, skillToolAllowlist } from "../src/selector.js";
import type { Skill, SkillSource } from "../src/types.js";

// 确定性 bag-of-words embedder：让匹配可断言（SimpleEmbedder 是哈希向量，语义不可控）。
const VOCAB = ["code", "review", "bug", "weather", "travel"];
const bow = (text: string): number[] => VOCAB.map((w) => (text.toLowerCase().includes(w) ? 1 : 0));
class BowEmbedder implements EmbeddingProvider {
  readonly name = "bow";
  readonly dimension = VOCAB.length;
  embed(t: string): number[] {
    return bow(t);
  }
  embedBatch(ts: readonly string[]): number[][] {
    return ts.map(bow);
  }
}

const mkSkill = (over: Partial<Skill> & Pick<Skill, "id">): Skill => ({
  name: over.id,
  description: "",
  triggers: [],
  taskTypes: [],
  priority: 0,
  version: "1.0.0",
  tools: [],
  body: "body",
  source: "test",
  ...over,
});

class MemSource implements SkillSource {
  readonly name = "mem";
  constructor(private readonly skills: Skill[]) {}
  async load(): Promise<Skill[]> {
    return this.skills;
  }
}

const buildRegistry = async (skills: Skill[]) => {
  const embedder = new BowEmbedder();
  const registry = new SkillRegistry({ sources: [new MemSource(skills)], embedder });
  await registry.load();
  return { registry, embedder };
};

describe("SkillRegistry", () => {
  it("load 后 size / list / 向量就绪；后源覆盖同 id", async () => {
    const a = new MemSource([mkSkill({ id: "x", description: "first" })]);
    const b = new MemSource([mkSkill({ id: "x", description: "second" })]);
    const registry = new SkillRegistry({ sources: [a, b], embedder: new BowEmbedder() });
    await registry.load();
    expect(registry.size).toBe(1);
    expect(registry.list()[0]!.description).toBe("second");
    expect(registry.vectorFor("x")).toBeDefined();
  });
});

describe("SkillSelector", () => {
  it("嵌入相似命中、无关不命中", async () => {
    const skills = [
      mkSkill({ id: "code-review", description: "code review bug", body: "审代码" }),
      mkSkill({ id: "trip", description: "weather travel", body: "行程" }),
    ];
    const { registry, embedder } = await buildRegistry(skills);
    const sel = new SkillSelector({ registry, embedder });

    const hit = await sel.select("please review my code for a bug");
    expect(hit).toHaveLength(1);
    expect(hit[0]!.skill.id).toBe("code-review");
    expect(hit[0]!.score).toBeGreaterThanOrEqual(0.35);

    const miss = await sel.select("stock market news today");
    expect(miss).toEqual([]);
  });

  it("关键词 trigger 命中加成能把边缘项抬过阈值", async () => {
    const skills = [
      mkSkill({ id: "deploy", description: "zzz", triggers: ["deploy"], body: "步骤" }),
    ];
    const { registry, embedder } = await buildRegistry(skills);
    // description 无 vocab 词 → 余弦 0；仅靠 trigger 子串加成。
    const sel = new SkillSelector({ registry, embedder, minScore: 0.1, keywordBoost: 0.5 });
    const hit = await sel.select("how do I deploy this service");
    expect(hit).toHaveLength(1);
    expect(hit[0]!.skill.id).toBe("deploy");
  });

  it("topK 限制命中数量", async () => {
    const skills = [
      mkSkill({ id: "a", description: "code review bug" }),
      mkSkill({ id: "b", description: "code review" }),
      mkSkill({ id: "c", description: "code bug" }),
    ];
    const { registry, embedder } = await buildRegistry(skills);
    const sel = new SkillSelector({ registry, embedder, topK: 2, minScore: 0.1 });
    const hit = await sel.select("code review bug");
    expect(hit.length).toBeLessThanOrEqual(2);
  });

  it("token 预算封顶：至少保留最高分一个，超预算的次项被裁", async () => {
    const big = "x ".repeat(2000); // ~ estimateTokens 远超预算
    const skills = [
      mkSkill({ id: "top", description: "code review bug", body: "small" }),
      mkSkill({ id: "second", description: "code review", body: big }),
    ];
    const { registry, embedder } = await buildRegistry(skills);
    const sel = new SkillSelector({
      registry,
      embedder,
      topK: 2,
      minScore: 0.1,
      maxBodyTokens: 50,
    });
    const hit = await sel.select("code review bug");
    expect(hit).toHaveLength(1);
    expect(hit[0]!.skill.id).toBe("top");
  });

  it("weightFor 偏置能改并列项排序，且封顶不压过语义", async () => {
    const skills = [
      mkSkill({ id: "a", description: "code review" }),
      mkSkill({ id: "b", description: "code review" }),
    ];
    const { registry, embedder } = await buildRegistry(skills);
    // 两者语义相同（同 description，query "code review bug" 下余弦 ~0.82 未封顶）；
    // 给 b 正权重偏置应把 b 抬到最前。
    const sel = new SkillSelector({
      registry,
      embedder,
      topK: 2,
      minScore: 0.1,
      weightFor: (id) => (id === "b" ? 6 : 0),
      weightGain: 0.05,
    });
    const hit = await sel.select("code review bug");
    expect(hit[0]!.skill.id).toBe("b");

    // 封顶：即便权重巨大，偏置也被 cap 到 0.3，不会让低语义项越过高语义项。
    const skills2 = [
      mkSkill({ id: "strong", description: "code review bug" }), // 语义命中
      mkSkill({ id: "weak", description: "weather travel" }), // 语义不相关
    ];
    const { registry: r2, embedder: e2 } = await buildRegistry(skills2);
    const sel2 = new SkillSelector({
      registry: r2,
      embedder: e2,
      topK: 1,
      minScore: 0.1,
      weightFor: (id) => (id === "weak" ? 1000 : 0),
      weightGain: 0.05,
      weightBiasCap: 0.3,
    });
    const hit2 = await sel2.select("code review bug");
    expect(hit2[0]!.skill.id).toBe("strong");
  });

  it("空查询 / 空注册表 → 空结果", async () => {
    const { registry, embedder } = await buildRegistry([]);
    const sel = new SkillSelector({ registry, embedder });
    expect(await sel.select("anything")).toEqual([]);
    const { registry: r2, embedder: e2 } = await buildRegistry([
      mkSkill({ id: "a", description: "code" }),
    ]);
    expect(await new SkillSelector({ registry: r2, embedder: e2 }).select("   ")).toEqual([]);
  });
});

describe("renderSkillPrompt", () => {
  it("无命中返回空串；有命中拼成技能块", () => {
    expect(renderSkillPrompt([])).toBe("");
    const block = renderSkillPrompt([
      { skill: mkSkill({ id: "cr", name: "Code Review", body: "审查步骤" }), score: 0.9 },
    ]);
    expect(block).toContain("[技能]");
    expect(block).toContain("## Code Review");
    expect(block).toContain("审查步骤");
  });

  it("声明了 tools 的技能渲染出硬隔离工具行", () => {
    const block = renderSkillPrompt([
      {
        skill: mkSkill({
          id: "cr",
          name: "Code Review",
          body: "审查",
          tools: ["readFile", "search"],
        }),
        score: 0.9,
      },
    ]);
    expect(block).toContain("本轮仅可使用工具：read_file, search");
  });

  it("未声明 tools 时不渲染软绑定行", () => {
    const block = renderSkillPrompt([
      { skill: mkSkill({ id: "cr", name: "Code Review", body: "审查" }), score: 0.9 },
    ]);
    expect(block).not.toContain("本轮仅可使用工具");
  });
});

describe("skillToolAllowlist", () => {
  it("命中技能工具子集并集：去重、保序", () => {
    const allow = skillToolAllowlist([
      { skill: mkSkill({ id: "a", tools: ["readFile", "search"] }), score: 0.9 },
      { skill: mkSkill({ id: "b", tools: ["search", "writeFile"] }), score: 0.8 },
      { skill: mkSkill({ id: "c" }), score: 0.7 },
    ]);
    expect(allow).toEqual(["read_file", "search", "write_file"]);
  });

  it("无命中 / 均未声明 tools → 空数组", () => {
    expect(skillToolAllowlist([])).toEqual([]);
    expect(skillToolAllowlist([{ skill: mkSkill({ id: "a" }), score: 0.9 }])).toEqual([]);
  });
});
