import type { EmbeddingProvider } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { SkillRegistry } from "../src/registry.js";
import { SkillSelector, renderSkillPrompt } from "../src/selector.js";
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
});
