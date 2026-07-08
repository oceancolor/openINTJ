import type { EmbeddingProvider } from "@openintj/core";
import { describe, expect, it } from "vitest";
import { DbSkillSource } from "../src/db-source.js";
import { SkillRegistry } from "../src/registry.js";
import type { Skill, SkillSource } from "../src/types.js";

const VOCAB = ["code", "review", "deploy", "weather"];
const bow = (t: string): number[] => VOCAB.map((w) => (t.toLowerCase().includes(w) ? 1 : 0));
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

const mkSkill = (id: string, description: string, source = "learned:db"): Skill => ({
  id,
  name: id,
  description,
  triggers: [],
  taskTypes: [],
  priority: 0,
  version: "1.0.0",
  tools: [],
  body: "body",
  source,
});

class FakeFs implements SkillSource {
  readonly name = "fake-fs";
  constructor(private readonly skills: Skill[]) {}
  async load(): Promise<Skill[]> {
    return this.skills;
  }
}

describe("DbSkillSource", () => {
  it("从 provider 供给已审批技能；reload 反映最新（approve 后新增）", async () => {
    let approved: Skill[] = [];
    const db = new DbSkillSource({ approvedSkills: () => approved });
    const registry = new SkillRegistry({ sources: [db], embedder: new BowEmbedder() });

    await registry.load();
    expect(registry.size).toBe(0);

    approved = [mkSkill("deploy-helper", "deploy service")];
    await registry.load();
    expect(registry.size).toBe(1);
    expect(registry.vectorFor("deploy-helper")).toBeDefined();
  });

  it("与 FsSkillSource 并列：后源（db）同 id 覆盖", async () => {
    const fs = new FakeFs([mkSkill("x", "code review", "fs")]);
    const db = new DbSkillSource({
      approvedSkills: () => [mkSkill("x", "db version", "learned:db")],
    });
    const registry = new SkillRegistry({ sources: [fs, db], embedder: new BowEmbedder() });
    await registry.load();
    expect(registry.size).toBe(1);
    expect(registry.list()[0]!.source).toBe("learned:db");
  });

  it("支持异步 provider", async () => {
    const db = new DbSkillSource({
      approvedSkills: async () => [mkSkill("a", "code")],
    });
    const registry = new SkillRegistry({ sources: [db], embedder: new BowEmbedder() });
    await registry.load();
    expect(registry.size).toBe(1);
  });
});
