import type { EmbeddingProvider } from "@openintj/core";
import type { Skill, SkillSource } from "./types.js";

const embed = async (e: EmbeddingProvider, text: string): Promise<number[]> => {
  const r = e.embed(text);
  return r instanceof Promise ? await r : r;
};

/** 用于嵌入匹配的文本：名称 + 描述 + 触发词（触发词也纳入语义信号）。 */
const matchText = (s: Skill): string =>
  [s.name, s.description, s.triggers.join(" ")].filter((x) => x.length > 0).join(" — ");

export interface SkillRegistryOpts {
  sources: readonly SkillSource[];
  embedder: EmbeddingProvider;
}

/**
 * 技能注册表：从各来源载入技能，并用注入的 embedder 预计算「匹配向量」。
 * 启动时 `load()` 一次；之后 `list()` / `vectorFor()` 供选择器使用。
 */
export class SkillRegistry {
  private readonly sources: readonly SkillSource[];
  private readonly embedder: EmbeddingProvider;
  private skills: Skill[] = [];
  private readonly vectors = new Map<string, number[]>();

  constructor(opts: SkillRegistryOpts) {
    this.sources = opts.sources;
    this.embedder = opts.embedder;
  }

  /** 载入全部来源（后加载来源同 id 覆盖先加载）并预计算向量。多次调用安全（每次全量重建）。 */
  async load(): Promise<void> {
    const byId = new Map<string, Skill>();
    for (const src of this.sources) {
      let loaded: Skill[] = [];
      try {
        loaded = await src.load();
      } catch {
        loaded = [];
      }
      for (const s of loaded) byId.set(s.id, s);
    }
    this.skills = [...byId.values()];
    this.vectors.clear();
    for (const s of this.skills) {
      this.vectors.set(s.id, await embed(this.embedder, matchText(s)));
    }
  }

  list(): Skill[] {
    return this.skills;
  }

  get size(): number {
    return this.skills.length;
  }

  vectorFor(id: string): number[] | undefined {
    return this.vectors.get(id);
  }

  /** 轻量目录（第一级）：`- name: description`，供调试或未来 LLM-pick 用；默认激活走检索。 */
  catalog(): string {
    return this.skills.map((s) => `- ${s.name}: ${s.description}`).join("\n");
  }
}
