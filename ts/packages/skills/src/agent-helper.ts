import type { EmbeddingProvider, HookBus, TaskTypeType } from "@openintj/core";
import { FsSkillSource, builtinSkillsDir, resolveSkillDirs } from "./fs-source.js";
import { SkillRegistry } from "./registry.js";
import { SkillSelector, type SkillSelectorOpts, renderSkillPrompt } from "./selector.js";

export interface SkillContext {
  /** 可用技能数量（>0 才值得注入）。 */
  readonly size: number;
  /**
   * 为一次 query 选择并渲染技能块（空串 = 未命中）。
   * 按 (taskType, query) 记忆化，避免同一轮多次 ReAct 迭代重复 embed。
   * 命中时（若给了 hooks）发 `event.SKILL_SELECTED`。
   */
  render(query: string, opts?: { taskType?: TaskTypeType; traceId?: string }): Promise<string>;
}

export interface AssembleSkillContextOpts {
  embedder: EmbeddingProvider;
  hooks?: HookBus;
  env?: NodeJS.ProcessEnv;
  /** 追加到「内建 + OPENINTJ_SKILLS_DIR」之外的技能目录。 */
  extraDirs?: readonly string[];
  /** 选择器参数覆盖（阈值 / topK / token 预算 / 加成）。 */
  selector?: Omit<SkillSelectorOpts, "registry" | "embedder">;
  /** 记忆化缓存上限，超过清空（防长会话无界增长）。默认 128。 */
  memoLimit?: number;
}

/**
 * 装配技能上下文（三端共用的 opt-in helper）：
 * 载入「内建 seed + `OPENINTJ_SKILLS_DIR` + extraDirs」下的 SKILL.md，用注入的 embedder 预计算向量，
 * 返回一个按 query 记忆化的 `render()`。没有任何可用技能时返回 `undefined`（调用方据此完全跳过注入）。
 *
 * 调用方负责 opt-in 门控（`OPENINTJ_SKILLS=1`）——只在开启时才 await 本函数。
 */
export async function assembleSkillContext(
  opts: AssembleSkillContextOpts,
): Promise<SkillContext | undefined> {
  const env = opts.env ?? process.env;
  const dirs = resolveSkillDirs(builtinSkillsDir(), env);
  for (const d of opts.extraDirs ?? []) if (!dirs.includes(d)) dirs.push(d);

  const registry = new SkillRegistry({
    sources: [new FsSkillSource({ dirs })],
    embedder: opts.embedder,
  });
  await registry.load();
  if (registry.size === 0) return undefined;

  const selector = new SkillSelector({
    registry,
    embedder: opts.embedder,
    ...(opts.selector ?? {}),
  });
  const memoLimit = opts.memoLimit ?? 128;
  const memo = new Map<string, Promise<string>>();

  return {
    get size(): number {
      return registry.size;
    },
    render(
      query: string,
      renderOpts: { taskType?: TaskTypeType; traceId?: string } = {},
    ): Promise<string> {
      const key = `${renderOpts.taskType ?? ""}\u0000${query}`;
      const cached = memo.get(key);
      if (cached) return cached;
      const p = (async (): Promise<string> => {
        const selected = await selector.select(
          query,
          renderOpts.taskType ? { taskType: renderOpts.taskType } : {},
        );
        if (selected.length === 0) return "";
        if (opts.hooks) {
          await opts.hooks.emit(
            "event.SKILL_SELECTED",
            { skills: selected.map((s) => ({ id: s.skill.id, score: s.score })), query },
            renderOpts.traceId ? { traceId: renderOpts.traceId } : {},
          );
        }
        return renderSkillPrompt(selected);
      })();
      if (memo.size >= memoLimit) memo.clear();
      memo.set(key, p);
      return p;
    },
  };
}
