import type { EmbeddingProvider, HookBus, TaskTypeType } from "@openintj/core";
import { FsSkillSource, builtinSkillsDir, resolveSkillDirs } from "./fs-source.js";
import { SkillRegistry } from "./registry.js";
import {
  SkillSelector,
  type SkillSelectorOpts,
  renderSkillPrompt,
  skillToolAllowlist,
} from "./selector.js";
import type { SkillSource } from "./types.js";

export interface SkillContext {
  /** 可用技能数量（>0 才值得注入）。 */
  readonly size: number;
  /**
   * 为一次 query 选择并渲染技能块（空串 = 未命中）。
   * 按 (taskType, query) 记忆化，避免同一轮多次 ReAct 迭代重复 embed。
   * 命中时（若给了 hooks）发 `event.SKILL_SELECTED`；并回调 `onSelected`（供自学习记住本轮命中）。
   */
  render(query: string, opts?: { taskType?: TaskTypeType; traceId?: string }): Promise<string>;
  /**
   * 重载技能（重新 load 各来源 + 重嵌入）并清空命中缓存。
   * 供自学习 approve/revoke 后调用，使新生效技能立刻可被选中，无需重启。
   */
  reload(): Promise<void>;
}

export interface AssembleSkillContextOpts {
  embedder: EmbeddingProvider;
  hooks?: HookBus;
  env?: NodeJS.ProcessEnv;
  /** 追加到「内建 + OPENINTJ_SKILLS_DIR」之外的技能目录。 */
  extraDirs?: readonly string[];
  /** 追加的非文件系统来源（Phase 2：DbSkillSource 承载已审批的学习技能）。 */
  extraSources?: readonly SkillSource[];
  /** 选择器参数覆盖（阈值 / topK / token 预算 / 加成）。 */
  selector?: Omit<SkillSelectorOpts, "registry" | "embedder" | "weightFor">;
  /** 强化权重供给（Phase 2 自学习）：给选择器做有界偏置，历史越用越准。 */
  weightFor?: (id: string) => number;
  /**
   * 命中回调（Phase 2 自学习）：render 命中后回传 (query, taskType, 命中技能 id, 工具子集并集)。
   * `tools` 为命中技能声明的工具子集去重并集（可空）——装配方可据此做工具面收窄。
   */
  onSelected?: (
    query: string,
    taskType: TaskTypeType | undefined,
    ids: string[],
    tools: string[],
  ) => void;
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
    sources: [new FsSkillSource({ dirs }), ...(opts.extraSources ?? [])],
    embedder: opts.embedder,
  });
  await registry.load();
  // Phase 2：即便当前无技能也可能因 approve 后重载出现，故仅当无任何来源时才跳过。
  if (registry.size === 0 && (opts.extraSources?.length ?? 0) === 0) return undefined;

  const selector = new SkillSelector({
    registry,
    embedder: opts.embedder,
    ...(opts.selector ?? {}),
    ...(opts.weightFor ? { weightFor: opts.weightFor } : {}),
  });
  const memoLimit = opts.memoLimit ?? 128;
  const memo = new Map<string, Promise<string>>();

  return {
    get size(): number {
      return registry.size;
    },
    async reload(): Promise<void> {
      await registry.load();
      memo.clear();
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
        const ids = selected.map((s) => s.skill.id);
        opts.onSelected?.(query, renderOpts.taskType, ids, skillToolAllowlist(selected));
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
