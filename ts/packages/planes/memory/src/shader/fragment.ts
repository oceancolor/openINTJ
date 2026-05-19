import {
  LODLevel,
  type ShadedFragment,
  type ShaderConfig,
  type ShaderModeType,
  type Summarizer,
  decayImportance,
  estimateTokens,
  getContentAtLod,
  truncateSummarize,
} from "@openintj/core";
import type { LodAssignment } from "./vertex.js";

export interface FragmentShaderOpts {
  config: Pick<ShaderConfig, "maxSummaryLength" | "recencyHalfLifeHours">;
  shaderMode: ShaderModeType;
  /** 总记忆 token 预算上限（tokens）。 */
  memoryBudgetTokens: number;
  summarize?: Summarizer;
  nowSec?: number;
}

/**
 * 片元着色阶段：按 LOD 生成最终内容 + token 预算控制。
 * - 优先使用 fragment.summaries[lod] 预生成摘要
 * - 否则用 summarize() 动态生成（默认 truncateSummarize）
 * - 强制不超过剩余 memoryBudget；超出则进一步压缩
 *
 * 对齐 Python memory_plane.ShaderPipeline._fragment_shader，但允许 async summarizer。
 */
export const fragmentShader = async (
  assignments: LodAssignment[],
  opts: FragmentShaderOpts,
): Promise<ShadedFragment[]> => {
  const summarize = opts.summarize ?? truncateSummarize;
  const now = opts.nowSec ?? Date.now() / 1000;
  let remaining = opts.memoryBudgetTokens;

  const out: ShadedFragment[] = [];
  for (const { ranked, lod } of assignments) {
    if (remaining <= 0) break;
    const fragment = ranked.fragment;

    let content = getContentAtLod(fragment, lod);
    if (lod !== LODLevel.LOD_0 && fragment.summaries[lod] === undefined) {
      const targetLen = Math.max(1, Math.floor(opts.config.maxSummaryLength / Math.max(1, lod)));
      content = await Promise.resolve(summarize(fragment.content, targetLen));
    }

    let tokens = estimateTokens(content);
    if (tokens > remaining) {
      // 强制进一步压缩（剩余预算映射回字符长度，乘 4）
      const forcedLen = Math.max(1, remaining * 4);
      content = await Promise.resolve(summarize(content, forcedLen));
      tokens = estimateTokens(content);
    }

    out.push({
      fragmentId: fragment.fragmentId,
      content,
      lod,
      score: Math.round(ranked.score * 10000) / 10000,
      shaderMode: opts.shaderMode,
      tokens,
      importance:
        Math.round(decayImportance(fragment, opts.config.recencyHalfLifeHours, now) * 10000) /
        10000,
    });

    remaining -= tokens;
  }

  return out;
};
