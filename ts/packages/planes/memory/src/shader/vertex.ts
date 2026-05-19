import {
  LODLevel,
  type LODLevelType,
  ShaderMode,
  type ShaderModeType,
  getLodForMode,
} from "@openintj/core";
import type { RankedMemory } from "../retriever.js";

export interface LodAssignment {
  ranked: RankedMemory;
  lod: LODLevelType;
}

const clampLod = (n: number): LODLevelType => {
  const v = Math.max(0, Math.min(4, Math.round(n)));
  return v as LODLevelType;
};

/**
 * 顶点着色阶段：为每个记忆片段分配 LOD 级别。
 * 输入按得分降序，靠前的获得更高细节（更小 LOD 数字）。
 *
 * 策略对齐 Python memory_plane.ShaderPipeline._vertex_shader：
 *  - HYBRID:        前 30% LOD-1, 其余 LOD+1
 *  - HIGH_FIDELITY: score > 0.7 时 LOD-1, 否则 base
 *  - LOW_FIDELITY:  score < 0.3 时 LOD+1, 否则 base
 *  - ADAPTIVE:      使用 base lod
 */
export const vertexShader = (
  ranked: RankedMemory[],
  mode: ShaderModeType,
  budgetRatio: number,
): LodAssignment[] => {
  if (ranked.length === 0) return [];
  const baseLod = getLodForMode(mode, budgetRatio);
  const total = ranked.length;
  const hybridThreshold = Math.max(1, Math.floor(total * 0.3));

  return ranked.map((r, i): LodAssignment => {
    let lod: LODLevelType = baseLod;
    switch (mode) {
      case ShaderMode.HYBRID:
        lod = i < hybridThreshold ? clampLod(baseLod - 1) : clampLod(baseLod + 1);
        break;
      case ShaderMode.HIGH_FIDELITY:
        lod = r.score > 0.7 ? clampLod(baseLod - 1) : baseLod;
        break;
      case ShaderMode.LOW_FIDELITY:
        lod = r.score < 0.3 ? clampLod(baseLod + 1) : baseLod;
        break;
      case ShaderMode.ADAPTIVE:
      default:
        lod = baseLod;
        break;
    }
    // 极端情况：分数极低时再降一级（视锥体远端剔除替代）
    if (r.score < 0.05 && lod < LODLevel.LOD_4) {
      lod = clampLod(lod + 1);
    }
    return { ranked: r, lod };
  });
};
