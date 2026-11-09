import { type ShaderConfig, decayImportance } from "@openintj/core";
import type { LodAssignment } from "./vertex.js";

export interface GeometryFilterOpts {
  config: Pick<
    ShaderConfig,
    "importanceThreshold" | "maxFragmentsPerQuery" | "recencyHalfLifeHours"
  >;
  /** 当前时间（秒）；默认 Date.now()/1000，可注入用于测试。 */
  nowSec?: number;
}

/**
 * 几何着色阶段：剔除不达标 + 限流。
 *
 * 等价 3D 视锥体剔除：
 *  1. 重要性阈值过滤（衰减后）
 *  2. maxFragmentsPerQuery 限流（保持得分顺序）
 *
 * 对齐 Python memory_plane.ShaderPipeline._geometry_shader。
 */
export const geometryShader = (
  assignments: LodAssignment[],
  opts: GeometryFilterOpts,
): LodAssignment[] => {
  const { config } = opts;
  const now = opts.nowSec ?? Date.now() / 1000;
  const halfLife = config.recencyHalfLifeHours;

  const filtered = assignments.filter((a) => {
    // F08：稳定 long_term 不做统一硬时间淘汰——时间只参与排名（vertex 已计
    // recency 分量），不用 24h 半衰期把旧事实直接剔除。短期/工作记忆保留衰减门。
    if (a.ranked.fragment.memoryType === "long_term") {
      return a.ranked.fragment.importance >= config.importanceThreshold;
    }
    const decayed = decayImportance(a.ranked.fragment, halfLife, now);
    return decayed >= config.importanceThreshold;
  });

  return filtered.slice(0, config.maxFragmentsPerQuery);
};
