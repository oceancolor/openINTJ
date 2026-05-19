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
    const decayed = decayImportance(a.ranked.fragment, halfLife, now);
    return decayed >= config.importanceThreshold;
  });

  return filtered.slice(0, config.maxFragmentsPerQuery);
};
