/**
 * 跨入口（cli / server / desktop）共用的「钝化记忆 persona 注入」开关解析。
 *
 * RFC-003 §3.5「教坏 agent」缓解 + §3.6 验收 #3「可观测的 A/B 差异」：
 * persona 注入需要一个显式的 on/off 杠杆，便于灰度、A/B 对照、以及用户级关停。
 *
 * 语义：仅当 dormant 子系统已启用时才有意义；此处只解析「是否把已批准 persona
 * 注入 system prompt」，不 import dormant / 执行平面，避免 shared 反向依赖。
 *
 * 优先级：显式 opts.enablePersona > env OPENINTJ_PERSONA > 默认开。
 * - `OPENINTJ_PERSONA=0` / `false` → 关闭注入（A 组：无 persona 基线）
 * - 其余（含未设置）→ 开启注入（B 组：注入已批准 persona）
 */
export interface PersonaInjectionInput {
  /** 显式开关；不传则回退 env / 默认。 */
  enablePersona?: boolean;
}

export const resolvePersonaInjection = (opts: PersonaInjectionInput = {}): boolean => {
  if (opts.enablePersona !== undefined) return opts.enablePersona;
  const env = process.env["OPENINTJ_PERSONA"];
  if (env === "0" || env === "false") return false;
  return true;
};
