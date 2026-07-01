/**
 * `@openintj/skills` —— 技能系统（Phase 1：作者编写的 SKILL.md 能力包）。
 *
 * 两级激活，opt-in 降 token：
 *  1. 目录 + 嵌入检索预筛（复用 memory embedder）→ 只挑相关技能；
 *  2. 命中才把技能全文注入 system prompt（预算封顶）。
 *
 * 可插拔 `SkillSource`：Phase 1 只有 `FsSkillSource`；Phase 2 可加 DB 源承载「学习出来」的技能，
 * 注入点 / 选择器逻辑不变。默认全关（agent 侧 `OPENINTJ_SKILLS=1` 才装配）。
 */
export * from "./types.js";
export * from "./frontmatter.js";
export * from "./fs-source.js";
export * from "./registry.js";
export * from "./selector.js";
export * from "./agent-helper.js";
