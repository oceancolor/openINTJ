import type { TaskTypeType } from "@openintj/core";

/**
 * 一个「技能」= 作者编写的能力包（Phase 1 为上下文/指令包）。
 *
 * - `description` 既进轻量目录，又用于嵌入匹配（两级激活的第一级）。
 * - `body` 是命中后注入 system prompt 的指令正文（第二级：命中才加载全文，省 token）。
 * - `triggers` / `taskTypes` 为可选的低成本加权信号，不是硬门槛。
 */
export interface Skill {
  /** 稳定 id（frontmatter 未给时由 FsSkillSource 用目录/文件名兜底）。 */
  id: string;
  /** 展示名。 */
  name: string;
  /** 一句话描述：进目录 + 参与嵌入匹配。 */
  description: string;
  /** 关键词触发（命中子串时给相似度加成）。默认 []。 */
  triggers: string[];
  /** 关联的 TaskType（与分类器/检索标签呼应）。默认 []。 */
  taskTypes: TaskTypeType[];
  /** 排序/并列打破用，越大越优先。默认 0。 */
  priority: number;
  /** 版本，默认 "0.0.0"。 */
  version: string;
  /** 命中后注入的指令正文（frontmatter 之后的 markdown 主体）。 */
  body: string;
  /**
   * 该技能建议/绑定的工具子集（工具名）。默认 []（不约束）。
   * 文本协议下作为**软绑定**：命中时在技能块里提示「优先使用这些工具」，引导 agent 的 Action 选择；
   * 装配方也可用 {@link import("./agent-helper.js").SkillContext} 暴露的并集做更硬的工具面收窄。
   */
  tools: string[];
  /** 溯源标识（文件路径 / "db" 等），便于调试与去重。 */
  source: string;
}

/**
 * 技能来源（可插拔）：Phase 1 只有 {@link FsSkillSource}（SKILL.md）；
 * Phase 2 可加 DB 源承载「学习出来」的技能，注入点与选择器逻辑不变。
 */
export interface SkillSource {
  readonly name: string;
  /** 载入该来源下的全部技能（启动时调用一次；实现应自行兜底解析错误、不抛出致命异常）。 */
  load(): Promise<Skill[]>;
}

/** 选择器命中的技能 + 得分。 */
export interface SelectedSkill {
  skill: Skill;
  /** 0-1 融合得分（嵌入余弦 + 关键词/任务类型加成，封顶 1）。 */
  score: number;
}

/**
 * 技能选择的强化权重（Phase 2，与 classifier exemplar 权重同哲学）。
 * `recordOutcome` 时对本轮命中的技能按 outcome 信号累加，选择器据此做有界偏置。
 */
export interface SkillWeight {
  skillId: string;
  /** 累计权重（成功加、失败减；runtime 侧 clamp 到有界区间防溢出）。 */
  weight: number;
  /** 最近一次强化时间（秒），便于调试/未来 LRU。 */
  lastUsed: number;
}

/**
 * 从成功轨迹蒸馏出的候选技能提案（Phase 2，抄 dormant 的 InternalizationProposal 语义）。
 * 只进 `pending`，人审批（approve）才写入活跃 DB 源；永不自动生效。
 */
export interface SkillProposal {
  proposalId: string;
  /** 候选技能全文（审批通过后原样成为 DB 源里的 Skill）。 */
  candidate: Skill;
  /** 证据：支撑该候选的成功轨迹信息。 */
  evidence: {
    queries: string[];
    taskType?: TaskTypeType;
    count: number;
  };
  /** pending 待审 / approved 已批准生效 / rejected 拒绝 / revoked 批准后撤销。 */
  status: "pending" | "approved" | "rejected" | "revoked";
  ts: number;
  decidedAt?: number;
}
