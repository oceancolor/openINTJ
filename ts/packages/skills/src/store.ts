import type { Skill, SkillProposal, SkillWeight } from "./types.js";

/** 一次性载入的持久化快照（hydrate 用）。 */
export interface SkillStoreSnapshot {
  /** 已审批生效的「学习技能」（DbSkillSource 据此供给注册表）。 */
  approvedSkills: Skill[];
  /** 全部提案（含各种状态）。 */
  proposals: SkillProposal[];
  /** 技能选择强化权重。 */
  weights: SkillWeight[];
}

/**
 * 技能自学习持久化接口（Phase 2）。
 *
 * 约定沿用 dormant/classifier：**接口在领域包（这里）、SQLite 实现在 `@openintj/storage-sqlite`**。
 * - `loadAll()` 只在 `hydrate()` 调一次；
 * - 其余写方法是热路径，**同步、不得抛错**（实现层自行 try/catch + log）；
 * - `close()` 只释放句柄（写已写穿）。
 */
export interface SkillStore {
  readonly name: string;
  loadAll(): Promise<SkillStoreSnapshot>;
  upsertProposal(proposal: SkillProposal): void;
  /** approve 时写入 / 覆盖一条生效学习技能。 */
  upsertApprovedSkill(skill: Skill): void;
  /** revoke 时移除一条生效学习技能。 */
  removeApprovedSkill(skillId: string): void;
  saveWeight(weight: SkillWeight): void;
  clearAll(): void;
  close(): Promise<void>;
}

/**
 * 纯内存实现（默认 / 测试 / 非持久化模式）。语义与未来 SQLite 实现一致。
 */
export class InMemorySkillStore implements SkillStore {
  readonly name = "in-memory-skills";
  private readonly approved = new Map<string, Skill>();
  private readonly proposals = new Map<string, SkillProposal>();
  private readonly weights = new Map<string, SkillWeight>();

  async loadAll(): Promise<SkillStoreSnapshot> {
    return {
      approvedSkills: [...this.approved.values()],
      proposals: [...this.proposals.values()],
      weights: [...this.weights.values()],
    };
  }

  upsertProposal(proposal: SkillProposal): void {
    this.proposals.set(proposal.proposalId, { ...proposal });
  }

  upsertApprovedSkill(skill: Skill): void {
    this.approved.set(skill.id, { ...skill });
  }

  removeApprovedSkill(skillId: string): void {
    this.approved.delete(skillId);
  }

  saveWeight(weight: SkillWeight): void {
    this.weights.set(weight.skillId, { ...weight });
  }

  clearAll(): void {
    this.approved.clear();
    this.proposals.clear();
    this.weights.clear();
  }

  async close(): Promise<void> {
    // 内存实现无句柄。
  }
}
