import type { Skill, SkillSource } from "./types.js";

export interface DbSkillSourceOpts {
  /** 来源名（进 Skill.source 溯源用外层已定，这里仅注册表调试）。默认 "db-skills"。 */
  name?: string;
  /**
   * 供给已审批生效的学习技能。通常传 `() => runtime.listApproved()` 读活跃内存状态，
   * 也可传直接读 store 的异步函数。注册表 `load()` 时调用。
   */
  approvedSkills: () => Skill[] | Promise<Skill[]>;
}

/**
 * DB 技能源（Phase 2）：把「学习+审批」出来的技能供给 {@link SkillRegistry}。
 * 与 `FsSkillSource` 并列进注册表，靠注册表既有「后源同 id 覆盖」规则合并。
 * 本身无状态，只是 approved 供给函数的适配器 —— approve/revoke 后由 `registry.load()` 拉取最新。
 */
export class DbSkillSource implements SkillSource {
  readonly name: string;
  private readonly provider: () => Skill[] | Promise<Skill[]>;

  constructor(opts: DbSkillSourceOpts) {
    this.name = opts.name ?? "db-skills";
    this.provider = opts.approvedSkills;
  }

  async load(): Promise<Skill[]> {
    const r = this.provider();
    return r instanceof Promise ? await r : r;
  }
}
