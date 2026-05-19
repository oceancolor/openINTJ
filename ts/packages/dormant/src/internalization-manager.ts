import { randomUUID } from "node:crypto";
import {
  type DormantPattern,
  type InternalizationProposal,
  InternalizationProposalSchema,
  type PersonaConfig,
  PersonaConfigSchema,
} from "./types.js";

export interface InternalizationOpts {
  /** 当 pattern 直接转成 proposal 时的字段映射器。 */
  mapToField?: (pattern: DormantPattern) =>
    | {
        targetField: string;
        value: unknown;
      }
    | undefined;
}

const setNested = (obj: Record<string, unknown>, path: string, value: unknown): void => {
  const keys = path.split(".");
  let cur: Record<string, unknown> = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const k = keys[i]!;
    const next = cur[k];
    if (!next || typeof next !== "object") {
      cur[k] = {};
    }
    cur = cur[k] as Record<string, unknown>;
  }
  cur[keys[keys.length - 1]!] = value;
};

const defaultMapToField = (
  p: DormantPattern,
): { targetField: string; value: unknown } | undefined => {
  switch (p.category) {
    case "preference":
      return { targetField: `preferences.${slug(p)}`, value: p.description };
    case "phrase":
      return { targetField: `phrases.${slug(p)}`, value: p.description };
    case "habit":
      return { targetField: `habits.${slug(p)}`, value: p.description };
    case "context":
      return { targetField: `context.${slug(p)}`, value: p.description };
    default:
      return undefined; // "other" 不自动建议
  }
};

const slug = (p: DormantPattern): string => p.patternId.replace(/[^a-zA-Z0-9_]/g, "_").slice(0, 16);

/**
 * InternalizationManager —— Pattern → PersonaConfig 写入闸门。
 *
 * 关键设计：所有变更都先生成 proposal，必须经用户审批（approve / reject）后
 * 才真正写入 PersonaConfig；避免 agent 自动修改用户人格设置。
 *
 * 工作流程：
 *  1) propose(pattern)：生成 proposal，进入 pending 队列
 *  2) listProposals(status="pending")：UI 拉取并展示
 *  3) approve(proposalId)：写入 PersonaConfig，标记 applied
 *     reject(proposalId)：标记 rejected
 *  4) snapshot()：返回当前 PersonaConfig
 */
export class InternalizationManager {
  private personaConfig: PersonaConfig;
  private proposals = new Map<string, InternalizationProposal>();
  private readonly opts: InternalizationOpts;

  constructor(initialConfig?: Partial<PersonaConfig>, opts: InternalizationOpts = {}) {
    this.personaConfig = PersonaConfigSchema.parse(initialConfig ?? {});
    this.opts = opts;
  }

  /** 把 pattern 转换成 proposal 入队（待审批）。 */
  propose(pattern: DormantPattern): InternalizationProposal | undefined {
    const mapper = this.opts.mapToField ?? defaultMapToField;
    const mapped = mapper(pattern);
    if (!mapped) return undefined;
    const p = InternalizationProposalSchema.parse({
      proposalId: randomUUID(),
      pattern,
      targetField: mapped.targetField,
      value: mapped.value,
      status: "pending",
      ts: Date.now(),
    });
    this.proposals.set(p.proposalId, p);
    return p;
  }

  /** 批量处理 mining 结果。 */
  proposeBatch(patterns: readonly DormantPattern[]): InternalizationProposal[] {
    const out: InternalizationProposal[] = [];
    for (const p of patterns) {
      const r = this.propose(p);
      if (r) out.push(r);
    }
    return out;
  }

  listProposals(status?: InternalizationProposal["status"]): InternalizationProposal[] {
    const arr = [...this.proposals.values()];
    if (status) return arr.filter((p) => p.status === status);
    return arr;
  }

  approve(proposalId: string): InternalizationProposal | undefined {
    const p = this.proposals.get(proposalId);
    if (!p || p.status !== "pending") return undefined;
    setNested(this.personaConfig as unknown as Record<string, unknown>, p.targetField, p.value);
    this.personaConfig.meta.lastUpdated = Date.now();
    this.personaConfig.meta.version += 1;
    p.status = "applied";
    p.decidedAt = Date.now();
    return p;
  }

  reject(proposalId: string): InternalizationProposal | undefined {
    const p = this.proposals.get(proposalId);
    if (!p || p.status !== "pending") return undefined;
    p.status = "rejected";
    p.decidedAt = Date.now();
    return p;
  }

  snapshot(): PersonaConfig {
    return PersonaConfigSchema.parse(JSON.parse(JSON.stringify(this.personaConfig)));
  }

  /** 重置（仅测试 / 用户主动清空）。 */
  reset(initialConfig?: Partial<PersonaConfig>): void {
    this.personaConfig = PersonaConfigSchema.parse(initialConfig ?? {});
    this.proposals.clear();
  }

  /**
   * 从持久化层恢复状态。仅 `DormantRuntime.hydrate()` 调；不会触发任何 lastUpdated / version 自增。
   *
   * - `proposals` 全量覆写当前内存 proposal 表
   * - `persona` 若提供则替换 personaConfig（保留其 meta.version / lastUpdated 原值）
   */
  restoreState(proposals: readonly InternalizationProposal[], persona?: PersonaConfig): void {
    this.proposals.clear();
    for (const p of proposals) {
      const parsed = InternalizationProposalSchema.parse(p);
      this.proposals.set(parsed.proposalId, parsed);
    }
    if (persona) {
      this.personaConfig = PersonaConfigSchema.parse(persona);
    }
  }
}
