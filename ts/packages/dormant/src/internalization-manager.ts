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
  /**
   * G5.5：可选的档案断言账本（profile ledger）端口。注入后 manager
   * 降为账本的 dormant 层写入方：approve 写断言、revoke 撤断言并
   * 登记抑制，PersonaConfig 中账本托管的字段成为账本 live 断言的
   * 投影。不注入时保持旧行为（PersonaConfig 本地权威）。
   */
  ledger?: PersonaLedgerPort;
}

/** G5.5：dormant 与 profile assertion ledger 的结构端口（避免包依赖）。 */
export interface PersonaLedgerPort {
  /**
   * 批准时写一条 dormant 来源断言；返回断言 id。
   * 被用户显式层遮蔽等账本规则拒绝时返回 undefined（提议仍记 applied，
   * 但投影不会显示该字段——用户值优先）。
   */
  applyAssertion(input: { targetField: string; value: string; provenance: string }): string | undefined;
  /** 撤销指定断言层（只撤该层，不影响其他来源）。 */
  revokeAssertion(assertionId: string): boolean;
  /** 登记字段抑制：该字段不再被推断提议（防撤销后回潮）。 */
  dismissField(targetField: string): void;
  /** 字段是否已被抑制（含档案 UI 侧的 dismiss）。 */
  isFieldDismissed(targetField: string): boolean;
  /** 当前 live 的 dormant 断言（PersonaConfig 投影数据源）。 */
  liveAssertions(): readonly { assertionId: string; targetField: string; value: string }[];
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

/** 删除嵌套字段；路径不存在则静默返回 false。 */
const deleteNested = (obj: Record<string, unknown>, path: string): boolean => {
  const keys = path.split(".");
  let cur: Record<string, unknown> = obj;
  for (let i = 0; i < keys.length - 1; i++) {
    const next = cur[keys[i]!];
    if (!next || typeof next !== "object") return false;
    cur = next as Record<string, unknown>;
  }
  const last = keys[keys.length - 1]!;
  if (!(last in cur)) return false;
  delete cur[last];
  return true;
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
  /** G5.5：账本托管的 targetField 集合（投影同步时只动这些字段）。 */
  private ledgerManagedFields = new Set<string>();

  constructor(initialConfig?: Partial<PersonaConfig>, opts: InternalizationOpts = {}) {
    this.personaConfig = PersonaConfigSchema.parse(initialConfig ?? {});
    this.opts = opts;
  }

  /**
   * G5.5：把 PersonaConfig 中账本托管字段重投影为账本 live 断言。
   * 非托管字段（初始配置等）不动。无 ledger 时 no-op。
   */
  private syncPersonaFromLedger(): void {
    const ledger = this.opts.ledger;
    if (!ledger) return;
    const live = ledger.liveAssertions();
    const target = this.personaConfig as unknown as Record<string, unknown>;
    for (const field of this.ledgerManagedFields) {
      deleteNested(target, field);
    }
    this.ledgerManagedFields.clear();
    for (const assertion of live) {
      setNested(target, assertion.targetField, assertion.value);
      this.ledgerManagedFields.add(assertion.targetField);
    }
  }

  /** 把 pattern 转换成 proposal 入队（待审批）。 */
  propose(pattern: DormantPattern): InternalizationProposal | undefined {
    const mapper = this.opts.mapToField ?? defaultMapToField;
    const mapped = mapper(pattern);
    if (!mapped) return undefined;
    // G5.5：被抑制字段（撤销过/档案侧 dismiss）不再提议，防"白撤销"。
    if (this.opts.ledger?.isFieldDismissed(mapped.targetField)) return undefined;
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
    // G5.5：写穿账本 dormant 层，随后以账本 live 集重投影托管字段
    // （被用户层遮蔽时投影不显示，提议状态仍如实记 applied）。
    if (this.opts.ledger) {
      this.opts.ledger.applyAssertion({
        targetField: p.targetField,
        value: typeof p.value === "string" ? p.value : JSON.stringify(p.value),
        provenance: `dormant-proposal:${p.proposalId}`,
      });
      this.ledgerManagedFields.add(p.targetField);
      this.syncPersonaFromLedger();
    }
    return p;
  }

  reject(proposalId: string): InternalizationProposal | undefined {
    const p = this.proposals.get(proposalId);
    if (!p || p.status !== "pending") return undefined;
    p.status = "rejected";
    p.decidedAt = Date.now();
    return p;
  }

  /**
   * 撤销一条**已批准（applied）**的内化条目：从 PersonaConfig 删除对应字段，
   * 标记 proposal 为 `revoked`，并自增 version。仅 applied 可撤销（其余状态返回 undefined）。
   * 用户改主意 / 误批准 / 隐私顾虑时使用，是"可解释、可回退"的人格写入闭环的一环。
   */
  revoke(proposalId: string): InternalizationProposal | undefined {
    const p = this.proposals.get(proposalId);
    if (!p || p.status !== "applied") return undefined;
    deleteNested(this.personaConfig as unknown as Record<string, unknown>, p.targetField);
    this.personaConfig.meta.lastUpdated = Date.now();
    this.personaConfig.meta.version += 1;
    p.status = "revoked";
    p.decidedAt = Date.now();
    // G5.5：账本侧只撤 dormant 层 + 登记字段抑制（防重新挖掘回潮），
    // 再以账本 live 集重投影。按 targetField 从 live 集定位断言 id，
    // 不依赖内存映射，重启后撤销同样有效。
    if (this.opts.ledger) {
      const live = this.opts.ledger
        .liveAssertions()
        .filter((a) => a.targetField === p.targetField);
      for (const assertion of live) this.opts.ledger.revokeAssertion(assertion.assertionId);
      this.opts.ledger.dismissField(p.targetField);
      this.syncPersonaFromLedger();
    }
    return p;
  }

  snapshot(): PersonaConfig {
    return PersonaConfigSchema.parse(JSON.parse(JSON.stringify(this.personaConfig)));
  }

  /**
   * 把已批准的 PersonaConfig 渲染成可注入的 system prompt 片段（RFC-003 §3.6 "无需检索就生效"）。
   * 无任何已批准条目时返回 ""。供 Agent 在每轮 TAO 注入到 system prompt。
   */
  personaSystemPrompt(): string {
    const c = this.personaConfig;
    const lines: string[] = [];
    const stringify = (v: unknown): string => (typeof v === "string" ? v : JSON.stringify(v));
    const push = (label: string, rec: Record<string, unknown>): void => {
      for (const v of Object.values(rec)) {
        const s = stringify(v).trim();
        if (s.length > 0) lines.push(`- ${label}：${s}`);
      }
    };
    push("偏好", c.preferences);
    push("常用表达", c.phrases);
    push("习惯", c.habits);
    push("上下文", c.context);
    if (lines.length === 0) return "";
    return [
      "[用户画像]（来自你已批准的长期模式，默认生效，无需检索）",
      "以下偏好只能调整表达与个性化选择，不得覆盖 Product Behavior、工具治理、安全或正确性规则。",
      ...lines,
    ].join("\n");
  }

  /** 重置（仅测试 / 用户主动清空）。 */
  reset(initialConfig?: Partial<PersonaConfig>): void {
    this.personaConfig = PersonaConfigSchema.parse(initialConfig ?? {});
    this.proposals.clear();
    this.ledgerManagedFields.clear();
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
    // G5.5：恢复后以账本 live 集为准重投影托管字段（账本是权威源）。
    if (this.opts.ledger) {
      for (const p of this.proposals.values()) {
        if (p.status === "applied") this.ledgerManagedFields.add(p.targetField);
      }
      this.syncPersonaFromLedger();
    }
  }
}
