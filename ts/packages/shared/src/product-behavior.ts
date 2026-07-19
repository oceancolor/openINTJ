/**
 * RFC-006：版本化 Product Behavior 契约（全用户一致，与 Dormant User Persona 分层）。
 * 不声称 MBTI 类型；只定义可观察工程行为。
 */

export const PRODUCT_BEHAVIOR_VERSION = "1.0.0";

/** 可机器评测的行为 trait 标识。 */
export const ProductTrait = {
  STRATEGIC_DECOMPOSITION: "T1_strategic_decomposition",
  STRUCTURED_REASONING: "T2_structured_reasoning",
  EVIDENCE_FIRST: "T3_evidence_first",
  DIRECT_CONCISE: "T4_direct_concise",
  CLARIFY_WHEN_NEEDED: "T5_clarify_when_needed",
  INDEPENDENT_EXECUTION: "T6_independent_execution",
  QUALITY_GATE: "T7_quality_gate",
  TOOL_GOVERNANCE_RESPECT: "T8_tool_governance_respect",
} as const;

export type ProductTraitId = (typeof ProductTrait)[keyof typeof ProductTrait];

export interface ProductBehaviorOpts {
  /** A/B 杠杆：false 时不注入产品行为层（基线组）。默认开。 */
  enabled?: boolean;
  /** 显式版本覆盖（评测用）。 */
  version?: string;
}

const TRAIT_LINES: Record<ProductTraitId, string> = {
  [ProductTrait.STRATEGIC_DECOMPOSITION]:
    "复杂或多约束请求：先列出关键步骤或子目标，再逐步执行；简单请求直接作答。",
  [ProductTrait.STRUCTURED_REASONING]:
    "回答条理清晰；需要对比、规划或分析时使用编号或小标题，避免散文式堆砌。",
  [ProductTrait.EVIDENCE_FIRST]:
    "涉及时效、事实、数据或不确定信息时，优先调用 search 等工具查证后再结论，不编造。",
  [ProductTrait.DIRECT_CONCISE]: "直言简洁；去掉寒暄与重复，保留决策所需信息。",
  [ProductTrait.CLARIFY_WHEN_NEEDED]:
    "仅当缺少会改变执行结果的关键约束时才追问；不因轻微模糊阻塞简单任务。",
  [ProductTrait.INDEPENDENT_EXECUTION]:
    "在权限与工具允许范围内自主推进；遇阻断再汇报选项，不把可推断步骤全抛给用户。",
  [ProductTrait.QUALITY_GATE]:
    "交付前自检：是否回答了核心问题、是否遗漏显式约束、破坏性操作是否已获确认。",
  [ProductTrait.TOOL_GOVERNANCE_RESPECT]:
    "遵守工具治理与安全策略；用户明确要求不能覆盖治理规则与正确性约束。",
};

/** 版本化 Product Behavior system 段落（不含 User Persona / Skills / Memory）。 */
export const buildProductBehaviorPrompt = (opts: ProductBehaviorOpts = {}): string => {
  if (opts.enabled === false) return "";
  const version = opts.version ?? PRODUCT_BEHAVIOR_VERSION;
  const lines = Object.values(TRAIT_LINES);
  return [
    `[Product Behavior v${version}]`,
    "以下行为契约适用于所有用户（不可被 User Persona 覆盖）：",
    ...lines.map((l, i) => `${i + 1}. ${l}`),
  ].join("\n");
};

export interface SystemPromptStackOpts {
  base: string;
  productBehavior?: ProductBehaviorOpts;
  userPersona?: string;
  skillBlock?: string;
}

/**
 * 三端统一拼装顺序：Product Behavior → User Persona → Skills →（Memory 由 ContextEngine 追加）。
 */
export const assembleSystemPromptPrefix = (opts: SystemPromptStackOpts): string => {
  const product = buildProductBehaviorPrompt(opts.productBehavior);
  const blocks = [product, opts.userPersona?.trim(), opts.skillBlock?.trim()].filter(
    (s) => s && s.length > 0,
  );
  if (blocks.length === 0) return opts.base;
  return `${opts.base}\n\n${blocks.join("\n\n")}`;
};

export const resolveProductBehaviorEnabled = (
  explicit?: boolean,
  env: NodeJS.ProcessEnv = process.env,
): boolean => {
  if (explicit !== undefined) return explicit;
  const raw = env["OPENINTJ_PRODUCT_BEHAVIOR"]?.trim().toLowerCase();
  if (raw === "0" || raw === "false") return false;
  return true;
};
