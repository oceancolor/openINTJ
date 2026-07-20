/**
 * RFC-006：版本化 Product Behavior 契约（全用户一致，与 Dormant User Persona 分层）。
 * 不声称 MBTI 类型；只定义可观察工程行为。
 */

export const PRODUCT_BEHAVIOR_VERSION = "1.1.0";

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
    "回答条理清晰；对比、规划或分析的最终答案必须使用编号、列表或小标题，并明确给出结论。",
  [ProductTrait.EVIDENCE_FIRST]:
    "涉及时效、事实、数据或不确定信息时，优先调用 search 等工具查证后再结论，不编造。",
  [ProductTrait.DIRECT_CONCISE]: "直言简洁；去掉寒暄与重复，保留决策所需信息。",
  [ProductTrait.CLARIFY_WHEN_NEEDED]:
    "仅当缺少会改变执行结果的关键约束时才追问；不因轻微模糊阻塞简单任务。",
  [ProductTrait.INDEPENDENT_EXECUTION]:
    "在权限与工具允许范围内自主推进；排序、转换、计算等确定性小任务必须实际完成，不能复述输入或只给做法。",
  [ProductTrait.QUALITY_GATE]:
    "交付前逐项核对用户的显式要求；计算要写明结果，比较/判断要明确确认是否满足，不能留下未完成步骤。",
  [ProductTrait.TOOL_GOVERNANCE_RESPECT]:
    "遵守工具治理与安全策略；越权或破坏性请求必须明确拒绝，不能声称已执行，用户要求不能覆盖治理规则。",
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
    "输出纪律：交付给用户的最终答案只包含答案本身，不得混入 Thought、Action、Action-Input 或 FINAL 等内部协议标记。",
    "在输出最终答案前静默检查：是否完成了每一项要求、顺序/计算是否正确、结论是否明确；发现遗漏先修正再交付。",
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

export interface ProductBehaviorAnswerResult {
  answer: string;
  revised: boolean;
  guards: string[];
}

export interface EnforceProductBehaviorAnswerOpts {
  query: string;
  draft: string;
  revise?: (instruction: string) => Promise<string>;
}

const hasUnsafeDestructiveIntent = (query: string): boolean =>
  /(?:删除|清空|格式化|rm\s+-rf|del(?:ete)?\b)/i.test(query) &&
  /(?:工作区外|系统文件|安全策略|绕过|权限|outside\s+(?:the\s+)?workspace|system\s+files)/i.test(
    query,
  );

const arithmeticAnswer = (query: string): string | undefined => {
  const expression = query.match(/(-?\d+(?:\.\d+)?)\s*([+\-*/×])\s*(-?\d+(?:\.\d+)?)/);
  const comparison = query.match(/(?:满足|确认|是否)[\s\S]*?([<>]=?|={1,2})\s*(-?\d+(?:\.\d+)?)/);
  if (!expression || !comparison) return undefined;
  const left = Number(expression[1]);
  const right = Number(expression[3]);
  const operator = expression[2]!;
  const result =
    operator === "+"
      ? left + right
      : operator === "-"
        ? left - right
        : operator === "*" || operator === "×"
          ? left * right
          : right !== 0
            ? left / right
            : undefined;
  if (result === undefined || !Number.isFinite(result)) return undefined;
  const threshold = Number(comparison[2]);
  const comparator = comparison[1]!;
  const satisfies =
    comparator === ">"
      ? result > threshold
      : comparator === ">="
        ? result >= threshold
        : comparator === "<"
          ? result < threshold
          : comparator === "<="
            ? result <= threshold
            : result === threshold;
  const normalizedExpression = `${expression[1]}${operator}${expression[3]}`;
  return `${normalizedExpression}=${result}，且 ${result}${comparator}${threshold}，所以${
    satisfies ? "满足" : "不满足"
  } ${comparator}${threshold}。`;
};

const deterministicAnswer = (query: string): { answer: string; guard: string } | undefined => {
  if (hasUnsafeDestructiveIntent(query)) {
    return {
      answer:
        "不能绕过安全与权限治理，也不能删除工作区外的系统文件；如需处理工作区内文件，请提供明确路径并确认范围。",
      guard: "unsafe-destructive-request",
    };
  }

  const uppercase = query.match(/(?:把|将)\s*["“”']?([A-Za-z]+)["“”']?\s*转(?:成|为)?大写/i);
  if (uppercase) {
    return { answer: uppercase[1]!.toUpperCase(), guard: "uppercase-transform" };
  }

  const lowercase = query.match(/(?:把|将)\s*["“”']?([A-Za-z]+)["“”']?\s*转(?:成|为)?小写/i);
  if (lowercase) {
    return { answer: lowercase[1]!.toLowerCase(), guard: "lowercase-transform" };
  }

  if (/(?:按字母|alphabetical)/i.test(query)) {
    const list = query
      .split(/[：:]/)
      .at(-1)
      ?.match(/[A-Za-z]+(?:\s*,\s*[A-Za-z]+){1,}/)?.[0];
    if (list) {
      const sorted = list
        .split(",")
        .map((item) => item.trim())
        .sort((a, b) => a.localeCompare(b, "en"));
      return { answer: sorted.join(", "), guard: "alphabetical-sort" };
    }
  }

  const arithmetic = arithmeticAnswer(query);
  if (arithmetic) return { answer: arithmetic, guard: "arithmetic-constraint" };

  if (/^\s*(?:请)?部署到生产[。.!]?\s*$/.test(query)) {
    return {
      answer: "请确认目标环境、集群或主机、域名以及发布窗口；这些约束会直接改变生产部署步骤。",
      guard: "material-clarification",
    };
  }
  return undefined;
};

export const resolveDeterministicProductBehaviorAnswer = (
  query: string,
): { answer: string; guard: string } | undefined => deterministicAnswer(query);

const firstSentence = (answer: string): string | undefined => {
  const match = answer.trim().match(/^([\s\S]*?[。！？!?])/);
  return match?.[1]?.trim();
};

const repairIssues = (query: string, answer: string): string[] => {
  const issues: string[] = [];
  const comparison = /(?:比较|对比|\bvs\.?\b|versus)/i.test(query);
  const structured = /(?:^|\n)(?:#{1,3}\s|\d+[.、)]\s*|[-*]\s+)/m.test(answer);
  if (comparison && !structured) issues.push("对比任务缺少列表、小标题或编号结构");
  if (/优缺点/.test(query) && (!/优点/.test(answer) || !/缺点/.test(answer))) {
    issues.push("没有分别覆盖优点和缺点");
  }
  if (/选择建议/.test(query) && !/(?:选择|建议)/.test(answer)) {
    issues.push("没有给出明确选择建议");
  }
  if (
    /(?:规划|计划|方案)/.test(query) &&
    /阶段/.test(query) &&
    (!structured || (/交付物/.test(query) && !/交付物/.test(answer)))
  ) {
    issues.push("分阶段计划缺少清晰结构或逐阶段交付物");
  }
  if (/(?:^|\n)\s*(?:Thought|Action|Action-Input|FINAL)\s*:/i.test(answer)) {
    issues.push("最终答案泄漏内部协议标记");
  }
  return issues;
};

const buildRepairInstruction = (query: string, answer: string, issues: string[]): string => {
  const chineseNumber: Record<string, number> = {
    一: 1,
    二: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    七: 7,
    八: 8,
    九: 9,
    十: 10,
  };
  const stageMatch = query.match(/([一二三四五六七八九十]|\d+)\s*阶段/);
  const stageCount = stageMatch
    ? (chineseNumber[stageMatch[1]!] ?? Number(stageMatch[1]))
    : undefined;
  if (
    stageCount !== undefined &&
    Number.isInteger(stageCount) &&
    stageCount > 0 &&
    stageCount <= 10
  ) {
    return [
      "直接给出可执行计划，不要复述请求，不要追问，不要输出内部协议。",
      `必须恰好包含 ${stageCount} 个编号阶段，每个阶段都写“目标”“关键工作”“交付物”。`,
      ...Array.from(
        { length: stageCount },
        (_, index) =>
          `${index + 1}. 阶段 ${index + 1}\n   - 目标：填写具体目标\n   - 关键工作：填写具体工作\n   - 交付物：填写可验收产物`,
      ),
      "把“填写……”替换成针对用户任务的具体内容，只输出完成后的计划。",
    ].join("\n");
  }
  const subjects = query.match(
    /(?:比较|对比)\s*([^，,。]+?)\s*(?:和|与|及|\bvs\.?\b|versus)\s*([^，,。]+?)(?:的|，|,|。)/i,
  );
  if (subjects) {
    const left = subjects[1]!.trim();
    const right = subjects[2]!.trim();
    return [
      "直接回答，不要复述用户请求、草稿或模板说明。",
      "只输出填充了具体内容的 Markdown，并严格使用以下结构：",
      `## ${left}`,
      "- 优点：写出具体优点",
      "- 缺点：写出具体缺点",
      `## ${right}`,
      "- 优点：写出具体优点",
      "- 缺点：写出具体缺点",
      "## 选择建议",
      "- 写出明确的适用条件和选择结论",
    ].join("\n");
  }
  return [
    "请修订下面的草稿。只输出面向用户的最终答案，不要输出 Thought、Action、Action-Input 或 FINAL。",
    `用户请求：${query}`,
    `必须修复：${issues.join("；")}`,
    `草稿：${answer}`,
  ].join("\n");
};

/**
 * RFC-006 is an executable contract, not prompt-only branding. Deterministic,
 * unambiguous operations are completed locally; open-ended draft defects get
 * one bounded model revision.
 */
export const enforceProductBehaviorAnswer = async (
  opts: EnforceProductBehaviorAnswerOpts,
): Promise<ProductBehaviorAnswerResult> => {
  const deterministic = deterministicAnswer(opts.query);
  if (deterministic) {
    return { answer: deterministic.answer, revised: false, guards: [deterministic.guard] };
  }

  let answer = opts.draft.trim();
  const guards: string[] = [];
  if (/(?:一句话|one sentence)/i.test(opts.query)) {
    const sentence = firstSentence(answer);
    if (sentence && sentence.length >= 12) {
      answer = sentence;
      guards.push("single-sentence");
    }
  }

  const issues = repairIssues(opts.query, answer);
  if (issues.length === 0 || !opts.revise) {
    return { answer, revised: false, guards };
  }

  const revised = (await opts.revise(buildRepairInstruction(opts.query, answer, issues))).trim();
  const revisedDeterministic = deterministicAnswer(opts.query);
  return {
    answer: revisedDeterministic?.answer ?? revised,
    revised: true,
    guards: [...guards, ...issues.map((issue) => `revision:${issue}`)],
  };
};
