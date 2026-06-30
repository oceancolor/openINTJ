/**
 * 脱敏（redaction）—— RFC-003 方向三隐私保护。
 *
 * 钝化记忆会把用户的原始输入/输出落盘并参与挖掘。落盘前先脱敏，避免把
 * 邮箱、电话、银行卡、API key 等敏感串持久化进 dormant_events 或被挖成 pattern。
 *
 * 设计：纯函数、零依赖、保守替换（宁可多打码，不漏）。可被 `DormantRuntime` 在 record 路径调用。
 */

export type Redactor = (text: string) => string;

interface RedactionRule {
  name: string;
  pattern: RegExp;
  /** 替换占位符。 */
  placeholder: string;
}

// 注意：顺序敏感——先打更"具体"的（邮箱、卡号），再打更"宽"的（长数字串）。
const DEFAULT_RULES: RedactionRule[] = [
  {
    name: "email",
    pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,
    placeholder: "[REDACTED_EMAIL]",
  },
  {
    // 常见 API key / token 前缀（sk-, ghp_, AKIA…）后跟一段较长的字母数字。
    name: "apiKey",
    pattern: /\b(?:sk|ghp|gho|github_pat|AKIA|xoxb|xoxp)[-_][A-Za-z0-9_-]{12,}\b/g,
    placeholder: "[REDACTED_KEY]",
  },
  {
    // 中国大陆身份证（18 位，末位可 X）。必须排在卡号/电话之前，否则数字会被它们先吞掉。
    name: "idCard",
    pattern: /\b\d{17}[\dXx]\b/g,
    placeholder: "[REDACTED_ID]",
  },
  {
    // 13-19 位（可含空格/连字符分组）的银行卡号。
    name: "creditCard",
    pattern: /\b(?:\d[ -]?){13,19}\b/g,
    placeholder: "[REDACTED_CARD]",
  },
  {
    // 国际/国内电话：可选 +，7 段以上数字（含分隔符）。放在卡号之后避免误吞。
    name: "phone",
    pattern: /(?<!\d)\+?\d[\d ()-]{6,}\d(?!\d)/g,
    placeholder: "[REDACTED_PHONE]",
  },
];

export interface RedactorOpts {
  /** 关闭某些内置规则（按 name）。 */
  disable?: string[];
  /** 追加自定义规则。 */
  extraRules?: Array<{ name: string; pattern: RegExp; placeholder: string }>;
}

/**
 * 构造一个脱敏函数。多次调用同一函数安全（每次都基于规则正则重新匹配）。
 */
export const createRedactor = (opts: RedactorOpts = {}): Redactor => {
  const disabled = new Set(opts.disable ?? []);
  const rules: RedactionRule[] = [
    ...DEFAULT_RULES.filter((r) => !disabled.has(r.name)),
    ...(opts.extraRules ?? []),
  ];
  return (text: string): string => {
    if (!text) return text;
    let out = text;
    for (const r of rules) {
      // 每条规则用全局正则；reset lastIndex 由 replace 内部处理。
      out = out.replace(r.pattern, r.placeholder);
    }
    return out;
  };
};

/** 默认脱敏函数（全部内置规则）。 */
export const defaultRedactor: Redactor = createRedactor();
