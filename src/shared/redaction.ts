/**
 * 事件脱敏管线（D8）
 *
 * 一条底线、三条纪律。
 *
 * **底线：密钥脱敏无条件生效。**
 * 不管调用方把 `redaction` 标成什么（包括 `'none'`），密钥模式一律替换。
 * 理由：密钥泄露是单向不可逆的伤害，它不该取决于「调用方有没有记得标对级别」。
 * 如果脱敏要靠每个调用点自觉，那它迟早会有漏网的地方 —— 而那种漏洞不会报错，只会静静漏。
 *
 * 三条纪律：
 *   ① 只处理字符串**值**，不碰键名 —— 否则 payload 的结构会被破坏，回放就没法用。
 *   ② 路径脱敏只把家目录前缀换成 `~`，不抹掉整条路径 —— 排障时还需要看是哪个文件。
 *   ③ 宁可漏一个自造密钥，也不遮掉所有 commit hash。
 *      曾想用「32 位以上十六进制串」当密钥规则，但 git commit hash 正好是 40 位 hex，
 *      一遮之后排障完全没法做。**误伤有用的信息，比漏掉一个边缘情况更糟。**
 */

import { homedir } from 'node:os'
import type { Redaction } from './events'

/** 一条密钥规则。`to` 用函数是为了保留可辨识的尾字符（如 key 末 4 位） */
interface SecretRule {
  id: string
  re: RegExp
  to: string | ((match: string) => string)
}

/**
 * 密钥模式表。
 *
 * 全部写成 `g` 正则并只用于 `String.replace` —— 不用 `.test()`，
 * 免得 lastIndex 状态在多次调用之间互相污染。
 */
const SECRET_RULES: SecretRule[] = [
  // 主流厂商的 sk- 前缀（OpenAI / Anthropic / 各类兼容网关）。留尾 4 位便于「认出是哪把」
  { id: 'sk-key', re: /\bsk-[A-Za-z0-9_-]{16,}\b/g, to: (m) => `sk-***${m.slice(-4)}` },
  // GitHub
  { id: 'gh-token', re: /\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, to: 'gh*_***' },
  // AWS Access Key ID
  { id: 'aws-akid', re: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g, to: 'AKIA***' },
  // Slack
  { id: 'slack-token', re: /\bxox[abprs]-[A-Za-z0-9-]{10,}\b/g, to: 'xox*-***' },
  // JWT（三段 base64url）
  {
    id: 'jwt',
    re: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g,
    to: '[JWT]'
  },
  // 私钥块。限长度上限，避免一个未闭合的 BEGIN 把后面几千字全吃掉
  {
    id: 'private-key',
    re: /-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----[\s\S]{0,4000}?-----END [A-Z ]{0,30}PRIVATE KEY-----/g,
    to: '[PRIVATE KEY]'
  },
  // 连接串里的密码：scheme://user:pass@host
  { id: 'url-credential', re: /\b([a-z][a-z0-9+.-]*:\/\/[^:/?#\s]+:)[^@/\s]+@/gi, to: '$1***@' },
  // Authorization: Bearer xxx
  { id: 'bearer', re: /\b(Bearer\s+)[A-Za-z0-9._\-+/=]{16,}/gi, to: '$1***' },
  // key=value 形式的凭据。键名留着 —— 「这里有 password」本身就是有用信息
  {
    id: 'kv-secret',
    re: /\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)(\s*[:=]\s*)(["']?)([^\s"',;}]{6,})/gi,
    to: '$1$2$3***'
  }
]

/** 终端控制序列。清掉它能让事件流在界面上正常显示，也避免转义字符注入 */
const ANSI_ESCAPE = /\u001b\[[0-9;?]*[A-Za-z]|\u001b\][^\u0007]*\u0007/g

export interface RedactionHit {
  ruleId: string
  count: number
}

export interface RedactionOutcome {
  value: unknown
  /**
   * 实际生效的级别。
   *
   * 注意它可能**高于**请求的级别：请求 `'none'` 但命中了密钥规则时，这里会返回 `'secret'`。
   * 事件上的 `redaction` 字段要写这个值 —— 它表示「这条事件真实被处理过」，
   * 而不是「调用方声称该怎么处理」。
   */
  level: Redaction
  hits: RedactionHit[]
}

const LEVEL_ORDER: Redaction[] = ['none', 'secret', 'path', 'content']

function maxLevel(a: Redaction, b: Redaction): Redaction {
  return LEVEL_ORDER.indexOf(a) >= LEVEL_ORDER.indexOf(b) ? a : b
}

/** 家目录的几种写法 —— Windows 上同一个位置可能有三种转义形式 */
function homeVariants(): string[] {
  let home: string
  try {
    home = homedir()
  } catch {
    return []
  }
  if (!home || home.length < 4) return []
  return [...new Set([home, home.replace(/\\/g, '/'), home.replace(/\//g, '\\')])]
}

/**
 * 递归处理字符串值。
 *
 * 深度上限 12 是为了防环 —— payload 来自 JSON，正常不会有环，
 * 但这里被调用在「落盘前」这个关键路径上，一旦抛错整条事件就丢了。
 */
function walk(value: unknown, fn: (s: string) => string, depth = 0): unknown {
  if (depth > 12) return value
  if (typeof value === 'string') return fn(value)
  if (Array.isArray(value)) return value.map((v) => walk(v, fn, depth + 1))
  if (value && typeof value === 'object') {
    const src = value as Record<string, unknown>
    const out: Record<string, unknown> = {}
    for (const key of Object.keys(src)) out[key] = walk(src[key], fn, depth + 1)
    return out
  }
  return value
}

/**
 * 对事件 payload 做脱敏。
 *
 * @param requested 调用方声明的级别。它只表示「希望额外做多少」，最低限度（密钥）恒生效。
 * @param homeOverride 测试用；生产走 `os.homedir()`
 */
export function applyRedaction(
  payload: unknown,
  requested: Redaction = 'none',
  homeOverride?: string
): RedactionOutcome {
  const hits = new Map<string, number>()

  let level = requested

  const secretPass = (s: string): string => {
    let out = s
    for (const rule of SECRET_RULES) {
      out = out.replace(rule.re, (m: string) => {
        hits.set(rule.id, (hits.get(rule.id) ?? 0) + 1)
        return typeof rule.to === 'function' ? rule.to(m) : rule.to
      })
    }
    return out
  }

  let fn = secretPass

  if (requested === 'path' || requested === 'content') {
    const homes = homeOverride ? [homeOverride] : homeVariants()
    fn = (s: string): string => {
      let out = secretPass(s)
      for (const h of homes) out = out.split(h).join('~')
      return out
    }
  }

  if (requested === 'content') {
    const withPath = fn
    fn = (s: string): string => withPath(s).replace(ANSI_ESCAPE, '')
  }

  const value = walk(payload, fn)

  // 密钥规则是底线：即使调用方标了 none，命中也要如实升级级别
  if (hits.size > 0) level = maxLevel(level, 'secret')

  return {
    value,
    level,
    hits: [...hits].map(([ruleId, count]) => ({ ruleId, count }))
  }
}

/** 只脱敏单个字符串（审计表用；比整包脱敏便宜，语义也更直接） */
export function redactText(text: string, level: Redaction = 'secret'): string {
  const r = applyRedaction(text, level)
  return typeof r.value === 'string' ? r.value : text
}
