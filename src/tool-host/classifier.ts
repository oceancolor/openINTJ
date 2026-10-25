/**
 * 命令风险分级器
 *
 * 依据 IC-08（按风险分级）与《系统设计》§3.2.M10：
 *   L0 只读 / L1 工作区内写入 / L2 工作区外写入 / L3 危险命令
 *   另有 R5-network（出网）与 R0-unknown（无法归类，按最严处理）
 *
 * 判定口径约束：一条命令同时命中多类时，取**最高**风险等级（工作区外的 rm -rf 判 L3）。
 *
 * 注意：这里是**静态启发式**分级，不是安全边界。真正的边界是「未获批准不产生进程」。
 * 分级表设计为可配置（`risk-rules`），此为首版内置规则集。
 */

import { resolve, isAbsolute, relative } from 'node:path'
import type { RiskClass } from '../shared/approval'

export interface ClassifyInput {
  command: string
  cwd: string
  /** 工作区根目录 —— 越界写入的判定边界 */
  workspaceRoot: string
}

export interface ClassifyResult {
  riskClass: RiskClass
  matchedRuleId: string
  reason: string
}

/** 内置危险命令规则（R4-01 ~ R4-08）。命中即 L3，不可降级 */
const DANGEROUS_RULES: Array<{
  id: string
  test: (cmd: string) => boolean
  reason: string
}> = [
  {
    id: 'R4-01',
    reason: '递归删除',
    test: (c) => /\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+/i.test(c) || /\brmdir\s+\/s/i.test(c) || /\bdel\s+\/[sS]/i.test(c)
  },
  {
    id: 'R4-02',
    reason: '格式化 / 分区写入',
    test: (c) => /\b(mkfs(\.[a-z0-9]+)?|format\s+[a-z]:|diskpart|fdisk|dd\s+if=|dd\s+of=\/dev\/)/i.test(c)
  },
  {
    id: 'R4-03',
    reason: '提权',
    test: (c) => /(^|\s)(sudo|doas|runas|pkexec)(\s|$)/i.test(c) || /\bStart-Process\b.*-Verb\s+RunAs/i.test(c)
  },
  {
    id: 'R4-04',
    reason: '监听端口并暴露到非回环地址',
    test: (c) =>
      /(--host[=\s]+0\.0\.0\.0|-H\s+0\.0\.0\.0|--bind[=\s]+0\.0\.0\.0)/i.test(c) ||
      /\b(nc|ncat|netcat)\b.*-l/i.test(c)
  },
  {
    id: 'R4-05',
    reason: '访问凭据目录',
    test: (c) =>
      /(~|\$HOME|\/Users\/[^/\s]+|\/home\/[^/\s]+|C:\\Users\\[^\\\s]+)[\\/](\.ssh|\.aws|\.tencentcloud|\.gnupg|\.kube|\.docker)([\\/\s]|$)/i.test(c) ||
      /(id_rsa|id_ed25519|\.pem|\.p12|credentials\.json)/i.test(c)
  },
  {
    id: 'R4-06',
    reason: '远程脚本直接执行',
    test: (c) => /\b(curl|wget)\b[^|]*\|\s*(ba|z|da)?sh\b/i.test(c)
  },
  {
    id: 'R4-07',
    reason: '权限放宽',
    test: (c) => /\bchmod\s+(-R\s+)?(777|a\+rwx)/i.test(c) || /\bicacls\b.*\/grant\b.*Everyone/i.test(c)
  },
  {
    id: 'R4-08',
    reason: 'Git 危险操作（丢弃本地改动）',
    test: (c) => /\bgit\s+(reset\s+--hard|clean\s+-[a-z]*f|push\s+--force|push\s+-f)\b/i.test(c)
  }
]

/** 只读命令白名单（L0）。仅列常见且无副作用者 */
const READONLY_RULES: Array<{ id: string; test: (cmd: string) => boolean }> = [
  { id: 'R1-01', test: (c) => /^\s*(ls|dir|pwd|cd|cat|head|tail|less|more|wc|file|stat|tree)\b/i.test(c) },
  { id: 'R1-02', test: (c) => /^\s*(grep|rg|ag|find|fd|which|where|type)\b/i.test(c) },
  {
    id: 'R1-03',
    test: (c) => /^\s*git\s+(status|log|diff|show|branch|remote|rev-parse|ls-files|blame)\b/i.test(c)
  },
  { id: 'R1-04', test: (c) => /^\s*(node|npm|pnpm|yarn|python|python3|go|cargo|rustc|tsc)\s+(-v|--version)\b/i.test(c) },
  { id: 'R1-05', test: (c) => /^\s*(echo|printf|whoami|date|uname|hostname|env)\b/i.test(c) }
]

/** 出网类（L5） */
const NETWORK_RULES: Array<{ id: string; test: (c: string) => boolean }> = [
  { id: 'R5-01', test: (c) => /\b(curl|wget|http|https|ssh|scp|rsync|ftp|telnet)\b/i.test(c) },
  { id: 'R5-02', test: (c) => /\b(npm|pnpm|yarn|pip|cargo|go)\s+(install|add|get|publish)\b/i.test(c) }
]

/**
 * 从命令中粗略提取可能被写入的路径（启发式）。
 * 无法可靠解析 shell 语法 —— 因此 L1 的判定采取「保守」策略：
 * 只要命令里出现了工作区之外的绝对路径或上跳路径，就判 L2。
 */
function extractPaths(command: string): string[] {
  const out: string[] = []
  const re = /(?:^|[\s'"])((?:[A-Za-z]:[\\/]|\/|~\/|\.\.?[\\/])[^\s'";|&<>]*)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(command)) !== null) {
    if (m[1]) out.push(m[1])
  }
  return out
}

/**
 * 工作区内写入迹象。
 *
 * 注意 git 的写操作必须列进来：`git init` / `git add` 等虽不含 rm/cp 这类字眼，
 * 但确实在工作区里落文件。漏掉会让它们落到「无法归类 → 每次都要审批」，
 * 不仅吵，还会把审批疲劳变成绕过审批的动机。
 */
const WRITE_HINT =
  /\b(mkdir|rmdir|touch|cp|mv|rm|tee|sed\s+-i|writeFile)\b|>\s*\S|\bgit\s+(init|add|commit|checkout|switch|restore|stash|apply|merge|rebase|tag|worktree|reset|clean|mv|rm)\b/i

export function classify(input: ClassifyInput): ClassifyResult {
  const { command, cwd, workspaceRoot } = input
  const trimmed = command.trim()

  if (!trimmed) {
    return { riskClass: 'R0-unknown', matchedRuleId: 'R0-01', reason: '空命令' }
  }

  // 1) 危险命令优先 —— 命中即 L3，不可被后续规则降级
  for (const rule of DANGEROUS_RULES) {
    if (rule.test(trimmed)) {
      return { riskClass: 'L3', matchedRuleId: rule.id, reason: rule.reason }
    }
  }

  // 2) 出网
  for (const rule of NETWORK_RULES) {
    if (rule.test(trimmed)) {
      return { riskClass: 'R5-network', matchedRuleId: rule.id, reason: '出网访问' }
    }
  }

  // 3) 只读
  for (const rule of READONLY_RULES) {
    if (rule.test(trimmed)) {
      return { riskClass: 'L0', matchedRuleId: rule.id, reason: '只读命令' }
    }
  }

  // 4) 写入判定：看路径是否越出工作区根
  const root = resolve(workspaceRoot)
  const paths = extractPaths(trimmed)
  for (const p of paths) {
    const abs = isAbsolute(p) ? resolve(p) : resolve(cwd, p)
    const rel = relative(root, abs)
    const outside = rel.startsWith('..') || isAbsolute(rel)
    if (outside) {
      return {
        riskClass: 'L2',
        matchedRuleId: 'R3-01',
        reason: `写入路径越出工作区根：${abs}`
      }
    }
  }

  // 5) 有写入迹象但都在工作区内
  if (WRITE_HINT.test(trimmed)) {
    return { riskClass: 'L1', matchedRuleId: 'R2-01', reason: '工作区内写入' }
  }

  // 6) 无法归类 —— 按最严处理，不猜测
  return {
    riskClass: 'R0-unknown',
    matchedRuleId: 'R0-02',
    reason: '无法归类的命令，按最严等级裁决'
  }
}
