/**
 * 推送前密钥体检 —— 针对「目标是公开仓库」这件事。
 *
 * 为什么单独写一个而不是交给 CI：这里要回答的是「现在这一刻工作区里有没有
 * 会被 git add 带上去的密钥」。等到 CI 报警时，密钥已经在公开历史里了 ——
 * 公开仓库的密钥一旦进历史，轮换也没用，得重写历史。
 *
 * 用法：node scripts/git/pre-push-scan.mjs
 * 退出码：0 = 干净；1 = 发现疑似密钥
 */

import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const ROOT = process.cwd()

/** 与 .gitignore 对齐的排除目录 —— 这些目录体积巨大且从不入库 */
const SKIP_DIRS = new Set([
  'node_modules',
  'out',
  'dist',
  'release',
  'vendor',
  '.cache',
  '.vite',
  '.git',
  '.snapshots'
])
const SKIP_FILES = new Set(['package-lock.json', '.npmrc'])

/** 二进制/非文本扩展名：直接跳过内容扫描（但仍然计入清单） */
const BINARY = /\.(png|jpe?g|gif|ico|zip|gz|tgz|7z|exe|dll|pak|dat|bin|node|wasm|pdf|ttf|woff2?)$/i

/**
 * 规则集。每条带一个 label，便于报告里说清「为什么它可疑」。
 * 顺序即优先级：命中即报，不重复 PAPER OVER。
 */
const RULES = [
  { label: 'OpenAI/兼容 key', re: /\bsk-[A-Za-z0-9_-]{20,}\b/g },
  { label: 'Anthropic key', re: /\bsk-ant-[A-Za-z0-9_-]{20,}\b/g },
  { label: 'GitHub token', re: /\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b/g },
  { label: 'Slack token', re: /\bxox[abpsr]-[A-Za-z0-9-]{10,}\b/g },
  { label: 'AWS access key', re: /\bAKIA[0-9A-Z]{16}\b/g },
  { label: '私钥', re: /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/g },
  { label: 'JWT', re: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g },
  { label: '硬编码 Bearer', re: /Bearer\s+[A-Za-z0-9._-]{24,}/g }
]

/** 这些是「看起来像密钥但其实是占位/示例」的白名单 —— 命中就放行 */
const EXAMPLES = [
  /placeholder/i,
  /example/i,
  /your[_-]?key/i,
  /\$\{[^}]+\}/, // `${OPENAI_API_KEY}` 这种模板引用
  /<[^>]+>/,
  /xxxx+/i,
  /^xxx/i
]

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    const rel = relative(ROOT, p).replace(/\\/g, '/')
    if (statSync(p).isDirectory()) {
      if (SKIP_DIRS.has(name)) continue
      walk(p, out)
      continue
    }
    if (SKIP_FILES.has(name)) continue
    out.push({ abs: p, rel })
  }
  return out
}

/**
 * 带理由的豁免标记：`// scan:ignore(理由)`。
 *
 * 为什么不用「按文件/目录黑名单」：黑名单太容易变成「这一片都不用看了」，
 * 真密钥混进去也就没人管。改成**逐行豁免且必须写理由**，扫描结果里还会把
 * 豁免清单打出来 —— 谁豁免了什么、为什么，永远是可审计的。
 *
 * 唯一已知正当用法：脱敏探针里的合成夹具（src/main/index.ts 的 verifyRedaction）。
 * 那些字符串天生长得像密钥，否则验不了「真密钥会被遮」。
 */
const IGNORE = /\/\/\s*scan:ignore\(([^)]{4,})\)/

const files = walk(ROOT)
const findings = []
const ignored = []
let scanned = 0

for (const f of files) {
  if (BINARY.test(f.rel)) continue
  let text
  try {
    text = readFileSync(f.abs, 'utf8')
  } catch {
    continue
  }
  scanned++
  const lines = text.split(/\r?\n/)
  lines.forEach((line, i) => {
    // 跳过注释行：文档里讲「长什么样」不等于真有密钥
    const trimmed = line.trim()
    if (trimmed.startsWith('*') || trimmed.startsWith('//') || trimmed.startsWith('#')) return
    const excuse = IGNORE.exec(line)?.[1]
    for (const r of RULES) {
      const hits = line.match(r.re)
      if (!hits) continue
      for (const h of hits) {
        if (EXAMPLES.some((ex) => ex.test(h))) continue
        if (excuse) ignored.push({ file: f.rel, line: i + 1, label: r.label, reason: excuse })
        else findings.push({ file: f.rel, line: i + 1, label: r.label, snippet: h.slice(0, 12) + '…' })
      }
    }
  })
}

console.log(`扫描 ${scanned} 个文本文件（已排除 node_modules / vendor / out / .cache / .snapshots）`)
if (ignored.length) {
  console.log(`\nℹ️  按标记豁免 ${ignored.length} 处（应当只有脱敏探针的合成夹具）：`)
  for (const g of ignored) console.log(`  ${g.file}:${g.line}  [${g.label}]  理由：${g.reason}`)
}
if (findings.length === 0) {
  console.log('\n✅ 未发现疑似密钥，可以推送')
  process.exit(0)
}
console.log(`\n❌ 发现 ${findings.length} 处疑似密钥：\n`)
for (const f of findings) {
  console.log(`  ${f.file}:${f.line}  [${f.label}]  ${f.snippet}`)
}
console.log('\n推送前请轮换这些凭据。注意：公开仓库里密钥一旦进历史，必须重写历史才行。')
process.exit(1)
