/**
 * 内嵌 IDE 的二进制引导（把 VSCodium 的 reh-web 服务端拉到 vendor/）
 *
 * ── 为什么拉的是 reh-web 而不是桌面版 ──────────────────────────────
 * 桌面版（VSCodium-win32-x64-*.zip，229MB）是**给人在桌面双击用的**，
 * 我们要的是「把工作台通过 HTTP 提供给 Electron 内嵌」。后者对应的分发物
 * 就是 `vscodium-reh-web-*` —— 实测它解压后自带 node 运行时与
 * `out/server-main.js`，直接 `node out/server-main.js` 就能起 HTTP 服务，
 * 不需要桌面版那一整个 Electron 外壳。
 *
 * 体积也从 229MB 降到 94MB，而且**运行时不再联网**：
 * `codium-tunnel serve-web` 那条路会在首次 HTTP 请求时下载这同一个包
 * （实测 94,038,270 字节 / 99 秒，期间请求全返回 202 + "is downloading"）。
 * 我们自己拉一次，之后启动就是秒起、断网也能用、版本可复现。
 *
 * ── 为什么不打进安装包 ─────────────────────────────────────────────
 * 94MB 会让包体翻倍，而并非每个用户都要用内嵌 IDE。所以按需拉取：
 * 没拉时面板显示安装入口，拉过之后一直留在本地（vendor/ 不入库）。
 *
 * 用法：
 *   node scripts/vscodium/fetch.mjs            # 拉最新稳定版
 *   node scripts/vscodium/fetch.mjs --check    # 只检查装了没，不下载
 *   node scripts/vscodium/fetch.mjs --version 1.140.06880
 *
 * 为什么不用 npm 依赖：VSCodium 不发布 npm 包；官方分发就是 GitHub Releases。
 */

import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const args = process.argv.slice(2)
const wantCheck = args.includes('--check')
const versionArg = args.includes('--version') ? args[args.indexOf('--version') + 1] : null

const REPO = 'VSCodium/vscodium'
const API = `https://api.github.com/repos/${REPO}/releases/latest`

/** vendor 下的落点 —— 与 src/main/vscodium.ts 的 locateServer() 必须一致 */
const VENDOR_DIR = 'vscodium-server'

/* ------------------------------------------------------------------ *
 * 平台 → 资产名
 * ------------------------------------------------------------------ */

function assetFor(platform, arch, version) {
  const os = platform === 'win32' ? 'win32' : platform === 'darwin' ? 'darwin' : 'linux'
  const a = arch === 'arm64' ? 'arm64' : 'x64'
  // reh-web 三个平台都是 tar.gz，没有 zip 变体
  return { name: `vscodium-reh-web-${os}-${a}-${version}.tar.gz`, kind: 'tgz' }
}

/**
 * 解压后 node 运行时叫什么 —— 与 src/main/vscodium.ts 的 locateServer() 必须一致。
 * 三个平台的 reh-web 都自带一份 node，不依赖系统上的 node 版本。
 */
export function nodeRelative(platform) {
  return platform === 'win32' ? 'node.exe' : 'node'
}

/** 校验「这份包真的能用」的最低条件 */
function isComplete(base, platform) {
  return existsSync(join(base, nodeRelative(platform))) && existsSync(join(base, 'out', 'server-main.js'))
}

function run(cmd, cmdArgs, opts = {}) {
  return new Promise((res, rej) => {
    const child = spawn(cmd, cmdArgs, { stdio: 'ignore', ...opts })
    child.on('error', rej)
    child.on('exit', (code) => (code === 0 ? res(code) : rej(new Error(`${cmd} 退出码 ${code}`))))
  })
}

/**
 * 压缩包可能是 `./bin/...` 这种 flat 结构，也可能多一层根目录。
 * 两种都认 —— 只认一种的话，上游换个打包方式这里就莫名其妙地失败。
 */
function findRoot(base, platform) {
  if (isComplete(base, platform)) return base
  for (const e of readdirSync(base)) {
    const p = join(base, e)
    if (statSync(p).isDirectory() && isComplete(p, platform)) return p
  }
  return null
}

async function main() {
  const platform = process.platform
  const arch = process.arch === 'arm64' ? 'arm64' : 'x64'

  const vendor = join(root, 'vendor', VENDOR_DIR)
  const existing = findRoot(vendor, platform)
  if (wantCheck) {
    if (existing) {
      console.log(`[ide] 已安装：${join(existing, nodeRelative(platform))}`)
      process.exit(0)
    }
    console.log(`[ide] 未安装（vendor/${VENDOR_DIR} 下没有可运行的服务端）`)
    process.exit(1)
  }

  let version = versionArg
  if (!version) {
    console.log(`[ide] 查询最新版本…（${API}）`)
    const res = await fetch(API, { headers: { accept: 'application/vnd.github+json' } })
    if (!res.ok) throw new Error(`查版本失败 HTTP ${res.status}（可手动指定 --version）`)
    const json = await res.json()
    version = String(json.tag_name ?? '').trim()
    if (!version) throw new Error('接口没返回 tag_name')
  }
  console.log(`[ide] 目标版本 ${version}（${platform}-${arch}）`)

  const asset = assetFor(platform, arch, version)
  const url = `https://github.com/${REPO}/releases/download/${version}/${asset.name}`
  const cacheDir = join(root, '.cache')
  mkdirSync(cacheDir, { recursive: true })
  const tmp = join(cacheDir, `${asset.name}.${randomUUID().slice(0, 8)}.part`)

  console.log(`[ide] 下载 ${asset.name} …（约 94MB，取决于网络）`)
  const started = Date.now()
  const res = await fetch(url)
  if (!res.ok) throw new Error(`下载失败 HTTP ${res.status}：${url}`)
  const total = Number(res.headers.get('content-length') ?? 0)
  let got = 0
  let lastPrint = 0
  const chunks = []
  for await (const chunk of res.body) {
    chunks.push(chunk)
    got += chunk.length
    const now = Date.now()
    if (now - lastPrint > 2000) {
      lastPrint = now
      const pct = total ? ((got / total) * 100).toFixed(1) : '?'
      console.log(`[ide]   ${(got / 1048576).toFixed(1)}MB / ${(total / 1048576).toFixed(1)}MB（${pct}%）`)
    }
  }
  writeFileSync(tmp, Buffer.concat(chunks))
  console.log(`[ide] 下载完成 ${(got / 1048576).toFixed(1)}MB，用时 ${((Date.now() - started) / 1000).toFixed(1)}s`)

  // 只做「大小与 content-length 一致」的自检，够挡掉半截文件。
  // 完整 sha256 要多下一次 .sha256，且并不能挡住「上游换了包」的情况。
  if (total && statSync(tmp).size !== total) {
    rmSync(tmp, { force: true })
    throw new Error(`文件大小不一致（期望 ${total}，实际 ${statSync(tmp).size}），已删除半成品`)
  }

  console.log(`[ide] 解压到 vendor/${VENDOR_DIR} …`)
  rmSync(vendor, { recursive: true, force: true })
  mkdirSync(vendor, { recursive: true })
  // 实测 win32 的包是 `./bin/...` 这种 flat 结构，strip 掉前导 `.` 更稳。
  // 若上游哪天改成带根目录，findRoot 仍然认得出来 —— 所以这里错了也不会致命。
  await run('tar', ['-xzf', tmp, '-C', vendor, '--strip-components=1'])

  renameSync(tmp, join(cacheDir, asset.name))

  const final = findRoot(vendor, platform)
  if (!final) {
    throw new Error(
      `解压后没找到 ${nodeRelative(platform)} 与 out/server-main.js，压缩包结构可能变了`
    )
  }

  // 记录来源，排障时能回答「这版是哪来的」
  const productPath = join(final, 'product.json')
  let commit = null
  try {
    commit = JSON.parse(await (await import('node:fs/promises')).readFile(productPath, 'utf8')).commit ?? null
  } catch {
    /* product.json 读不出来不影响运行，只是少一条溯源信息 */
  }

  writeFileSync(
    join(vendor, 'SOURCE.json'),
    JSON.stringify(
      {
        kind: 'reh-web',
        version,
        commit,
        asset: asset.name,
        url,
        size: statSync(join(cacheDir, asset.name)).size,
        fetchedAt: Date.now()
      },
      null,
      2
    )
  )

  console.log(`[ide] 就绪：${join(final, nodeRelative(platform))}`)
  console.log(`[ide] 版本 ${version}${commit ? ` · commit ${commit}` : ''}`)
  console.log(`[ide] 提示：vendor/ 不入库（见 .gitignore），换机器要重新拉一次`)
}

main().catch((err) => {
  console.error(`[ide] 失败：${err.message}`)
  process.exit(1)
})
