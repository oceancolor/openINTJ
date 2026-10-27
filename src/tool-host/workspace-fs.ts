/**
 * 工作区文件读写（文件树 / 工件预览 / 生图落盘）
 *
 * ── 为什么住在 tool-host ────────────────────────────────────────
 * 碰文件系统的只能是 P-04：它同时是裁决权的持有者。放到主进程或渲染层，
 * 等于让一条不经裁决的路径拿到了工作区的读写能力 —— 那正是 exec 那条路
 * 花了大力气堵住的东西，不能在这里又开一个口子。
 *
 * ── 三条硬边界 ─────────────────────────────────────────────────
 *   ① 路径必须是**相对工作区根**的相对路径，且解析后不得越出根（不接受绝对路径）
 *   ② 读有上限，超限如实说超限，不截断冒充全文
 *   ③ 写一律先过裁决链（在 index.ts 里做，本文件只负责「裁决通过后怎么写」）
 *
 * ── 内容与脱敏 ─────────────────────────────────────────────────
 * 读出来的内容**不走脱敏管线**，理由与终端输出相同：用户自己点开自己工作区里的
 * 文件，遮掉它只会让人以为文件是空的。代价是这条路径的内容同样**不进事件库** ——
 * 一旦将来要落盘或喂模型，必须先补脱敏。
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, isAbsolute, relative, resolve } from 'node:path'
import { isIgnored } from './file-watcher'
import { isPreviewableFile, previewKindOf } from '../shared/session'
import type { FsEntry, FsListResult, FsReadResult } from '../shared/ipc'

/** 单目录最多列多少项 —— 防 node_modules 之类把界面冲垮（且它们本来就被忽略） */
const MAX_LIST_ENTRIES = 500

/** 读文件的总字节上限。图片按张算，4MB 足够 1280x720 的 PNG */
const MAX_READ_BYTES = 4 * 1024 * 1024

/** 文本类预览最多显示多少字符 —— 再多就是日志，不是给人看的 */
const MAX_TEXT_CHARS = 200_000

/**
 * 相对路径 → 工作区内的绝对路径；越界或非法返回 null。
 *
 * 拒绝绝对路径是刻意的：调用方（渲染层 / agent-host）都不该知道工作区在磁盘的哪里，
 * 它们只该说「工作区里的哪个文件」。
 */
export function resolveInside(root: string, rel: string): string | null {
  const cleaned = (rel ?? '').trim().replace(/\\/g, '/').replace(/^\.\//, '')
  if (!cleaned || cleaned === '.') return resolve(root)
  if (isAbsolute(cleaned)) return null

  const abs = resolve(root, cleaned)
  const relCheck = relative(resolve(root), abs)
  if (relCheck.startsWith('..') || isAbsolute(relCheck)) return null
  return abs
}

/** 绝对路径 → 相对工作区根（`/` 分隔）；越界返回 null */
export function toRelPath(root: string, abs: string): string | null {
  const rel = relative(resolve(root), resolve(abs))
  if (!rel) return ''
  if (rel.startsWith('..') || isAbsolute(rel)) return null
  return rel.split('\\').join('/')
}

export function listWorkspace(root: string, rel: string): FsListResult {
  const abs = resolveInside(root, rel)
  if (!abs) return { ok: false, path: rel, entries: [], truncated: false }

  let entries
  try {
    entries = readdirSync(abs, { withFileTypes: true })
  } catch (err) {
    return { ok: false, path: rel, entries: [], truncated: false }
  }

  // 目录在前、文件在后，各自按名字排 —— 文件树不按这个排的话每次打开顺序都不一样
  const out: FsEntry[] = []
  for (const e of entries) {
    if (e.isDirectory()) {
      if (isIgnored(e.name)) continue
      out.push({ name: e.name, path: joinRel(rel, e.name), kind: 'dir', size: null, previewable: false })
      continue
    }
    if (!e.isFile()) continue
    const childRel = joinRel(rel, e.name)
    if (isIgnored(childRel)) continue
    let size: number | null = null
    try {
      size = statSync(resolve(root, childRel)).size
    } catch {
      size = null
    }
    out.push({
      name: e.name,
      path: childRel,
      kind: 'file',
      size,
      previewable: isPreviewableFile(childRel)
    })
  }

  out.sort((a, b) => {
    if (a.kind !== b.kind) return a.kind === 'dir' ? -1 : 1
    return a.name.localeCompare(b.name, 'zh-CN')
  })

  return { ok: true, path: rel, entries: out.slice(0, MAX_LIST_ENTRIES), truncated: out.length > MAX_LIST_ENTRIES }
}

function joinRel(base: string, name: string): string {
  const b = base.replace(/\/+$/, '')
  return b ? `${b}/${name}` : name
}

export function readWorkspace(root: string, rel: string): FsReadResult {
  const abs = resolveInside(root, rel)
  if (!abs) return { ok: false, reason: '路径非法或越出工作区根' }
  if (!existsSync(abs)) return { ok: false, reason: '文件不存在' }

  let stat
  try {
    stat = statSync(abs)
  } catch (err) {
    return { ok: false, reason: `读取失败：${(err as Error).message}` }
  }
  if (!stat.isFile()) return { ok: false, reason: '不是一个文件' }
  if (stat.size > MAX_READ_BYTES) {
    return { ok: false, reason: `文件 ${stat.size} 字节，超过预览上限 ${MAX_READ_BYTES} 字节` }
  }

  const buf = readFileSync(abs)
  const kind = previewKindOf(rel)
  if (kind?.kind === 'image') {
    return {
      ok: true,
      path: rel,
      kind: 'image',
      dataUrl: `data:${kind.mimeType};base64,${buf.toString('base64')}`,
      text: '',
      size: buf.byteLength,
      truncated: false
    }
  }

  // 非白名单里的扩展名一律按纯文本试读：读出来是乱码也如实显示，
  // 比「这个文件打不开」更能帮人判断它是什么
  const text = buf.toString('utf8')
  const truncated = text.length > MAX_TEXT_CHARS
  return {
    ok: true,
    path: rel,
    kind: 'text',
    dataUrl: '',
    text: truncated ? text.slice(0, MAX_TEXT_CHARS) : text,
    size: buf.byteLength,
    truncated
  }
}

/** 目标是否已存在 —— 覆盖判定要在**裁决之前**做完，否则那条审计会记成「批准了但没写」 */
export function workspaceFileExists(root: string, rel: string): boolean {
  const abs = resolveInside(root, rel)
  return abs ? existsSync(abs) : false
}

export type WorkspaceWriteResult =
  | { ok: true; path: string; bytes: number }
  | { ok: false; reason: string }

/**
 * 落盘。裁决在调用方（index.ts）完成 —— 本函数被调到就意味着**已经获批**。
 *
 * 覆盖是显式的：不传 overwrite 而文件已存在时，由调用方先判掉。
 */
export function writeWorkspace(
  root: string,
  rel: string,
  bytes: Uint8Array,
  overwrite: boolean
): WorkspaceWriteResult {
  const abs = resolveInside(root, rel)
  if (!abs) return { ok: false, reason: '路径非法或越出工作区根' }
  if (existsSync(abs) && !overwrite) {
    return { ok: false, reason: '目标文件已存在（未声明覆盖）' }
  }

  try {
    mkdirSync(dirname(abs), { recursive: true })
    writeFileSync(abs, bytes)
    return { ok: true, path: toRelPath(root, abs) ?? rel, bytes: bytes.byteLength }
  } catch (err) {
    return { ok: false, reason: `写入失败：${(err as Error).message}` }
  }
}
