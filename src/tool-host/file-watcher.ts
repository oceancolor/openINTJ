/**
 * 工作区文件变化监听（「变更」分页签的数据源）
 *
 * ── 为什么住在 tool-host ────────────────────────────────────────
 * 监听要读文件系统，而「碰文件系统」的进程只能是 P-04 —— 那里同时是裁决权的
 * 持有者。放到主进程或渲染层，等于让一条不经裁决的路径拿到了工作区的观察能力。
 *
 * ── 只报不存 ─────────────────────────────────────────────────────
 * 这里唯一的产出是「哪个文件动了」，经主进程转交给 agent-host 登记。
 * 它**不落事件库、也不喂模型**：文件内容可能含密钥，而这条路径没有过脱敏管线
 * （与终端输出同一条边界，见 shared/terminal.ts 的 TERMINAL_OUTPUT_IS_NOT_REDACTED）。
 * 将来若要落盘或喂模型，必须先补脱敏 —— 路径本身不算秘密，内容算。
 *
 * ── 平台差异 ─────────────────────────────────────────────────────
 * `fs.watch(root, { recursive: true })` 只有 Windows 与 macOS 支持；
 * Linux 上会直接抛 ERR_FEATURE_UNAVAILABLE_ON_PLATFORM。
 * 所以递归与否按平台决定，Linux 退化为只监听根目录一层，并把这件事**报出去**，
 * 而不是静默少报 —— 少报的表现是「改了文件但变更列表没反应」，极难归因。
 */

import { existsSync, readdirSync, watch, type FSWatcher } from 'node:fs'
import { join, relative, sep } from 'node:path'
import { ulid } from '../shared/ulid'

export type FileChangeKind = 'added' | 'modified' | 'deleted'

export interface FileChangeNotice {
  changeId: string
  /** 相对工作区根的路径，分隔符统一为 `/` */
  path: string
  kind: FileChangeKind
  at: number
}

/**
 * 不监听的目录。
 *
 * .git 里的对象文件每次提交都在动，node_modules 一次安装就是几万条 ——
 * 不过滤的话，「变更」页会变成噪音，用户反而看不到自己改了什么。
 * `dist` / `out` 是构建产物，不是人的改动。
 */
const IGNORED_DIRS = new Set([
  '.git',
  'node_modules',
  'dist',
  'out',
  'vendor',
  '.cache',
  '.snapshots',
  'build',
  '.workbuddy'
])

/** 去抖窗口：连续保存会产生多次事件，合并成一条 */
const DEBOUNCE_MS = 300

/** 基线扫描上限 —— 超大仓库不为了建基线把进程拖死 */
const BASELINE_MAX_FILES = 20_000
const BASELINE_MAX_DEPTH = 12

const RECURSIVE_SUPPORTED = process.platform === 'win32' || process.platform === 'darwin'

/** 绝对路径 → 相对路径（`/` 分隔）；越界返回 null */
function toRel(root: string, abs: string): string | null {
  const rel = relative(root, abs)
  if (!rel || rel.startsWith('..') || rel.startsWith(sep)) return null
  return rel.split(sep).join('/')
}

export class WorkspaceWatcher {
  private watcher: FSWatcher | null = null
  /** 基线里已知存在的文件 —— 用来区分「新增」与「修改」 */
  private known = new Set<string>()
  private timers = new Map<string, NodeJS.Timeout>()

  constructor(
    private readonly root: string,
    private readonly onNotice: (n: FileChangeNotice) => void
  ) {}

  /**
   * 开始监听。可重复调用（幂等）—— 重新 init 时先停后起。
   *
   * 启动时先扫一遍建立基线：没有基线的话，第一次事件会把所有已存在的文件
   * 报成「新增」，而它们根本没动过。
   */
  start(): { ok: boolean; recursive: boolean; files: number; reason?: string } {
    this.stop()
    this.seedBaseline()

    try {
      this.watcher = watch(this.root, { recursive: RECURSIVE_SUPPORTED }, (_event, filename) => {
        // 不同平台给的类型不一样（string / Buffer / null），统一成字符串即可
        const name = filename == null ? '' : String(filename)
        if (!name) return
        const rel = toRel(this.root, join(this.root, name))
        if (!rel || isIgnored(rel)) return
        this.schedule(rel)
      })
    } catch (err) {
      return { ok: false, recursive: RECURSIVE_SUPPORTED, files: this.known.size, reason: (err as Error).message }
    }
    return { ok: true, recursive: RECURSIVE_SUPPORTED, files: this.known.size }
  }

  stop(): void {
    try {
      this.watcher?.close()
    } catch {
      /* 已经关了 */
    }
    this.watcher = null
    for (const t of this.timers.values()) clearTimeout(t)
    this.timers.clear()
  }

  get active(): boolean {
    return this.watcher !== null
  }

  /** 建基线：只记「有哪些文件」，不读内容（读内容既是浪费，也会碰密钥） */
  private seedBaseline(): void {
    this.known = new Set<string>()
    const stack: Array<{ dir: string; depth: number }> = [{ dir: this.root, depth: 0 }]
    while (stack.length > 0 && this.known.size < BASELINE_MAX_FILES) {
      const { dir, depth } = stack.pop()!
      let entries
      try {
        entries = readdirSync(dir, { withFileTypes: true })
      } catch {
        continue
      }
      for (const e of entries) {
        if (e.isDirectory()) {
          if (IGNORED_DIRS.has(e.name)) continue
          if (depth + 1 < BASELINE_MAX_DEPTH) stack.push({ dir: join(dir, e.name), depth: depth + 1 })
          continue
        }
        if (!e.isFile()) continue
        const rel = toRel(this.root, join(dir, e.name))
        if (!rel || isIgnored(rel)) continue
        this.known.add(rel)
        if (this.known.size >= BASELINE_MAX_FILES) break
      }
    }
  }

  private schedule(rel: string): void {
    const prev = this.timers.get(rel)
    if (prev) clearTimeout(prev)
    const timer = setTimeout(() => {
      this.timers.delete(rel)
      const exists = existsSync(join(this.root, rel))
      const kind: FileChangeKind = !exists ? 'deleted' : this.known.has(rel) ? 'modified' : 'added'
      if (exists) this.known.add(rel)
      else this.known.delete(rel)
      this.onNotice({ changeId: ulid(), path: rel, kind, at: Date.now() })
    }, DEBOUNCE_MS)
    this.timers.set(rel, timer)
  }
}

function isIgnored(rel: string): boolean {
  return rel.split('/').some((seg) => IGNORED_DIRS.has(seg))
}

/** 监听是否支持递归 —— 界面要如实显示降级状态，不能装作全覆盖 */
export function watcherRecursiveSupported(): boolean {
  return RECURSIVE_SUPPORTED
}
