/**
 * 会话存储（进程内）
 *
 * ── 为什么放在 agent-host 而不是渲染层 ────────────────────────────
 * 渲染层是会被重建的（切页、刷新、崩溃重启），而它又不该持有业务状态
 * （《系统设计》§3.1：渲染进程只负责呈现）。消息放渲染层的结果是：
 * 界面一刷新，刚才聊过什么全没了 —— 但 run 还在跑，事件流里也还留着。
 * 那种「执行在继续、对话已清空」的错位比什么都难解释。
 *
 * 所以消息跟 run 放在一起，都由 P-03 持有。渲染层只是取来显示。
 *
 * ── 为什么不落 SQLite ───────────────────────────────────────────
 * 与 run 状态同样处理：首版内存态，进程重启即丢。
 * 理由也相同 —— 会话是「这一轮对话」的载体，不是要长期归档的事实；
 * 真正需要长期留存的那部分（事件流、审批审计）已经在库里了。
 * 这里若先落库，就得顺带定义会话的表结构、清理策略与迁移，
 * 而这项工作的价值要等「跨重启续聊」真正被需要时才成立。
 */

import { ulid } from '../shared/ulid'
import type {
  AgentSession,
  ChatMessage,
  ChatRole,
  ChatKind,
  FileChange,
  PreviewTarget,
  SessionState
} from '../shared/session'
import { previewKindOf } from '../shared/session'

interface SessionRecord {
  session: AgentSession
  messages: ChatMessage[]
}

const sessions = new Map<string, SessionRecord>()

/** 文件变化：runId → 变化列表（倒序，新的在前） */
const changesByRun = new Map<string, FileChange[]>()

/** 预览目标：runId → 目标列表（倒序） */
const previewsByRun = new Map<string, PreviewTarget[]>()

/** 单 run 最多留多少条变化 —— 防 `npm install` 之类把界面冲垮 */
const MAX_CHANGES_PER_RUN = 200
/** 单 run 最多留多少个预览目标 */
const MAX_PREVIEWS_PER_RUN = 20

function clipTitle(text: string): string {
  const one = text.trim().replace(/\s+/g, ' ')
  return one.length <= 24 ? one : `${one.slice(0, 24)}…`
}

export function listSessions(): AgentSession[] {
  return [...sessions.values()]
    .map((r) => r.session)
    .sort((a, b) => a.createdAt - b.createdAt)
}

export function createSession(title?: string): AgentSession {
  const now = Date.now()
  const session: AgentSession = {
    sessionId: ulid(now),
    title: title?.trim() ? clipTitle(title) : '新会话',
    createdAt: now,
    updatedAt: now,
    runId: null,
    state: 'empty'
  }
  sessions.set(session.sessionId, { session, messages: [] })
  return session
}

export function getSession(sessionId: string): AgentSession | undefined {
  return sessions.get(sessionId)?.session
}

/** 会话绑定的 run 列表（新建会话为空） */
export function runsOf(sessionId: string): string[] {
  const r = sessions.get(sessionId)
  if (!r) return []
  return r.session.runId ? [r.session.runId] : []
}

export function bindRun(sessionId: string, runId: string): void {
  const r = sessions.get(sessionId)
  if (!r) return
  r.session.runId = runId
  r.session.updatedAt = Date.now()
}

export function setState(sessionId: string, state: SessionState): void {
  const r = sessions.get(sessionId)
  if (!r) return
  r.session.state = state
  r.session.updatedAt = Date.now()
}

export function messagesOf(sessionId: string): ChatMessage[] {
  return sessions.get(sessionId)?.messages ?? []
}

/**
 * 追加一条消息。
 *
 * 顺序刻意是「先落再跑」：执行中途崩了，用户说过的话还在。
 * 反过来（先跑后落）的话，崩掉的那一轮会连提问一起消失 ——
 * 而提问恰恰是事后唯一能解释「它当时想干什么」的东西。
 */
export function appendMessage(input: {
  sessionId: string
  role: ChatRole
  kind: ChatKind
  text: string
  atSeq?: number | null
  payload?: unknown
}): ChatMessage {
  const message: ChatMessage = {
    messageId: ulid(),
    sessionId: input.sessionId,
    role: input.role,
    kind: input.kind,
    text: input.text,
    at: Date.now(),
    atSeq: input.atSeq ?? null,
    payload: input.payload ?? null
  }
  const rec = sessions.get(input.sessionId)
  if (rec) {
    rec.messages.push(message)
    rec.session.updatedAt = message.at
  }
  return message
}

/** 从一段输出里认出回环地址（dev server 的提示行） */
const LOOPBACK_URL = /https?:\/\/(?:127\.0\.0\.1|localhost|\[::1\]):\d{2,5}(?:\/[^\s"'<>]*)?/gi

/**
 * 摘出这段文本里的回环地址。
 *
 * 只认回环：预览服务的意义是「看本机刚起的服务」，外部地址既不该被自动打开，
 * 也不该让人误以为它属于这次任务。尾部的中英文标点要剔掉 ——
 * 那是句子的一部分，不是 URL 的一部分。
 */
export function detectLoopbackUrls(text: string): string[] {
  const out: string[] = []
  for (const m of text.matchAll(LOOPBACK_URL)) {
    const url = m[0].replace(/[.,;:)\]}"']+$/, '')
    if (url && !out.includes(url)) out.push(url)
  }
  return out
}

/**
 * 记一条文件变化。runId 由主进程补上（tool-host 不知道当前在跑哪个 run）。
 */
export function noteChange(
  change: Omit<FileChange, 'at' | 'changeId' | 'confirmed'>
): FileChange {
  const full: FileChange = { changeId: ulid(), confirmed: false, ...change, at: Date.now() }
  const list = changesByRun.get(full.runId) ?? []
  // 同一路径的变化只保留最新一条：列表是「现在哪些文件动过」，不是审计流水
  const next = [full, ...list.filter((c) => c.path !== full.path)].slice(0, MAX_CHANGES_PER_RUN)
  changesByRun.set(full.runId, next)
  return full
}

export function changesOf(runId: string): FileChange[] {
  return changesByRun.get(runId) ?? []
}

/**
 * 登记一个预览目标。
 *
 * `url` 类只认回环地址：预览指向的是本机起的服务，放行外部地址等于给「外部内容」
 * 开了一扇不用过裁决链的窗 —— 那和终端绕过策略层是同一类问题。
 *
 * 工件类（image / text）认的是**工作区内的路径**，不要求它是 URL：
 * 它本来就不是地址，是这次任务做出来的一个文件。
 */
export function notePreview(target: Omit<PreviewTarget, 'at'>): PreviewTarget | null {
  if (target.kind === 'url' && !isLoopbackUrl(target.url)) return null
  const full: PreviewTarget = { ...target, at: Date.now() }
  const list = previewsByRun.get(full.runId) ?? []
  const key = full.kind === 'url' ? `url:${full.url}` : `path:${full.path ?? ''}`
  // 同一地址/同一文件只留一条（dev server 的提示会重复输出很多次）
  if (list.some((p) => (p.kind === 'url' ? `url:${p.url}` : `path:${p.path ?? ''}`) === key)) {
    return full
  }
  previewsByRun.set(full.runId, [full, ...list].slice(0, MAX_PREVIEWS_PER_RUN))
  return full
}

export function previewsOf(runId: string): PreviewTarget[] {
  return previewsByRun.get(runId) ?? []
}

/**
 * 登记一个**工件**为预览目标。
 *
 * 与 `notePreview` 分开成两个入口，是因为两者的准入规则根本不同：
 * 一个是「这个地址是不是本机的」，另一个是「这个文件是不是可预览的」。
 * 合成一个函数就只能靠 `kind` 分叉，而这种分叉迟早会漏掉一边。
 *
 * 扩展名不可预览时返回 null —— 不登记，也不报错：
 * 生成了一个 .bin 不是错误，只是没什么可预览的。
 */
export function noteArtifact(input: {
  runId: string
  path: string
  title: string
  mimeType?: string
  size?: number
}): PreviewTarget | null {
  const kind = previewKindOf(input.path)
  if (!kind) return null
  return notePreview({
    previewId: ulid(),
    runId: input.runId,
    kind: kind.kind,
    url: '',
    title: input.title,
    path: input.path,
    mimeType: input.mimeType ?? kind.mimeType,
    size: input.size
  })
}

/** 预览地址白名单：只放行回环地址 */
export function isLoopbackUrl(raw: string): boolean {
  try {
    const u = new URL(raw)
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false
    const host = u.hostname.toLowerCase()
    return host === '127.0.0.1' || host === 'localhost' || host === '[::1]' || host === '::1'
  } catch {
    return false
  }
}

/** 用户确认过某条变化 —— 只打标记，不删记录（删了就不知道自己确认过什么） */
export function confirmChange(changeId: string): boolean {
  for (const [runId, list] of changesByRun) {
    const hit = list.find((c) => c.changeId === changeId)
    if (hit) {
      hit.confirmed = true
      changesByRun.set(runId, [...list])
      return true
    }
  }
  return false
}
