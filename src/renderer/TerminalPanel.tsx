/**
 * 内嵌终端面板（US-15 / F29）
 *
 * 形态：**只读输出区（xterm.js）+ 底部命令输入行**。
 *
 * 为什么不是「裸终端」——即 xterm 直接收键盘、按键原样送 pty：
 *   US-15 的 AC-2 是硬约束「终端不是绕过策略层的通道」。要做到执行**前**裁决，
 *   就必须掌握「一条命令的边界」。键盘直通下回车直接进 pty，命令立刻开跑，
 *   再想拦已经晚了。所以把「提交」这一步拿到自己手里：输入行攒出整条命令 →
 *   送 tool-host 分级 → 放行才写进 pty。
 *
 * 代价（写在这里，免得下一个人以为是 bug）：
 *   - 不支持交互式全屏程序（vim / top / ssh 后的远端终端）；
 *   - 方向键在输入行里是「历史记录」，不是终端里的光标移动；
 *   - 交互式问答可以回答：把 `y` 当一条命令提交即可。
 *
 * 输出仍是真 PTY 输出：颜色、进度条、光标控制都对 —— 那是 pty 给的，
 * 换成 child_process 就没有 TTY 语义了。
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import type { ApprovalRequest } from '../shared/approval'
import { TERMINAL_RUN_PREFIX } from '../shared/terminal'
import '@xterm/xterm/css/xterm.css'

interface Props {
  /** 页面是否可见；隐藏时不销毁会话，只是不抢焦点 */
  active: boolean
}

/** 深空底配色 —— 与应用整体视觉一致，避免出现一块「别人的终端」 */
const THEME = {
  background: '#08030f',
  foreground: '#e6e1f5',
  cursor: '#7cf5c4',
  selectionBackground: '#3a2a63',
  black: '#1a1428',
  red: '#ff6b81',
  green: '#5ee6a8',
  yellow: '#ffd479',
  blue: '#7aa2ff',
  magenta: '#c891ff',
  cyan: '#6ff0e0',
  white: '#e6e1f5',
  brightBlack: '#6b6382'
}

export default function TerminalPanel({ active }: Props): React.ReactElement {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const termRef = useRef<Terminal | null>(null)
  const fitRef = useRef<FitAddon | null>(null)
  const sessionRef = useRef<string | null>(null)

  const [session, setSession] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [status, setStatus] = useState<string>('正在打开终端 …')
  /** 面板自己的提示条 —— 见下方 note() 的注释：不往 pty 区域里插字 */
  const [notice, setNotice] = useState<{ tone: 'warn' | 'danger' | 'ok'; text: string } | null>(null)
  const [history, setHistory] = useState<string[]>([])
  const [histIdx, setHistIdx] = useState<number>(-1)
  /** 重建计数：重开 = 让下面的 effect 整体重跑（销毁旧会话 + 新拉一个） */
  const [epoch, setEpoch] = useState(0)
  /**
   * 终端自己的待审批项。
   *
   * 为什么不在终端页显示主控台那张待审批卡：终端页打开时控制台是 CSS 隐藏的，
   * 用户在终端里敲了命令 → 卡在看不见的地方 → 他只看到「什么也没发生」，
   * 然后重敲一遍。等待必须发生在他正在看的那个面板里。
   */
  const [pending, setPending] = useState<ApprovalRequest | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)

  /**
   * 面板级提示 —— **不写进 pty 区域**。
   *
   * 这里踩过一个坑，值得写下来：最早是把「✗ 已拒绝」「⏳ 等待裁决」直接 write 进 xterm。
   * 但那一刻 shell 正停在提示符上，xterm 的光标就停在提示符所在行的**末尾**——
   * 我们写的字落在同一行，随后 PowerShell 重绘提示符（带 `\x1b[K` 清行尾），
   * 把我们的提示整段擦掉。现象是「代码明明写了、用户什么也没看到」。
   *
   * 结论：pty 区域归 shell 所有，任何「面板要说的话」都必须走自己的一条渲染通道。
   */
  const note = useCallback((text: string, tone: 'warn' | 'danger' | 'ok' = 'warn'): void => {
    setNotice({ tone, text })
  }, [])

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    const term = new Terminal({
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
      fontSize: 12.5,
      lineHeight: 1.25,
      theme: THEME,
      cursorBlink: false,
      // 关键：输出区不接受键盘输入。输入走下面的输入行 —— 这样「命令边界」才是可控的，
      // 而且不会出现「本地回显一次、pty 回显一次」的双重回显。
      disableStdin: true,
      convertEol: true,
      scrollback: 5000
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.open(host)
    fit.fit()
    termRef.current = term
    fitRef.current = fit

    let disposed = false

    const offData = window.openintj.onTerminalData((p) => {
      if (p.sessionId === sessionRef.current) term.write(p.data)
    })
    const offExit = window.openintj.onTerminalExit((p) => {
      if (p.sessionId !== sessionRef.current) return
      // 会话结束时 shell 已经没了，最后一行不会再有谁去重绘它 —— 这时写进 pty 区域是安全的
      term.write(`\r\n\x1b[38;5;245m[会话已结束 exit=${p.exitCode}]\x1b[0m\r\n`)
      sessionRef.current = null
      setSession(null)
      setStatus('会话已结束，点「重开」重新拉起 shell')
    })

    window.openintj
      .terminalOpen({ cols: term.cols, rows: term.rows })
      .then((info) => {
        if (disposed) return
        sessionRef.current = info.sessionId
        setSession(info.sessionId)
        setStatus(`${info.shell.split(/[\\/]/).pop() ?? info.shell} · ${info.cwd}`)
        inputRef.current?.focus()
      })
      .catch((err: Error) => {
        setStatus(`终端不可用：${err.message}`)
        note(`终端不可用：${err.message}`, 'danger')
      })

    // 尺寸变化必须同步给 pty，否则长输出折行会错位（AC-5）
    const ro = new ResizeObserver(() => {
      const sid = sessionRef.current
      if (!sid) return
      try {
        fit.fit()
      } catch {
        /* 面板隐藏时 fit 会算不出尺寸，忽略即可 */
      }
      void window.openintj.terminalResize(sid, term.cols, term.rows)
    })
    ro.observe(host)

    return () => {
      disposed = true
      ro.disconnect()
      offData()
      offExit()
      const sid = sessionRef.current
      sessionRef.current = null
      if (sid) void window.openintj.terminalClose(sid)
      term.dispose()
      termRef.current = null
      fitRef.current = null
    }
    // epoch 变化 = 用户点了「重开」，整块重建（旧会话在 cleanup 里被关掉）
  }, [note, epoch])

  const submit = useCallback(async () => {
    const sid = sessionRef.current
    const cmd = input.trim()
    if (!sid) {
      note('会话未就绪，请稍等或点「重开」', 'danger')
      return
    }
    if (!cmd) return
    if (busy) return

    setInput('')
    setHistIdx(-1)
    setHistory((h) => [...h, cmd])
    setBusy(true)
    try {
      const res = await window.openintj.terminalSubmit(sid, cmd)
      switch (res.kind) {
        case 'ran':
          // 不自己回显命令：pty 会把它回显出来（终端驱动的 echo）。
          // 这里再写一遍就会出现两行一样的命令。放行成功就把上一条提示收掉。
          setNotice(null)
          break
        case 'need-approval':
          // 主进程此刻已经弹了审批条；这里说明「为什么停住了」
          note(`等待裁决：${res.command}（风险 ${res.riskClass} / 规则 ${res.matchedRuleId}）`, 'warn')
          break
        case 'denied':
          note(`已拒绝：${res.reason}（风险 ${res.riskClass} / 规则 ${res.matchedRuleId}）`, 'danger')
          break
        case 'empty':
          break
        case 'no-session':
          note('会话不存在（可能已退出），点「重开」', 'danger')
          break
      }
    } catch (err) {
      note(`提交失败：${(err as Error).message}`, 'danger')
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }, [input, busy, note])

  const interrupt = useCallback(async () => {
    const sid = sessionRef.current
    if (!sid) return
    await window.openintj.terminalInterrupt(sid)
    note('已发送中断（Ctrl+C）', 'ok')
    inputRef.current?.focus()
  }, [note])

  const reopen = useCallback(() => {
    // 不 reload 整个窗口：那会把控制台的事件流、目标契约一起丢掉。
    // 只让本组件的 effect 重跑 —— 旧会话在 cleanup 里被关掉，新的重新拉起。
    setSession(null)
    setStatus('正在重新打开终端 …')
    setEpoch((e) => e + 1)
  }, [])

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault()
        void submit()
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        if (history.length === 0) return
        const next = histIdx < 0 ? history.length - 1 : Math.max(0, histIdx - 1)
        setHistIdx(next)
        setInput(history[next] ?? '')
        return
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        if (histIdx < 0) return
        const next = histIdx + 1
        if (next >= history.length) {
          setHistIdx(-1)
          setInput('')
        } else {
          setHistIdx(next)
          setInput(history[next] ?? '')
        }
        return
      }
      // Ctrl+C：不裁决，直接中断（中断只减少副作用，不产生新命令）
      if (e.key === 'c' && e.ctrlKey) {
        e.preventDefault()
        void interrupt()
      }
      if (e.key === 'l' && e.ctrlKey) {
        e.preventDefault()
        termRef.current?.clear()
      }
    },
    [submit, history, histIdx, interrupt]
  )

  useEffect(() => {
    if (active) setTimeout(() => inputRef.current?.focus(), 0)
  }, [active])

  /**
   * 订阅审批请求，只挑出终端自己的。
   *
   * 过滤条件是 `runId` 前缀而不是 agentId：agentId 是给泳道分组用的字符串，
   * 万一将来有人把它改了，这里会静默失效 —— 而 runId 前缀是终端的身份。
   */
  useEffect(() => {
    const isMine = (r: ApprovalRequest): boolean => r.runId.startsWith(TERMINAL_RUN_PREFIX)
    const offReq = window.openintj.onApprovalRequest((r) => {
      if (isMine(r)) setPending(r)
    })
    const offCancel = window.openintj.onApprovalCancel((p) => {
      setPending((cur) => (cur && p.requestIds.includes(cur.requestId) ? null : cur))
    })
    return () => {
      offReq()
      offCancel()
    }
  }, [])

  const decide = useCallback(
    async (approved: boolean, remember: boolean) => {
      const req = pending
      if (!req) return
      setPending(null)
      await window.openintj.respondApproval(req.requestId, approved, remember)
      inputRef.current?.focus()
    },
    [pending]
  )

  return (
    <div className="term-wrap" data-testid="terminal-panel">
      <div className="term-status">
        <span className="mono term-dot" data-state={session ? 'on' : 'off'} />
        <span className="mono term-status-text">{status}</span>
        <span className="term-status-actions">
          <button className="pill" onClick={() => void interrupt()} disabled={!session}>
            中断
          </button>
          <button className="pill" onClick={reopen}>
            重开
          </button>
        </span>
      </div>

      <div className="term-out" ref={hostRef} />

      {notice && (
        <div className={`term-notice ${notice.tone}`} data-testid="term-notice">
          <span>{notice.text}</span>
          <button className="term-notice-x" onClick={() => setNotice(null)} aria-label="关闭提示">
            ×
          </button>
        </div>
      )}

      {pending && (
        <div className="term-approval" data-testid="term-approval">
          <div className="term-approval-head">
            <span className="risk-badge warn">{pending.riskClass}</span>
            <code className="term-approval-cmd">{pending.command}</code>
          </div>
          <div className="term-approval-actions">
            <span className="term-approval-note">这条命令需要你点头才执行（终端不是旁路）</span>
            <button className="pill ghost" onClick={() => void decide(false, false)}>
              拒绝
            </button>
            <button className="pill" onClick={() => void decide(true, false)}>
              允许一次
            </button>
            <button className="pill primary" onClick={() => void decide(true, true)}>
              本会话内允许
            </button>
          </div>
        </div>
      )}

      <div className="term-input-row">
        <span className="mono term-prompt">$</span>
        <input
          ref={inputRef}
          className="mono term-input"
          value={input}
          placeholder={session ? '输入命令后回车 —— 每条命令都会先经裁决' : '终端未就绪'}
          disabled={!session || busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          spellCheck={false}
          autoComplete="off"
        />
        <button className="pill" onClick={() => void submit()} disabled={!session || busy || !input.trim()}>
          {busy ? '执行中' : '执行'}
        </button>
      </div>

      <div className="term-hint">
        命令模式：每条命令执行前都要过关（只读自动放行，其余按档位裁决）。不支持 vim / top / ssh 这类全屏交互程序 ——
        它们的「命令边界」不是换行，硬拦会把终端弄坏；需要时请在代码面板里开。
      </div>
    </div>
  )
}
