/**
 * MCP 连接管理（P-04 tool-host 内）
 *
 * 为什么 MCP 客户端放 tool-host 而不是 agent-host：
 *   ① MCP server 是子进程，而 tool-host 的职责就是管子进程
 *   ② 工具调用的**裁决权**在 tool-host（D10），MCP 调用同样是「执行」，
 *      必须走同一条裁决与审计链 —— 放到 agent-host 就等于给模型开了一条旁路
 *
 * 失败策略与主流程相反：**连接失败不 fail-stop**。
 *   一个 MCP server 起不来，不该让整个 Agent 用不了 —— 少了几个工具而已。
 *   但失败原因必须如实上报到界面：用户只会看到「工具少了一个」，
 *   不告诉他原因，他会以为是产品坏了。
 */

import { readFileSync } from 'node:fs'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import {
  isValidServerName,
  qualifyMcpTool,
  type McpConfigFile,
  type McpServerConfig,
  type McpServerStatus,
  type McpToolInfo,
  type McpTrust
} from '../shared/mcp'

const DEFAULT_CONNECT_TIMEOUT_MS = 20_000
/** 单次工具调用上限。MCP 工具可能很慢（装包、拉网页），但不该无限等 */
const CALL_TIMEOUT_MS = 60_000

interface Entry {
  name: string
  config: McpServerConfig
  trust: McpTrust
  status: McpServerStatus['status']
  error?: string
  client?: Client
  tools: McpToolInfo[]
}

export interface McpCallResult {
  ok: boolean
  /** 结果的文本表示（给模型看的内容，已尽量压平成文本） */
  text: string
  isError: boolean
}

/** 把 MCP 的 content 数组压成文本。图片等非文本块只标注类型，不塞 base64 */
function flattenContent(content: unknown): string {
  if (!Array.isArray(content)) return String(content ?? '')
  const parts: string[] = []
  for (const block of content) {
    if (!block || typeof block !== 'object') {
      parts.push(String(block))
      continue
    }
    const b = block as Record<string, unknown>
    if (b['type'] === 'text' && typeof b['text'] === 'string') {
      parts.push(b['text'])
    } else if (b['type'] === 'image') {
      parts.push(`[图片 ${String(b['mimeType'] ?? 'unknown')}]`)
    } else if (b['type'] === 'resource' && typeof b['resource'] === 'object') {
      const r = b['resource'] as Record<string, unknown>
      parts.push(typeof r['text'] === 'string' ? r['text'] : `[资源 ${String(r['uri'] ?? '')}]`)
    } else {
      parts.push(`[${String(b['type'] ?? 'unknown')}]`)
    }
  }
  return parts.join('\n')
}

export class McpHub {
  private readonly entries = new Map<string, Entry>()
  /** 配置文件的读取错误 —— 与「没有配置」必须分开，前者要告诉用户 */
  private configError: string | null = null

  constructor(private readonly configPath: string) {}

  /** 读配置并连接所有启用的 server。可重复调用（会先关掉旧的） */
  async reload(): Promise<void> {
    await this.closeAll()
    this.configError = null

    let raw: McpConfigFile
    try {
      raw = JSON.parse(readFileSync(this.configPath, 'utf8')) as McpConfigFile
    } catch (err) {
      const e = err as NodeJS.ErrnoException
      if (e.code === 'ENOENT') return // 没配过，正常状态
      this.configError = `mcp.json 读取失败：${e.message}`
      console.warn(`[mcp] ${this.configError}`)
      return
    }

    const servers = raw.mcpServers ?? {}
    for (const [name, cfg] of Object.entries(servers)) {
      if (!isValidServerName(name)) {
        console.warn(`[mcp] 跳过非法 server 名「${name}」：不允许含 "__" 或特殊字符`)
        continue
      }
      if (!cfg || typeof cfg.command !== 'string' || !cfg.command.trim()) {
        console.warn(`[mcp] 跳过 server「${name}」：缺少 command`)
        continue
      }
      const trust: McpTrust = cfg.trust === 'readonly' ? 'readonly' : 'confirm'
      const entry: Entry = {
        name,
        config: cfg,
        trust,
        status: cfg.enabled === false ? 'disabled' : 'connecting',
        tools: []
      }
      this.entries.set(name, entry)
      if (entry.status !== 'disabled') await this.connect(entry)
    }
  }

  private async connect(entry: Entry): Promise<void> {
    const timeoutMs = entry.config.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS
    const transport = new StdioClientTransport({
      command: entry.config.command,
      args: entry.config.args ?? [],
      // 不传 env 时 SDK 用一份安全默认集（含 PATH）；传了就要自己保证 PATH 在
      ...(entry.config.env ? { env: { ...entry.config.env } } : {}),
      // stderr 直接继承：MCP server 的日志是排障的唯一线索，不能吞
      stderr: 'inherit'
    })

    const client = new Client({ name: 'openintj', version: '0.1.0' }, { capabilities: {} })

    let timer: NodeJS.Timeout | undefined
    try {
      await Promise.race([
        client.connect(transport),
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error(`连接超时 ${timeoutMs}ms`)), timeoutMs)
        })
      ])
      clearTimeout(timer)

      const listed = await client.listTools()
      entry.tools = (listed.tools ?? []).map((t) => ({
        server: entry.name,
        name: t.name,
        qualifiedName: qualifyMcpTool(entry.name, t.name),
        description: t.description ?? '',
        inputSchema: (t.inputSchema as Record<string, unknown>) ?? { type: 'object', properties: {} },
        trust: entry.trust
      }))
      entry.client = client
      entry.status = 'connected'
      entry.error = undefined
      console.log(`[mcp] ${entry.name} 已连接，${entry.tools.length} 个工具`)
    } catch (err) {
      clearTimeout(timer)
      entry.status = 'failed'
      entry.error = (err as Error).message
      console.warn(`[mcp] ${entry.name} 连接失败：${entry.error}`)
      // 连接失败要把半开的子进程收掉，否则会留一堆孤儿 node 进程
      try {
        await client.close()
      } catch {
        /* 已经关了就算了 */
      }
    }
  }

  listTools(): McpToolInfo[] {
    return [...this.entries.values()]
      .filter((e) => e.status === 'connected')
      .flatMap((e) => e.tools)
  }

  /** 找一条工具记录，用于裁决前确认「这个工具确实存在且属于这个 server」 */
  findTool(server: string, tool: string): McpToolInfo | null {
    const entry = this.entries.get(server)
    if (!entry || entry.status !== 'connected') return null
    return entry.tools.find((t) => t.name === tool) ?? null
  }

  status(): McpServerStatus[] {
    return [...this.entries.values()].map((e) => ({
      name: e.name,
      status: e.status,
      toolCount: e.tools.length,
      trust: e.trust,
      error: e.error
    }))
  }

  /** 配置本身读不出来时的错误（与「没配置」区分） */
  getConfigError(): string | null {
    return this.configError
  }

  async callTool(server: string, tool: string, args: unknown): Promise<McpCallResult> {
    const entry = this.entries.get(server)
    if (!entry) return { ok: false, text: `未知 MCP server：${server}`, isError: true }
    if (entry.status !== 'connected' || !entry.client) {
      return { ok: false, text: `MCP server「${server}」未连接${entry.error ? `：${entry.error}` : ''}`, isError: true }
    }
    if (!entry.tools.some((t) => t.name === tool)) {
      return { ok: false, text: `server「${server}」没有名为「${tool}」的工具`, isError: true }
    }

    let timer: NodeJS.Timeout | undefined
    try {
      const result = await Promise.race([
        entry.client.callTool({
          name: tool,
          arguments: (args ?? {}) as Record<string, unknown>
        }),
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error(`工具调用超时 ${CALL_TIMEOUT_MS}ms`)), CALL_TIMEOUT_MS)
        })
      ])
      clearTimeout(timer)

      const r = result as { content?: unknown; isError?: boolean }
      return {
        ok: true,
        text: flattenContent(r.content),
        isError: Boolean(r.isError)
      }
    } catch (err) {
      clearTimeout(timer)
      return { ok: false, text: `调用失败：${(err as Error).message}`, isError: true }
    }
  }

  async closeAll(): Promise<void> {
    for (const entry of this.entries.values()) {
      if (!entry.client) continue
      try {
        await entry.client.close()
      } catch {
        /* 关不掉就算了，别让它挡住退出 */
      }
      entry.client = undefined
    }
    this.entries.clear()
  }
}
