/**
 * MCP 接入契约（D6）
 *
 * 配置格式对齐主流的 `mcpServers`（command / args / env），
 * 这样用户从别处抄一份配置就能用，不用为本产品单独学一套。
 *
 * 两个本产品特有的字段：
 *   - `trust`：MCP server 的能力**无法静态判断**（它可能读文件、发 HTTP、连数据库），
 *     所以默认按最谨慎处理（`confirm` = 每次调用都要审批）。
 *     只有在用户明确知道那个 server 只读时，才该改成 `readonly` 自动放行。
 *   - `enabled`：允许留着配置但不启动，省得为了关一个 server 去删配置。
 *
 * 工具命名：`mcp__<server>__<tool>`。
 * 用双下划线而不是冒号，是因为 OpenAI 的 function name 只允许 `[a-zA-Z0-9_-]` ——
 * 冒号会被网关拒掉，而工具名要在两套协议里通用。
 */

export type McpTrust = 'readonly' | 'confirm'

export interface McpServerConfig {
  command: string
  args?: string[]
  env?: Record<string, string>
  /** 默认 confirm：MCP 能力不可预知，宁可多点一次「允许」 */
  trust?: McpTrust
  enabled?: boolean
  /** 连接超时，默认 20s。慢启动的 server（npx 现装包）可以调大 */
  connectTimeoutMs?: number
}

export interface McpConfigFile {
  mcpServers?: Record<string, McpServerConfig>
}

export type McpConnectionStatus = 'connecting' | 'connected' | 'failed' | 'disabled'

export interface McpServerStatus {
  name: string
  status: McpConnectionStatus
  toolCount: number
  trust: McpTrust
  /** 失败原因 —— 界面必须显示，否则用户只会看到「工具少了一个」 */
  error?: string
}

export interface McpToolInfo {
  server: string
  /** server 里的原始工具名 */
  name: string
  /** Agent 侧看到的全名：mcp__<server>__<tool> */
  qualifiedName: string
  description: string
  inputSchema: Record<string, unknown>
  trust: McpTrust
}

export const MCP_TOOL_PREFIX = 'mcp__'
const SEP = '__'

export function qualifyMcpTool(server: string, tool: string): string {
  return `${MCP_TOOL_PREFIX}${server}${SEP}${tool}`
}

export function isMcpToolName(name: string): boolean {
  return name.startsWith(MCP_TOOL_PREFIX)
}

/**
 * 拆解 `mcp__server__tool`。
 *
 * server 名里不允许出现 `__`（加载配置时会被拒），所以从第一个 `__` 处切分是无歧义的。
 * 工具名本身可以含下划线 —— 从右侧切会出错，这里从左侧切。
 */
export function parseMcpToolName(
  qualified: string
): { server: string; tool: string } | null {
  if (!isMcpToolName(qualified)) return null
  const rest = qualified.slice(MCP_TOOL_PREFIX.length)
  const at = rest.indexOf(SEP)
  if (at <= 0 || at + SEP.length >= rest.length) return null
  return { server: rest.slice(0, at), tool: rest.slice(at + SEP.length) }
}

/** server 名的合法性：不含 `__`，非空，只允许常见安全字符 */
export function isValidServerName(name: string): boolean {
  return /^[A-Za-z0-9_.-]+$/.test(name) && !name.includes(SEP) && name.length <= 48
}

/**
 * MCP 调用在「命令」这一栏里的表示：`mcp:server:tool {参数摘要}`。
 *
 * 为什么要有这么个字符串：审批票据是**绑定命令原文哈希**的，审批卡片也要显示
 * 「到底要干什么」。MCP 调用没有 shell 命令，但它在裁决与审计链里必须占同一个位置，
 * 否则就得为它抄一套票据逻辑 —— 那就是两条安全链路，迟早只有一条是对的。
 *
 * agent-host（发起方，负责弹卡片、写事件）和 tool-host（裁决方，负责验票）
 * **必须算出逐字节相同的字符串**，所以这个函数放在 shared 里，只有一份实现。
 */
export function mcpCommandString(server: string, tool: string, args: unknown): string {
  let brief = ''
  try {
    brief = JSON.stringify(args ?? {})
  } catch {
    brief = '(参数无法序列化)'
  }
  if (brief.length > 480) brief = `${brief.slice(0, 480)}…`
  return `mcp:${server}:${tool} ${brief}`
}
