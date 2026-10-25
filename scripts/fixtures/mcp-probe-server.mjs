/**
 * 探针用 MCP server（stdio，行分隔 JSON-RPC）
 *
 * 为什么自己写一个而不是装个现成的：
 *   ① 现成的 server（filesystem / github）要装包、要网络、要凭据，冒烟测试不能依赖这些
 *   ② 断言需要**确定的输出**：`echo` 返回什么必须是我们说了算的
 *   ③ 它要能被 `ELECTRON_RUN_AS_NODE=1` 跑起来 —— 探针环境里只有 Electron 自带的 node
 *
 * 只实现握手 + 列工具 + 调工具三条路径，够验证「连接、发现、裁决、调用」全链路。
 */

import { createInterface } from 'node:readline'

const SERVER_INFO = { name: 'openintj-mcp-probe', version: '0.1.0' }
const PROTOCOL_VERSION = '2024-11-05'

const TOOLS = [
  {
    name: 'echo',
    description: '原样回显传入的 text。用于验证链路连通，不产生任何副作用。',
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string', description: '要回显的内容' } },
      required: ['text'],
      additionalProperties: false
    }
  },
  {
    name: 'boom',
    description: '总是以 isError 返回。用于验证错误结果能如实回灌给模型。',
    inputSchema: { type: 'object', properties: {}, additionalProperties: true }
  }
]

function callTool(name, args) {
  if (name === 'echo') {
    return { content: [{ type: 'text', text: `pong:${String(args?.text ?? '')}` }] }
  }
  if (name === 'boom') {
    return { content: [{ type: 'text', text: 'boom: 这是预期内的失败结果' }], isError: true }
  }
  throw Object.assign(new Error(`未知工具：${name}`), { code: -32602 })
}

function handle(msg) {
  switch (msg.method) {
    case 'initialize':
      return {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: SERVER_INFO
      }
    case 'ping':
      return {}
    case 'tools/list':
      return { tools: TOOLS }
    case 'tools/call':
      return callTool(msg.params?.name, msg.params?.arguments)
    default:
      throw Object.assign(new Error(`未实现的方法：${msg.method}`), { code: -32601 })
  }
}

const rl = createInterface({ input: process.stdin, terminal: false })

rl.on('line', (line) => {
  const text = line.trim()
  if (!text) return

  let msg
  try {
    msg = JSON.parse(text)
  } catch {
    return // 解析不了就丢，MCP 侧没有「回复一个坏 JSON」的合法方式
  }

  // 通知没有 id，不需要回包
  if (msg.id === undefined || msg.id === null) return

  try {
    const result = handle(msg)
    process.stdout.write(`${JSON.stringify({ jsonrpc: '2.0', id: msg.id, result })}\n`)
  } catch (err) {
    process.stdout.write(
      `${JSON.stringify({
        jsonrpc: '2.0',
        id: msg.id,
        error: { code: err.code ?? -32603, message: String(err.message ?? err) }
      })}\n`
    )
  }
})

// 父进程关掉 stdin 就退出 —— 不然会留下一堆孤儿 node 进程
rl.on('close', () => process.exit(0))
