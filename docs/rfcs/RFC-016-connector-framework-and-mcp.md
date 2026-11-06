# RFC-016：连接器框架 + MCP + 只读连接器

| 字段 | 值 |
|---|---|
| 状态 | **Accepted** |
| 起草日期 | 2026-11-05 |
| 决策日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 依赖 | RFC-013 审批/授权、RFC-015 Scheduler（后续定时拉取） |
| 后继 | 可写连接器（P2）、OAuth 凭据、云端日历/邮件 API |

---

## 摘要

Agent 现在只能看工作区文件、跑白名单命令、搜网页——看不见用户的日历、
邮件、笔记，就谈不上"个人"助手。本 RFC 加连接器框架：连接器把外部数据源
变成 ToolHub 工具，统一走治理门禁（策略/配额/RFC-013 风险分级）。

v1 只做**只读**连接器，且只接本地数据源——不碰 OAuth、不存第三方凭据：

- **笔记**：本地目录（Markdown/text）列举/读取/搜索；
- **日历**：本地 `.ics` 文件解析（VEVENT + 基础 RRULE 展开）→ 近期日程/搜索；
- **邮件**：本地 mbox 归档解析 → 列表/读取/搜索；
- **MCP**：stdio JSON-RPC 客户端（initialize / tools/list / tools/call），
  把任意 MCP server 的工具命名空间化注册进 ToolHub。

云端 API（Google/Microsoft/IMAP）等凭据与 OAuth 方案单独立项；框架不得
预设只有文件数据源。

## 1. 框架

```typescript
interface ConnectorTool { descriptor: ToolDescriptor; handler: ToolHandler }
interface Connector {
  id: string;
  displayName: string;
  init?(): Promise<void>;            // MCP 等异步发现；文件类无需
  tools(): readonly ConnectorTool[]; // init 前可为空（MCP）
  status(): ConnectorStatus;         // { id, displayName, ok, detail?, toolCount }
}
class ConnectorRegistry {
  register(connector): void;
  attachHub(hub: ToolRegistrar): void;  // 立即注册已有工具
  initAll(): Promise<void>;             // init 异步连接器并补注册新工具
  statuses(): ConnectorStatus[];
}
```

- 连接器工具就是普通 ToolHub 工具：治理 gate、熔断、allowlist 全部照旧，
  框架**不另开旁路**。
- 风险分级沿用 RFC-013 `classifyToolRisk`：文件类只读工具
  `permissions: ["filesystem.read"], idempotent: true` → read；
  MCP 工具权限未知，`permissions: ["mcp.tool"]` + 非幂等 → 默认 write，
  桌面默认配置下首次调用要审批，可用 grant 放行——这是刻意的安全默认。
- 配置只走 env / opts，不落盘凭据：`OPENINTJ_NOTES_DIR`、
  `OPENINTJ_CALENDAR_ICS`、`OPENINTJ_MAIL_MBOX`、
  `OPENINTJ_MCP_SERVERS`（JSON 数组 `[{id, command, args?, env?}]`）。
  三端 agent 统一暴露 `connectors` 与 `initConnectors()`；宿主在装配后
  调用 `initConnectors()`（失败只记日志，不阻断启动）。

## 2. 只读连接器（v1）

| 连接器 | 工具 | 数据源 |
|---|---|---|
| notes | `notes_list` / `notes_read` / `notes_search` | 本地目录递归（.md/.txt，深度与数量有界，防路径逃逸） |
| calendar | `calendar_upcoming` / `calendar_search` | `.ics` 文件；RRULE 仅展开 DAILY/WEEKLY/MONTHLY/YEARLY + INTERVAL/COUNT/UNTIL，窗口内有界 |
| mail | `mail_list` / `mail_read` / `mail_search` | mbox 文件；RFC2047 主题解码、multipart 取首个 text/plain |

解析器全部手写零依赖（量级可控），正文输出统一截断，防上下文爆炸。

## 3. MCP 客户端

- 传输层抽象 `McpTransport`（send/onMessage/close）：stdio 实现
  （spawn 子进程、换行分隔 JSON）+ 内存实现（测试）。
- `McpClient`：initialize（protocolVersion `2024-11-05`）→
  `notifications/initialized` → tools/list / tools/call，请求超时可配。
- 工具命名：`mcp_<serverId>_<tool>`（非法字符折为 `_`）。
- server 进程随 `close()` 终止；连接失败 → connector status `ok:false`，
  不影响其它连接器与 agent 本体。

## 4. 宿主面

| 宿主 | 入口 |
|---|---|
| Desktop | IPC `connectors.list`（状态）+ preload；main 装配后 `initConnectors()`。专用面板暂缓 |
| Server | `GET /api/connectors`；进程启动后 `initConnectors()` |
| CLI | `agent.connectors` 编程面 + env 连接器 |

## 5. 验收

- 单测：ICS 解析（全天/UTC/浮动时间、RRULE 展开与窗口裁剪）、mbox 解析
  （编码主题、多段正文）、notes 沙箱（路径逃逸拒绝）、MCP 全链路
  （内存传输 + 假 server：initialize/list/call/错误）、registry 补注册。
- 接线：带 notes 连接器的 agent 可经 ToolHub 调到 `notes_search`；
  MCP 工具出现在 `toolHub.list()` 且名字带命名空间。
- 治理：MCP 工具风险级为 write（未知权限默认），文件类为 read。

## 6. 不做

- 不做可写连接器（发邮件/建日程/改笔记）——等 RFC-013 grant 语义在
  真实场景验证后再开，归 P2。
- 不做 OAuth / 凭据保险库 / 云端 API；不做 MCP 的 resources/prompts/
  sampling，只做 tools；不做 MCP over HTTP/SSE（stdio 先行）。
- 不做连接器热插拔与市场；连接器在装配期固定。

## 7. 落地记录（2026-11-05）

- 新包 `@openintj/connectors`（依赖 `@openintj/core` 工具类型）：
  `Connector/ConnectorRegistry`（同步工具 attachHub 即注册、MCP 等异步
  连接器 initAll 后补注册、按工具名幂等）；三个手写零依赖解析器连接器
  （notes 目录沙箱、ICS + RRULE 展开、mbox + RFC2047 解码）；MCP
  stdio 客户端（initialize/tools/list/tools/call、请求超时、子进程托管）
  + 内存传输；`connectorsFromEnv`（四个 OPENINTJ_* 变量）。7 个单测全过
  （含假 MCP server 全链路与失败隔离）。
- 三端接线：opts `connectors`/`enableConnectors`，装配时 opts 连接器与
  env 连接器叠加注册进同一个 ToolHub；agent 统一暴露 `connectors` +
  `initConnectors()`；server/desktop `close()` 时关停 MCP 子进程。
  server `GET /api/connectors`、desktop IPC `connectors.list` + preload；
  server main 与 desktop main 装配后后台 `initConnectors()`（失败只警告/
  记状态，不阻断启动）。
- 治理验证：notes/calendar/mail 工具 `filesystem.read` → read 级；
  MCP 工具 `mcp.tool` + 非幂等 → 默认 write 级，桌面默认审批配置下
  首次调用需批准（CLI 接线测试断言工具进 ToolHub 且可调用）。
- 验证：connectors 7 / CLI 46 / server 70 / desktop 101 测试全过；
  六包 `tsc --noEmit` 全过；biome check 已跑。
- 已知取舍：桌面暂无连接器面板（状态经 IPC 可查，UI 暂缓）；ICS 不处理
  TZID 时区换算（浮动时间按本地）；MCP 只做 tools + stdio。
