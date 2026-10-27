# openINTJ Agent Console

目标治理 + 安全执行的桌面 AI Agent 工作台。架构依据 `delivery/` 下五份已过 Gate 的设计文档实现。

当前版本：**v0.1.0-alpha · Phase 7 首版（壳层与进程骨架）**

---

## 快速开始

```bash
npm install          # 首次；Electron 二进制走 .npmrc 里的 npmmirror 镜像
npm run typecheck    # 主进程 + 渲染层类型检查（两次 tsc）
npm run dev          # 开发模式（HMR）
npm run build        # 构建三进程 + 渲染层到 out/
npm run smoke        # 构建后跑冒烟自检并退出（CI 可用）
npm run start        # 以已构建产物启动
```

> **务必用 `npm run dev` / `npm run start`，不要直接 `electron .`。**
> 部分宿主环境会注入 `ELECTRON_RUN_AS_NODE=1`，让 Electron 退化成纯 Node，
> 表现为 `Cannot read properties of undefined (reading 'whenReady')`——看着像代码 bug，其实是环境。
> `scripts/run.mjs` 已在启动子进程前剔除该变量。

---

## 打包桌面版

```bash
npm run dist:dir   # 只出 dist/win-unpacked/（可直接双击 openINTJ.exe 运行，最快）
npm run dist       # 出 dist/openINTJ-Setup-<version>.exe（NSIS 安装程序）
```

配置在 `electron-builder.yml`。两点值得知道：

- `npmRebuild: false` —— `better-sqlite3` 与 `node-pty` 都是 Node-API 产物，本机编好的在 Electron 里直接能加载，开 rebuild 反而要依赖 MSVC + Python 工具链
- `extraResources` 把 `vendor/vscodium-server`（284MB）分发到 `resources/`，主进程 `locateServer()` 在打包态找的就是这个位置；**先跑一次 `npm run vscodium:fetch`**，否则打包出来的编辑器页会显示「服务端未安装」
- 打包态的工作区根目录不再取 `process.cwd()`（那会是安装目录），默认落到 `%USERPROFILE%/openINTJ`

---

## 进程模型

| 进程 | 位置 | 职责 |
| --- | --- | --- |
| P-01 主进程 | `src/main/` | 窗口、IPC 路由、`safeStorage` 保险箱、子进程监护。不承载 Agent 主循环，不执行命令 |
| P-02 渲染进程 | `src/renderer/` | 界面。`contextIsolation=true` / `nodeIntegration=false`，只经 preload 窄接口访问 |
| P-03 agent-host | `src/agent-host/` | `utilityProcess`。Agent 主循环、事件写入、检查点、上下文压缩、LLM 调用 |
| P-04 tool-host | `src/tool-host/` | `utilityProcess`。命令执行、风险分级、会话级授权 |
| preload | `src/preload/` | `contextBridge` 暴露的窄接口，不暴露 `ipcRenderer` 本体 |

**进程间一律走 `MessagePort` 结构化克隆，不开本地 HTTP 端口**（《系统设计》§3.1、C8）。
主进程侧的请求/响应以 `requestId` 关联，**超时即 reject，绝不无限等待**。

---

## 目录

```
src/
  shared/          跨进程契约（唯一事实来源）
    events.ts      事件封闭词表 + EventEnvelope + 任务状态机 + 检查点
    approval.ts    三档真值表 + RiskDecisionContext 六要素 + 平台策略
    approval-token.ts  HMAC 审批票据（签发 / 校验）
    ipc.ts         WireEnvelope + BootInfo + 方法名 + 事件通道
    llm.ts         三协议配置（含 parseProtocol 唯一判定点）+ UnifiedMessage/ContentPart/TokenUsage
    contract.ts    目标契约（IC-07 硬闸门的对象）
    redaction.ts   脱敏管线（密钥九类规则 + 路径 + ANSI）
    mcp.ts         MCP 配置/工具命名契约（mcp__server__tool）
    terminal.ts    终端契约（会话 / 提交结果 / 输出事件 / 「不脱敏」的边界）
    ide.ts         IDE 契约（阶段 / 状态 / 布局矩形）+ 「外部产品边界」四条收紧的说明
    schema.ts      SQLite 表结构 + 迁移 + 版本号
    ulid.ts        无依赖单调 ULID
  main/            主进程：index.ts（路由 / 保险箱 / 档位恢复）+ process-manager.ts（WorkerHost）
                   vscodium.ts（直启 reh-web 服务端：随机端口 + 随机令牌 + 就绪探测 + 杀进程树）
  agent-host/      event-store.ts（SQLite）+ index.ts（run 状态机）+ agent-loop.ts（模型驱动循环）
                   llm/ = anthropic.ts + openai.ts + responses.ts（三套协议各自原生实现，index.ts 只做分派）
  tool-host/       classifier.ts（R1~R5 规则）+ terminal.ts（PTY 会话 + 命令模式）+ mcp-client.ts（MCP 连接池）+ audit-log.ts + index.ts（裁决 + spawn）
  preload/         窄接口
  renderer/        App.tsx + SettingsPage.tsx + TerminalPanel.tsx（xterm.js）+ IdePanel.tsx + styles.css（承接设计资产视觉系统）
scripts/run.mjs   统一启动器（见上方环境坑说明）
scripts/vscodium/fetch.mjs  按需拉取 VSCodium reh-web 服务端到 vendor/（约 94MB，不入库）
scripts/fixtures/ 探针用 stdio MCP server（自足，不装包、不联网）
delivery/         五份架构文档 + 交付说明 + 验收报告
```

数据落盘位置：`%APPDATA%/openintj/openintj.db`（P-01 经 `OPENINTJ_DATA_DIR` 下发给子进程）。
应用设置在同一库的 `t_app_setting` 表（首版只有审批档位）；MCP 配置在 `%APPDATA%/openintj/mcp.json`。
调试时可用 `OPENINTJ_EVENT_STORE=memory` 走内存实现，但日志会明确警告「重启即丢」。

---

## 已落实的硬约束

| 约束 | 落点 |
| --- | --- |
| 裁决权不外包 | `ToolExecParams` 里没有 `risk` 字段：风险分级、档位、工作区根、裁决结果全部由 tool-host 自持；调用方只能「请求执行」，不能「声明已批准」 |
| 「已批准」凭票据 | 主进程 `issueTicket` 签发 HMAC 票据，绑定 `runId` + 命令哈希 + 风险等级，单次使用、5 分钟过期；密钥只下发给 tool-host，agent-host 无密钥即无法伪造 |
| 审计由执行方写 | tool-host 自己写 `t_approval_audit`；**先留痕再执行**，被拦下的尝试同样入库（`executed=0`） |
| 拒绝也要留痕 | 用户拒绝 / 审批超时走 `tool.audit.deny`：只分级 + 落审计，不执行 |
| 无审计不执行 | 审计库未就绪时 `exec` 直接拒绝，不「先跑了再说」 |
| 审批超时即拒绝 | 120s 无响应按拒绝处理（`basis:'timeout'`），绝不「超时视为同意」 |
| 手动入口不是旁路 | 控制台「发起调用」与将来 LLM 驱动的工具调用同一入口 `agent.tool.invoke`，同一条裁决链 |
| 未获批准不产生进程 | 裁决不通过时在 `spawn` 之前抛错 |
| 命令不经 shell | `spawn(file, args, { shell: false })`，自研引号解析，不做变量展开与管道 |
| 最小环境变量 | 只透传 `PATH` / `HOME` / `LANG` / `GIT_TERMINAL_PROMPT=0`，避免凭据随子进程泄露 |
| 危险规则优先不可降级 | `classifier.ts` 先扫 R4-01~R4-08，命中即 L3，后续规则不得下调 |
| cwd 越界即按工作区外 | 用相对路径消解边界的情况在分类前就被拦 |
| Windows 禁「宽松」档 | `isTierAllowedOnPlatform('autonomous','win32') === false`，UI 上该胶囊 disabled；收紧档位时清空会话级授权 |
| C1 降级必显 | `BootInfo.sandbox.notice` 顶部常驻告警条，不提供关闭开关 |
| 保险箱不可用即 fail-stop | `api:vaultStatus` 返回 `policy:'fail-stop'`，绝不回退明文文件 |
| 不伪造模型输出 | 模型未配置 / 解析失败时 `GoalContractDrafted` 显式标 `awaiting-model` / `unparseable`，绝不产出看似合理的假契约 |
| 模型也只能「请求执行」 | Agent 循环拿到的工具里没有 `risk`、没有 `approval`。`run_command` 只是请求，分级与裁决仍在 tool-host；票据由主进程直发 tool-host，循环全程不知道票据长什么样 |
| 循环必有终点 | 步数硬上限 `MAX_STEPS=12`；唯一正常出口是模型调 `finish`。「说了几句就停下」判 `failed`，不算完成 |
| 交付不许凑数 | `finish` 要求逐条给出 `satisfied`（含证据）与 `unmet`（含原因）；未满足的判据不许塞进 satisfied |
| 结果以事件为准 | 执行结果卡片从 `RunDelivered` / `RunFailed` 事件派生，不依赖某次 IPC 返回值 —— 界面重开结果还在 |
| 界面上的东西必须是活的 | 审批卡是否消失由 `ApprovalDecided` 事件决定，而不是由「谁点的按钮」决定 —— 只在 onClick 里移除是不够的，别的应答路径会让卡片永远贴在屏幕上。run 收尾时再经 `approval.cancel` 通道收回无人等待的请求 |
| 明文密钥不落盘 | `safeStorage` 加密后写 `credentials.enc`；冒烟探针**把文件字节读出来搜明文密钥**，搜得到即判失败 —— 不靠「代码里写了加密」来自证 |
| 保险箱不可用即拒绝保存 | 没有系统密钥链就不写，绝不降级成明文文件 |
| 明文密钥不回渲染层 | 只给掩码（`sk-…tEbM`）。明文只出现在两处：保存那一瞬、下发给 agent-host 的 MessagePort |
| 凭据优先级只在一处判断 | `resolveCredential()`：保险箱 > 环境变量。子进程按需反向拉取，不自己判断优先级 —— 两处实现迟早会不一致 |
| 密文解不开要说清楚 | 「文件不存在」与「解密失败」分开报。混报成「未配置」会让用户以为配置丢了，而实际是换了机器或密钥环被重置 |
| 用量口径统一 | `TokenUsage.inputTokens` 恒为「未命中缓存的输入」，`cacheReadTokens` 是并列的命中量；OpenAI 侧已减去 `cached_tokens`，避免缓存命中被按全额重复计费 |
| 关思考三套各自实现 | `disableThinking` 在 `anthropic.ts` / `openai.ts` / `responses.ts` 里各写一遍、各测一遍。同一个「关思考」在 `/v1/messages`、`/v1/chat/completions`、`/v1/responses` 上是三种写法、三种效果 —— 只实现一套会让开关在另一套上**静默失效**（不报错，只是慢） |
| 协议判定只有一个入口 | `shared/llm.ts` 的 `parseProtocol()`：环境变量、保险箱凭据、设置页三条路径都走它。各写一份三元表达式的后果是「界面上选了 Responses，实际打 `/v1/messages`」—— 而且探针全绿 |
| 协议名认不出不猜 | 未知协议名按「配置不完整」返回、把原因写进 `missing` 让界面显示；绝不默认 anthropic。探针 ⑮-b 会比对「声明的协议」与「实际生效的协议」，不一致就红 |
| 密钥脱敏无条件生效 | 落盘前统一过 `emitEvent`，**调用方请求 `redaction:'none'` 也照样遮密钥**，命中即把级别升为 `secret`。底线不能依赖调用方标对级别 |
| 脱敏不许误伤 | 刻意不遮「32 位以上 hex」—— git commit hash 正是 40 位，遮掉它排障就废了。输出脱敏保留路径结构（`~/projects/app/src/index.ts`），只换凭据字面量 |
| MCP 调用不是旁路 | MCP 工具走**同一条**裁决与审计链：`tool.mcp.precheck` → 审批 → `tool.mcp.call`。MCP 客户端在 tool-host 内，不在 agent-host —— 放到 agent-host 就等于给模型开了一条侧门 |
| MCP 票据绑定参数 | 审批票据绑定的是 `mcp:server:tool {参数}` 这串的哈希，换参数即失效。MCP 没有命令行，不绑参数的话「批准一次」就能被换成任意参数重放 |
| MCP 分级只看 trust | 不复用 shell 分类器（正则匹配命令字符串对结构化参数无意义）：`readonly` → L0 自动放行，`confirm` → L2 逐次审批；找不到工具时按 `confirm` 兜底 |
| MCP 被拒也分对类 | 拒绝走 `tool.mcp.deny` 而不是复用 shell 的 `recordDenial` —— 后者会给 `mcp:...` 这串算出错误的风险等级，审计里写错等级比不写更糟 |
| MCP 失败不 fail-stop | 一个 server 起不来只是少几个工具，但**失败原因必须上报到界面**（`McpServerStatus.error`）。只显示「工具少了一个」会让用户以为产品坏了 |
| MCP 配置坏 ≠ 没配置 | `mcp.json` 解析失败报 `configError`，与「没配过」分开显示 |
| 档位重启不丢 | 落 `t_app_setting`；恢复时**三处一起改**（主进程内存 / tool-host 裁决方 / 渲染层推送）。只改一处的结果是「界面显示严格档、实际按标准档裁决」，比不持久化更危险 |
| 档位先改到位再落库 | 顺序是「推 tool-host → 推 agent-host → 落库」。反过来的话落库成功而推送失败，用户会以为自己已经收紧了 |
| 脏设置不猜 | 库里存了无法识别的档位（或本平台不允许的档位）就回落默认并告警，不照着用 |
| 终端不是旁路 | 终端提交的是**整行命令**而不是按键流：`tool.term.input` 在 tool-host 内分级 → 裁决 → 放行才写进 pty。掌握命令边界是「执行前拦截」的唯一前提；键盘直通下回车已经进 pty，再拦就晚了 |
| pty 住在 tool-host | 终端会产生进程，所以和命令执行同住 P-04（裁决权持有者）。放到主进程或渲染层等于给终端开了一条不经裁决的通道 |
| pty 不许留孤儿 | 会话关闭 / 进程退出 / SIGTERM 三处都回收：`node-pty` 的 shell 不会跟着 Utility Process 一起死。留下的孤儿 powershell 极难归因到「那个终端」 |
| 面板提示不写进 pty 区域 | 提示符那一行归 shell 所有：往里插字会被 PowerShell/PSReadLine 重绘时的 `\x1b[K` 擦掉，表现为「代码写了、用户没看见」。所有面板级提示走独立的提示条 |
| 终端输出不脱敏，但也不落盘 | 终端回显是用户自己屏幕上的东西（他 `cat .env` 是他自己的选择，遮掉只会让他以为文件是空的）；所以它不进事件库、不过 `emitEvent`。将来若要落盘或喂模型，必须先补脱敏 |

---

## 冒烟自检口径

`npm run smoke` 的每一行都是**可失败的断言**，不是日志：

1. `P-03 / P-04` 拿到跨进程请求返回值（证明端口真绑上了，不只是进程起来了）
2. `rm -rf /tmp/...` 被判 `L3 / R4-01`
3. 无票据执行 L3 → 拒绝，且不产生进程
4. **伪签票据**（错误密钥）→ 拒绝，证明 agent-host 造不出有效批准
5. 票据跨命令复用 → 拒绝，证明票据绑定命令
6. 只读命令自动放行并拿到真实输出（`node -v`）
7. 会话授权：首次凭 session 票据，复用免票据
8. 界面实际渲染文本含版本号、三进程「在线」、C1 告警原文

加 `--smoke-run` 可再验一轮完整闭环：渲染层 → 主进程 → agent-host → 事件回流渲染层。
加 `--smoke-approval` 验 D4 审批闭环：**在真实渲染进程里读界面文字、点界面按钮**
（卡片上屏 → 点「允许一次」真的执行 → 点「拒绝」真的不执行 → 4 条并发聚合成批量卡 → 批量拒绝清空队列）。
加 `--smoke-llm` 验真实模型起草契约（需 `OPENINTJ_LLM_*`）：⑮ 接入状态 → **⑮-b 声明协议与实际生效协议一致**（防止协议被静默降级）→ ⑯ 契约由模型起草 → ⑰ 卡片上屏。
加 `--smoke-agent` 验 Agent 主循环端到端：⑱ 契约起草 → ⑲ 确认过闸门 → ⑳ 模型自己驱动到收尾 → ㉑ 它请求的命令确实过了裁决链（事件流里有 `ModelMessage` + `ToolInvoked` + `riskClass`）→ ㉒ 结果卡片上屏 → ㉓ 收尾后无残留审批卡。三套协议（`anthropic` / `openai` / `openai-responses`）各跑一遍才算验完 —— ⑳ 的 `steps>=2` 是「工具调用被正确识别」的唯一证据。
加 `--smoke-cred` 验凭据中心：㉔ 存进保险箱 → ㉕ 回传给界面的只有掩码 → ㉖ **把落盘文件字节读出来搜明文密钥（搜得到就算失败）** → ㉗ 生效来源切到 vault → ㉘ 用保险箱里的凭据真调一次模型 → ㉚ 设置页上屏 → ㉛ 清除后回落环境变量。探针结束会把保险箱按原字节还原，不留副作用。
加 `--smoke-redact` 验脱敏管线（D8）：㉜ 声明 `none` 仍遮密钥（级别自动升 `secret`）→ ㉝ 私钥 / JWT / 连接串密码 / kv 四类都遮 → ㉞ **40 位 commit hash 不被误伤** → ㉟ 路径脱敏后结构仍可读 → ㊱ 嵌套结构与非密钥内容原样 → ㊲ ANSI 控制序列清理。
加 `--smoke-mcp` 验 MCP 接入（D6）：⓪ init 已下发配置路径 → ㊳ 两个 server 真连上 → ㊴ 工具发现与命名解析往返一致 → ㊵ 只读 server 自动放行并拿到真实返回 → ㊶ confirm 级无票据被拒 → ㊷ 被拒调用已入库（`executed=false`、规则 `R6-mcp-call`）→ ㊸ 持票据执行成功 → ㊹ **票据绑定参数（换参数即失效）** → ㊺ server 返回 `isError` 如实透传 → ㊻ 设置页真的列出了 server → ㊼ MCP 工具真的进了发给模型的 tool spec → ㊽ 坏配置如实上报 → ㊾ 命名规范校验。探针自带一个 stdio MCP server（`scripts/fixtures/`），跑完按原字节还原用户自己的 `mcp.json`。
加 `--smoke-tier` 验档位持久化（D9）：51 界面点「严格」→ 库里与裁决方同时变 → 52 模拟重启（三处全部打回默认，含重载渲染层）→ 53 恢复时三处一起回到严格 → 54 库里塞脏值被忽略并回落默认。
加 `--smoke-term` 验内嵌终端（US-15）：55 真拉起 PTY 会话 → 56 只读命令自动放行且输出是真 PTY 输出 → 57 **严格档下同一条命令立刻被拦在审批条前**（终端不是旁路）→ 58 **批准前命令确实没有执行**（只验「有没有弹卡」不够）→ 59 拒绝后提示条明示原因且审计留下 `executed=false` → 60 允许后真的执行 → 61 票据绑定命令（换命令即失效）→ 62 resize 下发到 pty → 63 中断不经裁决 → 64 关闭后会话回收、再提交返回 `no-session`。
加 `--smoke-ide` 验内嵌 IDE（F27 / D5）：65 编辑器页可用 → 66 服务端就位（版本取自 reh-web 的 `product.json`）→ 67 点启动到 `ready` 且耗时有值 → 68 **错令牌 403**（连接令牌真的在校验）→ 69 正确令牌 302 → 70 **去连非回环地址连不上**（真的只绑了 `127.0.0.1`，不是只看 URL 里写了什么）→ 71 独立 view 里查得到 `.monaco-workbench`（页面真渲染了）→ 72 点停止到 `stopped` → 73 **端口不再应答**（进程树杀干净，没留占端口的孤儿）→ 74 界面文案不含禁用商标。
加 `--snapshot` 把当前界面存成 PNG（走 Electron 自带 `capturePage`，不依赖系统截图工具），落在 `.snapshots/`。与 `--smoke-mcp` 同跑会额外存一张 MCP 面板的实拍；与 `--smoke-ide` 同跑会额外存一张工作台实拍。

```bash
npm run smoke -- --smoke-run
npm run smoke -- --smoke-approval
npm run smoke -- --smoke-run --smoke-approval   # 三条一起跑
npm run smoke -- --smoke-cred --snapshot        # 凭据中心 + 界面实拍
npm run smoke -- --smoke-redact --smoke-mcp --smoke-tier   # D8 / D6 / D9
npm run smoke -- --smoke-term --snapshot        # 内嵌终端 + 实拍
npm run smoke -- --smoke-ide --snapshot        # 内嵌 IDE + 界面/工作台实拍
```

> 编号 ①~㊿ 已用尽，51 起改用普通数字。数字是可追溯的承诺：改行为就得改断言，删断言要在验收报告里写明原因。

### 配模型：界面里配（推荐）

启动后切到顶部 **设置与凭据** 页，填协议 / Base URL / 模型 / API Key，点「保存到保险箱」。

- 密钥由系统密钥链加密后写 `%APPDATA%/openintj/credentials.enc`，**明文不落盘**
- 保存后界面只显示掩码（`sk-…tEbM`）；改模型名时 Key 留空即沿用原值
- 「测试连接」会用当前凭据真发一次最小请求，把「连得上」和「用不了」区分开
- 系统密钥链不可用时**拒绝保存**，不会退回明文文件

### 配模型：环境变量（等价备选，适合 CI / 脚本）

保险箱为空时才会读环境变量。两者都有值时**保险箱优先**，界面会显示当前生效来源。

```bash
# bash
export OPENINTJ_LLM_PROTOCOL=anthropic
export OPENINTJ_LLM_BASE_URL=https://tokenhub.tencentmaas.com/v1
export OPENINTJ_LLM_API_KEY=sk-xxx
export OPENINTJ_LLM_MODEL=deepseek/deepseek-flash
export OPENINTJ_LLM_DISABLE_THINKING=1     # 默认就关；写 0 才开

# PowerShell 等价写法
$env:OPENINTJ_LLM_PROTOCOL='anthropic'
$env:OPENINTJ_LLM_BASE_URL='https://tokenhub.tencentmaas.com/v1'
$env:OPENINTJ_LLM_API_KEY='sk-xxx'
$env:OPENINTJ_LLM_MODEL='deepseek/deepseek-flash'
```

> 凭据只走环境变量，**不写进任何配置文件**。这是「设置与凭据中心」（IC-06）落地前的唯一安全路径：
> 配置文件会落明文盘，环境变量不会。代价是每次开新终端要重设一遍。
> 设置前先确认 `OPENINTJ_LLM_API_KEY` 没被别处的 shell 历史或脚本记下来。

> 模型选型实测（tokenhub 网关）：
>
> | 模型 | 协议 | 关思考后 | 备注 |
> | --- | --- | --- | --- |
> | `hy4-preview` | openai | 1.24s / 7 tok（一句「你好」） | 不关思考是 8.81s / 231 tok（其中思考 221） |
> | `hy4-preview` | openai-responses | 1.30s / 7 tok | `reasoning:{effort:'none'}`，官方规范，真关掉了 |
> | `hy4-preview` | anthropic | **关不掉** | `/v1/messages` 上不认 `thinking:{type:'disabled'}` |
> | `deepseek/deepseek-flash` | anthropic | ~3s 出合规 JSON | 契约起草实测首选 |
> | `glm-5.3-flash` | — | 不认 | 直接 400，代码会去掉该参数重试一次 |
>
> **同一个参数在三套协议上行为可能完全不同** —— 关思考是三套各自实现、各自实测的，不能互相推论。
> 端到端实测：OpenAI 协议 + `hy4-preview` + 关思考，契约起草 → 确认 → 2 步交付，输出 246 tok。

#### 三套协议怎么选

| 协议 | 端点 | 什么时候用 |
| --- | --- | --- |
| `anthropic` / `claude` / `messages` | `POST {baseUrl}/messages` | Claude 原生；要 prompt caching 与完整 thinking block |
| `openai` / `chat` | `POST {baseUrl}/chat/completions` | 兼容面最广，网关基本都认 |
| `openai-responses` / `responses` | `POST {baseUrl}/responses` | 网关支持 Responses 时优先它：工具定义更简洁、`reasoning.effort` 是官方规范 |

三者的线格式差别不是「换个字段名」的量级，所以是三个独立文件（`anthropic.ts` / `openai.ts` / `responses.ts`），
**不共用一个「OpenAI 兼容层」**：

| | Chat Completions | Responses |
| --- | --- | --- |
| 系统提示 | `messages[0].role='system'` | 顶层 `instructions` |
| 输出 | `choices[0].message` | `output[]`（多个 item 并列） |
| 工具调用 | `message.tool_calls[]` | `output[]` 里 `type:'function_call'` 的 item |
| 工具结果 | `role:'tool'` + `tool_call_id` | `{type:'function_call_output', call_id, output}` |
| 工具定义 | `tools[].function{name,description,parameters}` | `tools[]` **扁平** `{type,name,description,parameters}` |
| 结束原因 | `finish_reason` | **没有这个字段**，从 output 里有没有 `function_call` 反推 |
| 关思考 | `thinking:{type:'disabled'}` | `reasoning:{effort:'none'}` |

> 最后一行「没有 finish_reason」是最容易写错的地方：漏了 `function_call` 判断，
> 模型一调工具就被当成「说完了」，Agent 循环直接退出 —— 表现为「Agent 只会聊天，不干活」。

> 协议名认不出时**不会静默降级成 anthropic**：`readLlmConfig` 会把它当成「配置不完整」返回，
> 设置页把 `OPENINTJ_LLM_PROTOCOL=xxx（未知协议…）` 原样显示出来。
> 静默降级的症状是「探针全绿但测的不是想测的通道」，比报错难查得多。

---

### 配 MCP server

配置文件在 `%APPDATA%/openintj/mcp.json`（不放工作区里 —— 否则 Agent 能自己改掉它）。格式与其他 MCP 客户端一致：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:/work"],
      "trust": "confirm"
    },
    "my-reader": {
      "command": "node",
      "args": ["D:/tools/my-reader-mcp.js"],
      "trust": "readonly"
    }
  }
}
```

- `trust` 决定分级：`readonly` → L0 自动放行，`confirm`（默认）→ 每次调用都要你点头。
  外部进程能干什么静态判断不了，所以默认最谨慎那档；只有你确定那个 server 只读时才改成 `readonly`
- `enabled: false` 可以留着配置但不启动；`connectTimeoutMs` 给 `npx` 现装包留足时间（默认 20s）
- server 名里**不能有 `__`**（工具名用 `mcp__<server>__<tool>` 拼，双下划线是分隔符；用冒号会被 OpenAI 的 function name 规则拒掉）
- 改完在 **设置与凭据** 页点「重新加载」，或用 `--smoke-mcp` 之外的正常启动流程重启（init 阶段会自动加载）
- 连不上的 server 不会拖垮应用，但失败原因会显示在设置页上

---

## 内嵌 IDE（D5 / F27）

首次使用先拉一次服务端（约 94MB，一次性；`vendor/` 不入库）：

```bash
npm run vscodium:fetch     # 拉最新稳定版到 vendor/vscodium-server
npm run vscodium:check     # 只检查装了没，不下载
```

然后启动应用 → 切到 **编辑器** 页 → 点「启动」。工作区默认是项目根目录。

### 为什么不是 `serve-web`

`codium-tunnel serve-web` 会在**首次 HTTP 请求时**联网下载服务端本体（实测
94,038,270 字节 / 99 秒），期间每个请求都返回 202 + 一页「正在下载」——
首次打开是 100 秒白屏、断网不可用、版本不可复现。

而它下载的那坨（`vscodium-reh-web-*`）**本身就是一个可独立运行的 HTTP 服务端**，
所以我们自己拉一次、运行时直启 `node.exe out/server-main.js`：**启动 ~300ms**、
离线可用、版本由 `vendor/vscodium-server/SOURCE.json` 钉住。

### 边界怎么收紧的

| # | 措施 |
| --- | --- |
| 1 | 只绑 `127.0.0.1`；端口传 `0` 交给系统分配（不用默认 8000、不撞本机已有服务） |
| 2 | 每次启动现生成 `randomBytes(24)` 令牌，**只留在内存**；`IdeStatus.url` 只给渲染层 origin，令牌不出主进程 |
| 3 | 停止时 `taskkill /pid <pid> /T /F` 杀整棵树（它会派生扩展宿主，只杀父进程会留下占端口的孤儿） |
| 4 | 工作台用独立 session 分区（`persist:ide`）的 `WebContentsView` 承载，与主界面 cookie / storage 隔离；`before-quit` 里 `dispose()` |

HTTP 握手（实测）：无令牌 / 错令牌 → **403**；`?tkn=<正确>` → **302** 并下发
cookie `vscode-tkn`；带该 cookie 再取 `/` → **200** + workbench HTML。

### 代价（不是缺陷，是取舍）

工作台是**主进程盖在占位区上的一层 `WebContentsView`**，不是 iframe ——
iframe 第三方工作台要放松 CSP，而 CSP 是整个渲染进程一份的，为一个面板降主界面
的安全基线不划算。代价是那块区域对渲染层来说是「透明的洞」：没法在上面叠自己的
DOM，也截不到它的键盘事件，所以面板状态条放在占位区**上方**。

另外，服务端本身**没有**我们 tool-host 那套裁决链 —— 它能在工作区里读写文件是
「外部产品」这个选择的固有代价，首版不做二次拦截。

---

## 从主干合入的四项能力

主干（`origin/main`）上的 harness 是**另一套架构**（`ts/apps/desktop` + monorepo 包），
与本分支的 `src/` 契约无法目录级合并（试合产生 49 个 modify/delete 冲突）。
所以按「移植到 `src/` 架构」的方式重做：只把**算法**搬过来
（`src/vendor/openintj/`，来源与同步方式见那里的 README），
外围全部按本工程的进程分层与裁决链重写。

| 项 | 落在哪 | 关键约束 |
| --- | --- | --- |
| 工件识别 + 预览 | 一张扩展名表（shared/session）→ 文件树标记、预览登记、渲染三处共用 | 预览只渲染**工作区内**的工件；外部地址一律交系统浏览器 |
| 生图任务 | Agent 内置工具 `generate_image` → 产物落 `artifacts/` | **落盘必经 tool-host 裁决**（precheck → 票据 → 审计），agent-host 不直接 `fs.writeFile` |
| 工作区文件树 | tool-host 的 `fs.list` / `fs.read`（只读） | 只认工作区相对路径，上跳与绝对路径一律拒；读有 4MB 上限，超限如实说超了 |
| 输入结构化 | `structureUserInput` 前置到契约起草 | 歧义分 ≥ 阈值且有问题时进 `needsClarification`（最多 3 条）；结构化失败按 pass-through 继续，**不阻断起草** |

生图失败时落的是一张**写着失败原因的占位 SVG**，不是随机图也不是静默失败 ——
它同样要过审批才准落盘。

### 生图是独立凭据

与对话模型分开存（`VaultCredential.image` 段，同一份密文）。理由：端点不同
（`{baseUrl}/wand/hunyuan-image/v3-generation`）、计费不同（按张计费）、
失败原因也不同（402 = 未开通后付费且无免费额度）。
**不用对话模型的 key 兜底** —— 那样只会让生图报 401/402，
而用户会去改对话模型，越改越远。

```bash
export OPENINTJ_IMAGE_BASE_URL=https://tokenhub.tencentmaas.com/v1
export OPENINTJ_IMAGE_API_KEY=...
# 可选：OPENINTJ_IMAGE_MODEL=hy-image-v3  OPENINTJ_IMAGE_SIZE=1280x720
```

也支持在「设置与凭据 → 生图服务」里存进系统保险箱（需先存过对话模型凭据）。

---

## 尚未接入（按依赖排序）

| 项 | 依赖 |
| --- | --- |
| 审批倒计时提示 / 泳道间并发 / 审批历史查询 | 无 |
| 上下文压缩与检查点续跑（`ContextCompressed` 词表已留，逻辑未写） | 无 |
| 多套凭据档案（现在只有一份「当前生效」） | 无 |
| MCP 的 HTTP / SSE 传输（现在只有 stdio） | 无 |
| 终端不支持全屏交互程序 | 命令模式下「命令边界 = 换行」，vim / top / ssh 的按键流没法按命令裁决。这是为「终端不绕过策略层」付的价，不是待修 bug |
| 打包分发内嵌 IDE（`vendor/vscodium-server` → `extraResources`） | electron-builder 配置 |
| IDE 推荐/预装扩展（Open VSX 市场已默认可用） | 无 |
| 云端工作区 / 沙箱 / 消息通道 | 腾讯云账号与实例规格（Q4-a / Q4-b） |

设计文档已定、代码未写的内容一律以 `delivery/` 为准；本 README 不重复架构决策。
逐项偏离与实测见 `delivery/Phase7-首版验收报告.md`。

### 已定的超时阈值（N-04b，2026-10-22 拍板）

| 环节 | 阈值 | 超时怎么办 |
| --- | --- | --- |
| 模型单步（含契约起草、循环每一步） | 180s | 如实报错进事件流，**不自动重试** —— 静默重试会让「卡住」变成「更慢地卡住」 |
| 审批等待 | 120s | 按拒绝处理（`basis:'timeout'`），绝不「超时视为同意」 |
| 命令执行 | 由 tool-host 自管 | 超时 SIGKILL，输出 512KB 截断 |

---

## 版本与许可

v0.1.0 · UNLICENSED · 内部工程
