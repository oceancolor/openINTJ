# AICoding 架构设计 · 行业调研报告 — openINTJ

> 本文档为《AICoding 架构设计》核心产物之一，定位为**行业调研报告（research_report）**。
> 上游输入：`material_digest.md`（资料摘要 v1.0，knowledge-ingest-engineer 产出，G1 已通过）与用户原始诉求；
> 下游输出：驱动 `business-architect`（业务架构师）的行业调研判断，最终落入《高层架构设计》的 §3 行业调研章节。

> **工具说明**：由 `research-analyst`（研究分析师 - 查有据）负责产出，经 G2 自动校验与人工审核通过后方可进入下游消费。
> **结构纪律**：全文按「事实 → 对比 → 建议 → 风险」四段式组织，严禁四段之间倒序或跳段。

---

## 文档定位与阅读说明

| 约定 | 含义 | 本报告处理方式 |
| --- | --- | --- |
| 已核实 | 可追溯到官方文档、官方仓库或权威公开资料的具体段落 | 在 §6 给出 URL 与访问日期，正文标注「已核实」 |
| 推断 | 由多条已核实事实综合推导，无单一来源直接声明 | 正文标注「推断」并给出推导所依据的已核实事实编号 |
| 综合归纳 | 多个来源一致表述、但无官方单一声明 | 正文标注「综合归纳」 |
| 建议 | 本报告的取舍倾向 | §4 全章；**不构成决策授权**，最终边界由 business-architect 冻结 |
| 风险 | 可能对下游造成负面影响的不确定项 | §5 全章 |

**调研边界声明**：本轮调研基准日为 2026-10-20。凡涉及版本号、性能数字、定价与许可条款的陈述，均以调研当日公开资料为准，且必须指向 §6 的来源条目；未取得公开证据的部分一律列入 §5.2 待确认项，不做臆测。

---

## 0. 元信息：修订记录

```yaml
标题: openINTJ - 行业调研报告 v1.2
版本: v1.2
状态: Draft   # Draft | Reviewing | Approved | Deprecated
创建日期: 2026-10-20
最后更新: 2026-10-20
调研人: research-analyst（研究分析师 - 查有据）
审核人:
  - team-lead（主理人）

关联文档:
  上游输入:
    - 用户诉求: material_digest.md D1（套用 openINTJ 设计任务产出物，开发 openINTJ 程序）
    - 用户诉求: material_digest.md D2 §2.1（14 项能力原子原话）
    - 用户诉求: material_digest.md D2 §2.2（直连真实大模型 API，Anthropic 原生协议 + OpenAI 兼容协议双协议并行）
    - 用户诉求: material_digest.md D2 §2.3（先出架构文档再开发）
    - 调研目标: 由主理人在 Phase 2 任务下发时注入（9 家对标产品 + 12 个技术选型论题）
  下游产出:
    - 高层架构设计 §3 行业调研: 将由 business-architect 整合到此章节

已裁决事项（不得推翻）:
  - X1 产品形态 = 桌面客户端（非 Web 应用），以 D2 §2.1 为准
  - ternimal = terminal（终端）笔误
  - T1 桌面壳层 = Electron  # 裁决人: 用户；裁决日期: 2026-10-20；经中间确认 IC-01
  - T2 IDE 嵌入形态 = 内嵌 VSCodium（Code-OSS 的 MIT 构建，遥测已移除）+ Open VSX 扩展源  # 裁决人: 用户；裁决日期: 2026-10-20；经中间确认 IC-02
  - T8 云端沙箱供给方式 = 自建（自有云账号），云厂商 = 腾讯云  # 裁决人: 用户；裁决日期: 2026-10-20；经中间确认 IC-03 + IC-04

本轮发出的中间确认（3 题，均已获用户裁决并冻结）:
  - T1 桌面壳层选型: 经中间确认，裁决 = Electron（已冻结）
  - T2 IDE 嵌入形态: 经中间确认，裁决 = 内嵌 VSCodium（MIT 构建）+ Open VSX（已冻结）
  - T8 云端沙箱供给方式: 经中间确认，裁决 = 自建（自有云账号），云厂商 = 腾讯云（已冻结）
```

### 0.1 经中间确认的裁决记录

> 依 `skills/aicoding-team-bootstrap/protocols/intermediate_confirmation.md` §5，以下决策由用户裁决后回注，已写入本报告并标注为「经中间确认」。裁决人均为用户，裁决时间均为 2026-10-20。

| 编号 | 论题 | 裁决结果 | 裁决人 | 裁决时间 | 用户选择时所依据的代价说明 | 落点章节 |
| --- | --- | --- | --- | --- | --- | --- |
| IC-01 | T1 桌面壳层选型 | **Electron**（捆绑 Chromium + Node） | 用户 | 2026-10-20 | 用户已接受：包体 80–150MB、空闲内存 150–300MB、冷启动 1–3 秒 | §3.1.2、§3.2、§4.1 原子 2、§4.3、§5.1 R-09 |
| IC-02 | T2 IDE 嵌入形态 | **内嵌 VSCodium（Code-OSS 的 MIT 构建，遥测已移除）+ Open VSX 扩展源** | 用户 | 2026-10-20 | 用户已接受：不得使用 VS Code 商标，C/C++、C# Dev Kit、Remote-SSH / Remote-Containers / Remote-WSL 等微软专有扩展不可用 | §3.1.2、§3.2、§4.1 原子 3、§4.3、§5.1 R-01 |
| IC-03 | T8 云端沙箱供给方式 | **自建（自有云账号）**，云厂商定为腾讯云 | 用户 | 2026-10-20 | 用户是在**明知**「工程量最大，需自建镜像 / 快照 / 预热池 / 回收 / 网络策略，且必须先补齐云账号基线（Q4 未解则无法设计）」这一代价的前提下做出的选择，本记录保留该前提，不作淡化 | §3.1.2、§3.2、§4.1 原子 8/9/14、§4.3、§4.4、§5.1 R-02 / R-03 / R-10 |
| IC-04 | Q4 云基线（部分补齐） | 云厂商 = **腾讯云**（CVM / 轻量应用服务器 + COS + VPC + CAM）；国内合规与备案便利，数据驻留境内 | 用户 | 2026-10-20 | 用户明示「后续告诉我可用 Region 与预算上限即可」，即 Region 与预算仍在补缺中 | §4.4、§5.1 R-02 / R-03、§5.2 U-01 |

**IC-01 / IC-02 / IC-03 生效后，原候选清单中以下方案不再作为候选**：

- T1：Tauri 2 与 Neutralinojs 退出候选（其分值仅作为已记录的评估证据保留，不参与后续选型）。
- T2：VS Code 扩展形态、内嵌 code-server、fork Code-OSS 定制发行版退出候选。其中「连接用户已安装的 VS Code」仍保留为**降级路径**（非并列候选）。
- T8：E2B 与 Daytona 退出**供给方式**候选；E2B 因 Apache-2.0 且可自托管，保留为自建方案的**技术参考实现**（Firecracker microVM 路线）；Daytona 因生产平台于 2026-06 转为闭源，不适合作为参考。

| 版本 | 日期 | 作者 | 变更内容 | 评审状态 |
| --- | --- | --- | --- | --- |
| v1.0 | 2026-10-20 | `research-analyst` | 初稿：11 家标杆盘点 + 12 个技术选型论题加权评分 + 三层结论 + 8 条风险 + 附录含中间确认自检 | Draft |
| v1.1 | 2026-10-20 | `research-analyst` | 修订稿：回注 IC-01（Electron）/ IC-02（内嵌 VSCodium + Open VSX）/ IC-03（自建云端沙箱）/ IC-04（云厂商 = 腾讯云）四项裁决；新增 §4.4 自建云端沙箱最小能力集；新增 §0.1 裁决记录；风险清单改写 R-02 / R-03 并新增 R-09~R-12；来源新增 SR-60~SR-64 | Draft |
| v1.2 | 2026-10-20 | `research-analyst` | 定点返工稿（依主理人 G2 返工指令 11 条）：§0 yaml 补入 T1 / T2 / T8 三条已裁决事项并把中间确认行改为「均已获用户裁决并冻结」；§3.1.2 补写「分值一律保留不动」与各论题最终裁决结果；§3.2 结论形态列改为「已冻结（经中间确认，用户裁决 2026-10-20）」并在首位候选列补注实际裁决项；§3.3 组合 B 已为自建口径，并核对「以组合 C 为 MVP 形态」结论后下修为「组合 A + 本地沙箱」；§4.1 原子 8 补齐自建最小能力集六项名称；§4.2 补「云厂商已定后是否解锁」的重新判断；§4.3 T1 / T2 / T8 统一加注「（经中间确认，已冻结）」；§5.2 原 U-05 改写为腾讯云 CVM 嵌套虚拟化核实项（原暂编号 U-09 并入 U-05，全文引用同步）；附录 B.1 处置列补「已发起并获用户裁决，已回注冻结（2026-10-20）」及是否为打分首位 | Draft |

---

## 1. 调研问题收敛

> 调研启动前，先围绕用户诉求收拢为明确的调研问题集合，确保调研不偏离当前项目背景。

### 1.1 原始调研种子

> 从用户诉求（material_digest.md D2 §2.4 的 14 项能力原子）与主理人注入的 12 个技术选型论题中，提取需要调研验证的种子，逐条给出调研优先级。

| 编号 | 待验证论题 | 来源（用户诉求要点） | 调研优先级 | 备注 |
| --- | --- | --- | --- | --- |
| S1 | 桌面壳层方案（Electron / Tauri 2 / Neutralinojs）及其对 node-pty、内嵌编辑器的支撑能力 | D2 §2.4 原子 2「ts 全栈的桌面 AI agent 客户端」+ 主理人论题 T1 | 高 | 决定进程模型与包体基线 |
| S2 | IDE 嵌入形态与 VS Code 商标 / 许可证边界 | D2 §2.4 原子 3「嵌入 vscode 作为 IDE」+ 主理人论题 T2 | 高 | 涉及对外可感知的形态与合规 |
| S3 | 终端嵌入路径（xterm.js + node-pty / 复用 VS Code 集成终端 / 系统终端唤起） | D2 §2.4 原子 4「嵌入 ternimal（terminal）」+ 主理人论题 T3 | 中 | 依赖 S1 结论 |
| S4 | Agent SDK 形态与 LLM 双协议适配层设计 | D2 §2.4 原子 1 + D2 §2.2 双协议要求 + 主理人论题 T4、T5 | 高 | 用户已明确双协议，适配层不可省 |
| S5 | MCP 与 Skill 标准协议的当前标准状态与实现路径 | D2 §2.4 原子 6、原子 7 + 主理人论题 T6、T7 | 高 | Skill 是否已有统一标准是核心疑问 |
| S6 | 云端工作区 / 云端沙箱的隔离原语、启动时延、成本与数据驻留 | D2 §2.4 原子 8、原子 9 + 主理人论题 T8 | 高 | 与资料摘要 X3 冲突、Q4 阻塞项直接相关 |
| S7 | 本地沙箱安全运行环境与本地系统权限的跨平台实现 | D2 §2.4 原子 10、11、12 + 主理人论题 T9 | 高 | 安全设计的高风险项 |
| S8 | 消息通道（Webhook / IM Bot / MQ）的接入抽象 | D2 §2.4 原子 9「沙箱可以接入消息通道」+ 主理人论题 T10 | 中 | 用户未定义通道类型（资料摘要 Q5） |
| S9 | 长程任务编排（Goal-Milestone-Task 三层 + Critic 自检 + 重规划 + 检查点与恢复）的业界实现 | D3 §3.2 / §3.3 的 UI 模块清单 + 主理人论题 T11 | 高 | 直接承接上次设计产出物的 Screen A / Screen B |
| S10 | 本地与云端部署、测试、预览服务的实现路径 | D2 §2.4 原子 13、14 + 主理人论题 T12 | 中 | 依赖 S6 结论 |
| S11 | 同类产品的形态、商业模式与可借鉴程度 | 主理人注入的 9 类对标产品 | 高 | 为 §2 提供事实基线 |

### 1.2 调研问题收敛

> 将 §1.1 的种子收敛为 5 个可执行的调研问题。每条问题明确调研对象、调研目标和产出预期。

| 编号 | 调研问题 | 调研对象 | 调研目标 | 预期产出 | 关联种子 |
| --- | --- | --- | --- | --- | --- |
| RQ1 | 桌面端 AI Agent 客户端这一品类，现有头部产品的形态、技术栈组合与商业模式分别是什么？ | Claude Code / Agent SDK、OpenAI Codex CLI / Agents SDK、Cursor、Cline、Roo Code、OpenHands、GitHub Copilot Agent HQ、Windsurf、Devin、Zed、腾讯 CodeBuddy / WorkBuddy | 建立 11 家标杆的标签化画像与逐家详述，识别形态分歧点 | §2.1 标杆清单 + §2.2 详述卡片 + §2.3 横向事实表 | S11 |
| RQ2 | 14 项能力原子中，哪些已有成熟可复用组件、哪些必须自研？各自的技术栈与代价是什么？ | 桌面壳层、IDE 嵌入、终端嵌入三类框架与官方仓库；MCP 官方规范；Agent Skills 开放标准 | 对每个论题给出候选方案、证据与加权分值 | §3.1.2 技术选型评分矩阵 + §4.1 自研 / 采购 / 复用边界 | S1、S2、S3、S5 |
| RQ3 | 云端沙箱与本地沙箱分别有哪些隔离原语可选？启动时延、隔离强度、成本与合规边界如何权衡？ | E2B、Daytona、Modal、自建 Firecracker / 容器；macOS Seatbelt、Windows 受限令牌 / AppContainer、Linux Landlock / seccomp | 给出两类沙箱的推荐组合与退出通道 | §3.1.2 中 T8、T9 行 + §4.3 对应行 | S6、S7 |
| RQ4 | 长程任务编排（Goal-Milestone-Task 三层分解 + Critic 自检 + 重规划 + 检查点与恢复）在业界有哪些已验证的实现范式？ | OpenHands EventStream、LangGraph Checkpoint / Time Travel、Claude Agent SDK 的 Task / Subagent 与 Hooks、VS Code Agent HQ 的 Git worktree 隔离 | 抽象出可在 TS 中落地的编排骨架 | §3.1.2 中 T11 行 + §4.3 对应行 + §5.3 下游依赖 | S9 |
| RQ5 | 本地与云端部署 / 测试 / 预览服务，以及消息通道接入，有哪些可复用路径与成本量级？ | Vercel / Netlify 式 preview environment、自建 ephemeral preview（泛域名 + 反代 + 自动回收）、Webhook / IM Bot / MQ 适配器 | 给出双轨预览与统一通道适配器的建议 | §3.1.2 中 T10、T12 行 + §4.3 对应行 | S8、S10 |

---

## 2. 事实：标杆系统盘点和方案详述

> **四段式「事实」段**。只陈列调研发现的事实，不做引申建议或边界裁决。

### 2.1 行业标杆清单

> 完整盘点调研覆盖的所有标杆系统，给出标签化画像。

**硬指标**：≥ 3 家；至少包含 1 家头部 SaaS 代表 + 1 家开源/自研代表。本轮实际覆盖 11 家，其中头部 SaaS 代表 5 家（B3、B7、B8、B9、B11），开源 / 自研代表 5 家（B1、B2、B4、B5、B6、B10）。

| 编号 | 标杆系统 | 厂商 / 社区 | 部署形态 | 场景覆盖 | 技术亮点 | 商业模式 | 调研来源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | Claude Code + Claude Agent SDK | Anthropic | CLI + 库（Python / TypeScript）；本地进程直连 Anthropic API | 终端原生编码 Agent；SDK 可嵌入任意应用 | 内置 Read/Write/Edit/Bash/Glob/Grep/WebSearch/WebFetch/AskUserQuestion/Agent 工具；Hooks 事件体系；Subagents；MCP client + in-process MCP server；Permission modes | 商业（API 用量计费 + 订阅） | SR-01、SR-02、SR-03 |
| B2 | OpenAI Codex CLI + OpenAI Agents SDK | OpenAI | CLI（Rust 实现）+ Python 优先 SDK；本地或 CI | 终端编码 Agent；多 Agent handoff | approval_policy × sandbox_mode 双轴权限；macOS Seatbelt / Linux Landlock+seccomp / Windows 受限令牌；AGENTS.md 分层记忆 | 商业（API 用量计费） | SR-04、SR-05、SR-06 |
| B3 | Cursor | Anysphere | 桌面 IDE（VS Code fork）+ 云端后台 Agent | AI 原生编辑器；并行 Agent；后台 / 云端 Agent | 最多 8 个并行 Agent；Git worktree 隔离；Background Agents 跑在隔离 Ubuntu VM；Composer 自研模型；MCP 支持 stdio 与 SSE（有 40 工具上限） | SaaS 订阅（20 美元/月起）+ Max Mode | SR-07、SR-08、SR-09 |
| B4 | Cline | Cline（开源社区） | VS Code 扩展 + JetBrains 插件 + CLI + Kanban + SDK | 人工在环的编码 Agent | 每步操作显式审批；MCP Marketplace；`@cline/sdk` Agent Core 可复用；支持 ACP 模式接入 Zed / Neovim | 开源免费 + Teams 订阅 + BYOK | SR-10、SR-11 |
| B5 | Roo Code | Roo Code（开源社区） | VS Code 扩展 | 多模式自治编码 Agent | 5 种模式（Code / Architect / Ask / Debug / Custom）；Cloud Agents（GitHub / Web 触发）；Slack / Linear 集成（Team 版） | 开源免费 + Cloud 按小时计费（5 美元/小时）+ Team 99 美元/月 | SR-12、SR-13 |
| B6 | OpenHands（原 OpenDevin） | OpenHands 社区（All Hands AI） | Web + CLI；本地 Docker 或云端 | 通用软件工程 Agent 平台 | EventStream 事件流为单一事实来源；Runtime 抽象（Local / Docker / E2B）；CodeActAgent；多 Agent 委派；SWE-bench Verified 领先 | Apache-2.0 开源 + 云订阅；2025-11 获 1885 万美元 A 轮 | SR-14、SR-15、SR-16、SR-17 |
| B7 | GitHub Copilot Agent Mode / VS Code Agent HQ | Microsoft / GitHub | VS Code 1.107+（2025-12-10 发布）内置 | 多 Agent 编排中枢 | Agent HQ 统一管理本地 / 后台 / 云端 Agent；后台 Agent 跑在独立 Git worktree；可复用 Claude Skills；组织级自定义 Agent | 商业（Copilot 订阅） | SR-18、SR-19 |
| B8 | Windsurf | Cognition（2025 年 7 月收购） | 桌面 IDE（VS Code fork） | 最大化自动化程度的 Agent IDE | Cascade Agent；深度代码库索引；自然语言驱动终端 | SaaS 订阅（15 美元/月） | SR-20、SR-21 |
| B9 | Devin | Cognition | 云端 SaaS（Brain 在 Cognition 云 + Devbox） | 工单级异步自治软件工程师 | Brain / Devbox 两段架构；沙箱含 shell + 浏览器 + 编辑器；Slack / Jira / Linear / GitHub 集成；DeepWiki 代码库索引；ACU 计费 | 按 ACU 计费（约 2.25 美元/ACU，1 ACU 约 15 分钟）+ 团队计划 | SR-22、SR-23、SR-24 |
| B10 | Zed | Zed Industries | 桌面编辑器（macOS / Linux / Windows） | 高性能原生编辑器 + Agentic 编辑 | Rust + 自研 GPUI（GPU 光栅化）；CRDT 多人协同；并行运行 Agent；可编辑 diff 审批；支持 Ollama 本地模型 | 编辑器开源免费（GPL / Apache），AI 服务按用量 | SR-25、SR-26 |
| B11 | 腾讯 CodeBuddy / WorkBuddy | 腾讯云 | 插件 + IDE + CLI 三形态；WorkBuddy 为桌面 Agent | 编码与办公双场景智能体 | 与 CodeBuddy 同一套 Agent 智能体架构；内置 20+ Skills 技能包与 MCP；多窗口多 Agent 并行；安全网关代理（规避内网穿透）；支持企微 / QQ / 飞书 / 钉钉接入 | 商业（腾讯云账号与计费体系） | SR-27、SR-28、SR-29 |

### 2.2 标杆方案详述

> 每家标杆逐一展开；每段区分「已核实的事实」与「推断 / 假设」。

#### 2.2.1 B1 - Claude Code 与 Claude Agent SDK（Anthropic）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | Claude Code 是 Anthropic 的终端原生编码 Agent；Claude Agent SDK 是把 Claude Code 的完整能力（工具、权限、Hooks、Subagents、MCP）开放为编程接口的库 | 已核实（SR-01、SR-02） |
| 目标用户 | 需要在自有应用中嵌入「能读仓库、跑命令、验证结果」的 Agent 的工程团队 | 已核实（SR-02） |
| 核心能力 | 内置工具集（Read / Write / Edit / Bash / Glob / Grep / WebSearch / WebFetch / AskUserQuestion / Agent）；`tool()` + `createSdkMcpServer()` 定义进程内 MCP 工具；`mcpServers` 接入外部 MCP；Hooks 事件（PreToolUse / PostToolUse / PostToolUseFailure / SubagentStart / SubagentStop / SessionStart / SessionEnd / Stop / PreCompact / PermissionRequest 等，TypeScript 版比 Python 版多出 SessionStart、SessionEnd、TeammateIdle、TaskCompleted 等事件）；permissionMode 五档（default / plan / acceptEdits / dontAsk / bypassPermissions） | 已核实（SR-01、SR-02、SR-03） |
| 架构特点 | query() 返回异步消息流；Hooks 以 matcher 过滤工具名，回调可返回 deny / allow / 修改 input / 注入 context；权限决策取最严格结果 | 已核实（SR-01） |
| 部署形态 | 库（npm 包 `@anthropic-ai/claude-agent-sdk`）+ CLI；进程运行在用户机器上，直连 Anthropic API | 已核实（SR-03） |
| 集成方式 | npm 安装；TypeScript 与 Python 双一等公民；`settingSources` 可加载 `.claude/settings.json` 与 CLAUDE.md / Skills | 已核实（SR-02、SR-03） |
| 定价模式 | 走 Anthropic API 用量计费，SDK 本身不单独收费 | 综合归纳 |
| 优势 | 与本项目能力原子重合度最高：MCP、Skill、Subagent（对应并行子 Agent 集群）、Hooks（对应权限与本地沙箱前置拦截）、SessionStart/Stop（对应检查点与恢复）均有现成范式 | 综合归纳 |
| 局限 | 仅支持 Claude 模型，不支持 OpenAI 兼容协议——与用户「Anthropic + OpenAI 双协议并行」的明确要求直接冲突；Process 级权限与沙箱需宿主自行补齐 | 已核实 + 推断（依据 SR-02 的模型范围声明与 D2 §2.2 双协议要求） |
| 对本项目的参考价值 | 高。借鉴其**能力模型与协议约定**（Hooks 事件名、matcher 语义、permissionMode 分档、Skill 目录约定），不借鉴其闭源运行时与单供应商绑定 | 推断 |

#### 2.2.2 B2 - OpenAI Codex CLI 与 OpenAI Agents SDK

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | Codex CLI 是 OpenAI 的终端编码 Agent（Rust 实现）；Agents SDK 是以 agents / handoffs / guardrails 三概念为核心的轻量多 Agent 框架 | 已核实（SR-04、SR-06） |
| 目标用户 | 需要细粒度权限分级的工程团队；需要在 CI 中无人值守运行的场景 | 已核实（SR-04） |
| 核心能力 | approval_policy（untrusted / on-request / on-failure / never）× sandbox_mode（read-only / workspace-write / danger-full-access）双轴；profile 预设；AGENTS.md 三级合并（全局 / 仓库根 / 当前目录）；`mcp_servers` TOML 配置；`codex mcp` 可反向作为 MCP server 暴露 | 已核实（SR-04、SR-05、SR-06） |
| 架构特点 | **审批策略与沙箱策略正交可配**，是本品类中最清晰的权限模型；沙箱落地到 OS 原语而非容器 | 已核实（SR-04、SR-05） |
| 部署形态 | CLI（macOS 12+ / Ubuntu 20.04+ / Debian 10+ / Windows 11 经 WSL2）；Windows 原生沙箱为实验特性 | 已核实（SR-04） |
| 集成方式 | npm 安装 `@openai/codex`；Rust 二进制；SDK 以 Python 为主，TypeScript 版覆盖核心能力但对新特性滞后 | 已核实（SR-05、SR-06） |
| 定价模式 | 走 OpenAI API 用量计费 | 综合归纳 |
| 优势 | 跨平台本地沙箱的原语级实现（macOS Seatbelt + sandbox-exec；Linux bwrap + seccomp；Windows 受限令牌 + AppContainer + capability SID，并覆写代理环境变量禁网）是本项目 T9 的直接可借鉴蓝本 | 已核实（SR-04） |
| 局限 | Windows 原生沙箱仍为实验特性，且其已知限制是「无法阻止对 Everyone SID 具备写权限的目录（world-writable）的写入 / 删除 / 创建」；Linux 沙箱在 Codex 0.115 后迁至 bwrap，WSL1 不再支持；TypeScript SDK 能力滞后于 Python | 已核实（SR-04、SR-06） |
| 对本项目的参考价值 | 高（沙箱与权限维度）。借鉴其**双轴权限模型与 OS 原语映射表**；其 TS SDK 不作为本项目主 SDK | 推断 |

#### 2.2.3 B3 - Cursor（Anysphere）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | fork 自 VS Code 的 AI 原生桌面 IDE，把 LLM 嵌入每一个编辑流程 | 已核实（SR-07、SR-08） |
| 目标用户 | 追求极致编辑速度 + 并行 Agent 工作流的开发者与团队 | 已核实（SR-07） |
| 核心能力 | Tab 补全（Fusion 本地稀疏模型）；Composer / Agent 多文件任务；Background Agents（2025-05 上线）跑在隔离 Ubuntu VM，可从 IDE / 浏览器 / 手机 / Slack 启动并自动开 PR；最多 8 个并行 Agent，用 Git worktree 隔离；MCP 支持 stdio 与 SSE | 已核实（SR-07、SR-08、SR-09） |
| 架构特点 | 本地 Tab 模型处理行内编辑，重型任务路由到 OpenAI / Anthropic / Google / xAI；语义 diff + 二级模型自愈语法错误 | 综合归纳（SR-08、SR-09） |
| 部署形态 | 桌面客户端（VS Code fork）+ 云端后台 Agent（AWS 基础设施） | 已核实（SR-09） |
| 集成方式 | 自带设置面板；MCP 在 Settings → Features → MCP 添加；Cursor Rules / AGENTS.md | 已核实（SR-07） |
| 定价模式 | 订阅制（20 美元/月起）+ Max Mode；后台 Agent 另计 | 已核实（SR-07） |
| 优势 | 形态与本项目的目标形态最接近：**一体化桌面客户端 + 内嵌编辑器 + 本地 Agent + 云端并行 Agent**，且已验证 Git worktree 是并行 Agent 隔离的可行解 | 已核实（SR-07、SR-09） |
| 局限 | 闭源，内核不可借鉴；MCP 仅支持 tools 不支持 resources，且有 40 工具上限；SSH 场景下 MCP 可能不可用 | 已核实（SR-07） |
| 对本项目的参考价值 | 中高。借鉴其**并行 Agent 的 worktree 隔离范式**与「本地交互 + 云端委派」的双层分工；不借鉴其闭源内核 | 推断 |

#### 2.2.4 B4 - Cline（开源社区）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | 居住在编辑器与终端里的 AI 编码 Agent，**每一步操作都需要用户显式批准** | 已核实（SR-10） |
| 目标用户 | 对安全与可控性要求高、希望「人在环」的开发者与受监管行业团队 | 已核实（SR-10、SR-11） |
| 核心能力 | 读写文件、执行命令、驱动无头浏览器；diff 视图可编辑 / 可回滚；监听 linter / compiler 错误自动修复；MCP Marketplace（可一键发现安装 MCP server）；文件 Timeline 记录便于回溯 | 已核实（SR-10、SR-11） |
| 架构特点 | **Agent Core 与宿主解耦**：同一套核心引擎之上派生 CLI、Kanban（并行多 Agent + per-card worktree + 依赖链）、VS Code 扩展、JetBrains 插件四种应用；SDK 以 `@cline/sdk` 发布 | 已核实（SR-10） |
| 部署形态 | VS Code 扩展 / JetBrains 插件 / CLI（含 CI 无头模式）/ Web Kanban | 已核实（SR-10） |
| 集成方式 | 支持 VS Code、Cursor、Windsurf、JetBrains 全家、Antigravity，以及经 ACP 模式的 Zed 与 Neovim；模型侧 BYOK 或官方计费 | 已核实（SR-10） |
| 定价模式 | 扩展开源免费；Teams 计划（2026 年 Q1 前免费，其后 20 美元/用户/月，前 10 席位永久免费）；ClinePass 9.99 美元/月 | 已核实（SR-10、SR-12） |
| 优势 | 「Agent Core 一套、宿主多端」的分层方式，是本项目「桌面客户端 + CLI + 云端」多形态的可借鉴组织方式；审批在环的 UX 对本地系统权限设计有直接参考价值 | 综合归纳 |
| 局限 | 本地为主，云端能力弱于 Roo Code；无内置云端沙箱 | 已核实（SR-12） |
| 对本项目的参考价值 | 中。借鉴其 **Agent Core / 宿主解耦**与**审批 UX**；其 VS Code 扩展形态与本项目「内嵌 IDE」目标不同，仅作降级路径参考 | 推断 |

#### 2.2.5 B5 - Roo Code（开源社区）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | VS Code 中的自治多模式编码 Agent，定位「编辑器里的虚拟开发团队」 | 已核实（SR-12、SR-13） |
| 目标用户 | 需要按任务类型切换 Agent 人格与工具权限的开发者 | 已核实（SR-13） |
| 核心能力 | 5 种模式（Code / Architect / Ask / Debug / Custom），每种模式可配置独立 system prompt、工具权限与模型；diff 式精确编辑；上下文压缩；逐会话 token 统计；Cloud Agents 可从 GitHub / Web 触发；Slack / Linear 集成（Team 版） | 已核实（SR-12、SR-13） |
| 架构特点 | **模式即 Agent 配置档**——把「Agent 人格」降维成一份可版本化的配置，这是低成本实现「并行子 Agent 集群」的思路 | 推断（依据 SR-13 的模式描述） |
| 部署形态 | VS Code 扩展（本地）+ Cloud Agents（云端） | 已核实（SR-12） |
| 集成方式 | Open VSX / VS Code Marketplace；BYOK（Anthropic / OpenRouter / DeepSeek / 本地 Ollama）；MCP 手动配置，无市场 | 已核实（SR-12、SR-13） |
| 定价模式 | 扩展开源免费；Cloud Agents 5 美元/小时；Cloud Team 99 美元/月（不限席位） | 已核实（SR-12） |
| 优势 | 模式化配置与 Cloud Agents 的「本地 / 云端同构」设计，与本项目「本地工作区 + 云端工作区」双工作区诉求呼应 | 综合归纳 |
| 局限 | 仅 VS Code；MCP 无市场，需手工配置 | 已核实（SR-12） |
| 对本项目的参考价值 | 中。借鉴其**模式化 Agent 配置**与**本地 / 云端同构的任务派发** | 推断 |

#### 2.2.6 B6 - OpenHands（原 OpenDevin）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | 面向「通用软件开发者 Agent」的开源平台，同时是研究基准的统一评测框架 | 已核实（SR-14、SR-16） |
| 目标用户 | 需要自建 Agent 平台、或需要可审计可复现执行轨迹的团队 | 已核实（SR-14） |
| 核心能力 | EventStream 事件流（Action / Observation 全量按序落盘，是状态的单一事实来源）；Runtime 抽象（LocalRuntime / DockerRuntime / E2BRuntime）；动作封闭词表（CmdRunAction / FileEditAction / FileReadAction / IPythonRunCellAction / BrowseURLAction / AgentFinishAction）；Context Condensation 做上下文压缩；多 Agent 委派 | 已核实（SR-14、SR-15、SR-17） |
| 架构特点 | Agent 发射 Action → Runtime 在沙箱内执行 → 结果包装为 Observation → 追加进 EventStream → 循环。Agent 上下文每轮由事件流重建 | 已核实（SR-14、SR-15） |
| 部署形态 | 本地 Docker 沙箱 或 云端 Runtime；Web 前端 + CLI | 已核实（SR-14） |
| 集成方式 | Python 为主；LLM 层经 LiteLLM 对接任意 completion 模型；Runtime 可插拔 | 已核实（SR-14、SR-17） |
| 定价模式 | Apache-2.0 开源可自托管；云端为订阅制 | 已核实（SR-16） |
| 优势 | **EventStream 是本轮调研中与本项目 UI 规格（Screen A 的「实时执行流」「长程任务执行图谱」「检查点与恢复」）契合度最高的架构范式**；Apache-2.0 许可证最宽松，可完全借鉴甚至复用设计 | 已核实 + 综合归纳（SR-14、SR-15 与 D3 §3.2） |
| 局限 | 长任务事件流过长会撑爆上下文（40 步调试可超 100K tokens），依赖 condensation 缓解；Python 实现，TS 项目需自行移植 | 已核实（SR-15） |
| 对本项目的参考价值 | **最高**。EventStream + 封闭动作词表 + Runtime 抽象三层可直接作为本项目长程任务编排与检查点恢复的骨架 | 推断 |

#### 2.2.7 B7 - GitHub Copilot Agent Mode / VS Code Agent HQ（Microsoft）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | VS Code 1.107（2025-12-10 发布）起的多 Agent 编排中枢，统一管理 Copilot 与自定义 Agent | 已核实（SR-18、SR-19） |
| 目标用户 | 已在使用 VS Code 与 GitHub 的开发者与组织 | 已核实（SR-18） |
| 核心能力 | Agent sessions 集成进 Chat 视图；本地 session 关闭后 Agent 继续后台运行；Continue in 可把本地会话移交后台 / 云端 Agent；**后台 Agent 跑在专属 Git worktree 中做隔离**；可附加 selection / problem / symbol / search / git commit 作为上下文；组织级自定义 Agent；**可复用 Claude Skills** | 已核实（SR-18、SR-19） |
| 架构特点 | 本地 / 后台 / 云端三类 Agent 共用一个会话视图与上下文传递协议；隔离靠 Git worktree 而非容器 | 已核实（SR-18） |
| 部署形态 | VS Code 桌面端（含云端 Agent 会话） | 已核实（SR-18） |
| 集成方式 | 设置项驱动（chat.viewSessions.enabled 等）；MCP；自定义 Agent 文件 | 已核实（SR-18） |
| 定价模式 | Copilot 订阅（商业） | 综合归纳 |
| 优势 | 证明了「Agent 会话」应作为一等公民实体建模（有状态、可归档、可移交、可跨环境复用），这直接对应本项目 Screen A 的「并行子 Agent 集群」与「检查点与恢复」 | 综合归纳 |
| 局限 | 强绑定 VS Code 与 GitHub 生态；worktree 隔离仅覆盖文件变更，不含进程 / 网络隔离 | 已核实 + 推断 |
| 对本项目的参考价值 | 中高。借鉴其 **Agent Session 一等公民建模 + worktree 隔离 + 跨环境移交**；不借鉴其生态绑定 | 推断 |

#### 2.2.8 B8 - Windsurf（Cognition）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | 以 Cascade Agent 为核心的 VS Code fork IDE，主打最大化自动化 | 已核实（SR-20、SR-21） |
| 目标用户 | 愿意调试非自己编写的代码、追求自动化上限的独立开发者 | 已核实（SR-21） |
| 核心能力 | Cascade 多文件编辑与自主任务执行；大型单体仓库的深度索引；自然语言驱动终端；MCP 集成 | 已核实（SR-20、SR-21） |
| 架构特点 | VS Code fork + 深度上下文索引，以「上下文宽度」换「架构一致性」 | 综合归纳（SR-20） |
| 部署形态 | 桌面 IDE（VS Code fork） | 已核实（SR-21） |
| 集成方式 | VS Code 扩展生态兼容；MCP | 已核实（SR-20） |
| 定价模式 | SaaS 订阅（15 美元/月起） | 已核实（SR-20） |
| 优势 | 价格低于 Cursor；深上下文索引对大仓库有效 | 已核实（SR-20） |
| 局限 | VS Code 基底带来的 UI 卡顿与内存问题被用户诟病；简单任务存在「过度取上下文」问题；2025 年 7 月被 Cognition 收购，与 Devin 产品线存在整合不确定性 | 已核实（SR-20、SR-21） |
| 对本项目的参考价值 | 中低。主要作为「VS Code fork 路线」的反面参考（性能与维护代价） | 推断 |

#### 2.2.9 B9 - Devin（Cognition）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | 工单级异步自治软件工程师：接 Jira / Linear 工单或 Slack 请求，规划、编码、测试，最后开 PR 交人评审 | 已核实（SR-22、SR-23） |
| 目标用户 | 有清晰可验证工单积压、需大规模机械迁移的工程团队 | 已核实（SR-22） |
| 核心能力 | Brain（无状态云端服务，始终在 Cognition 云）+ Devbox（执行环境，含 shell / 浏览器 / 编辑器）；长上下文记忆；Manager Devin 分发子任务的 Sub-Agent 协作；DeepWiki 代码库索引；GitHub / GitLab / Bitbucket / Slack / Teams / Jira / Linear 集成；消费 MCP server 并暴露 REST API | 已核实（SR-22、SR-23、SR-24） |
| 架构特点 | **Brain 与 Devbox 分离**：推理在供应商云，执行在沙箱。企业云为多租户隔离；客户专属部署为单租户 VPC + AWS Private Link | 已核实（SR-24） |
| 部署形态 | 云端 SaaS（Enterprise Cloud）或客户专属 VPC 部署 | 已核实（SR-24） |
| 集成方式 | Web / Slack / API；REST API；MCP 消费方 | 已核实（SR-22、SR-24） |
| 定价模式 | 按 ACU 计费，1 ACU 约 15 分钟，约 2.25 美元/ACU；Core 计划 20 美元起（含 9 ACU）；Teams 计划约 500 美元/月 | 已核实（SR-22） |
| 优势 | 消息通道集成（Slack / Jira / Linear）与「云端沙箱 + 长程记忆 + 并行委派」的组合，直接对应本项目「云端沙箱可接入消息通道」能力原子；SOC 2 Type II；不用于训练客户代码 | 已核实（SR-22、SR-23） |
| 局限 | 闭源，Brain 在供应商云不可自持；ACU 计费在开放式任务上成本不可预测；对含糊需求与绿地架构能力弱；不开源 | 已核实（SR-22） |
| 对本项目的参考价值 | 中。借鉴其 **Brain / Devbox 分离架构**与**消息通道接入方式**；否决其整体产品形态（闭源 SaaS、数据出境、计费不可控） | 推断 |

#### 2.2.10 B10 - Zed（Zed Industries）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | 用 Rust 从零构建的高性能代码编辑器，定位为 Cursor / Windsurf 的开源替代 | 已核实（SR-25、SR-26） |
| 目标用户 | 重视性能、可控性与透明度的开发者与团队 | 已核实（SR-25） |
| 核心能力 | 自研 GPUI 框架（GPU 直接光栅化整个窗口）；CRDT 多人协同（含语音与文本线程）；Agentic 编辑（多文件改动后给出可编辑 diff，用户审批）；并行运行多个 Agent；Agent Extensions 允许第三方 Agent 接入；支持 Ollama 本地模型 | 已核实（SR-25、SR-26） |
| 架构特点 | 拒绝 Electron，以 Rust + GPUI 换启动速度与渲染流畅度；Agent 变更一律以「可编辑 diff」呈现 | 已核实（SR-25、SR-26） |
| 部署形态 | 桌面编辑器（macOS / Linux / Windows） | 已核实（SR-26） |
| 集成方式 | 自研扩展体系（Wasm 方向）；Agent Extensions；MCP | 已核实（SR-25） |
| 定价模式 | 编辑器开源（GPL / Apache）；AI 服务按用量计费 | 已核实（SR-25） |
| 优势 | 证明了「非 Electron 路线」在性能上的可行性上限；可编辑 diff 审批的 UX 值得借鉴 | 已核实（SR-25） |
| 局限 | 无法原生运行 VS Code 扩展，扩展生态显著落后；自研 UI 框架意味着极高的长期维护投入 | 已核实（SR-25） |
| 对本项目的参考价值 | 中低。作为 T1「是否放弃 Electron」的反面证据：**换性能必须放弃 VS Code 扩展生态**，而本项目要求「嵌入 vscode 作为 IDE」，二者不可兼得 | 推断 |

#### 2.2.11 B11 - 腾讯 CodeBuddy / WorkBuddy（腾讯云）

| 维度 | 内容 | 置信度 |
| --- | --- | --- |
| 产品定位 | CodeBuddy 为覆盖插件 / IDE / CLI 三形态的 AI 编程工具系列；WorkBuddy 为基于同一套 Agent 智能体架构的**桌面智能体平台**（2026 年 2 月内测） | 已核实（SR-27、SR-28） |
| 目标用户 | 腾讯内部与企业客户的研发、以及 HR / 行政 / 运营 / 销售等非技术岗位 | 已核实（SR-28） |
| 核心能力 | 内置 20+ Skills 技能包与 MCP 协议，支持一键导入官方技能、自定义或从零新建；多窗口多 Agent 并行，动态任务分解；内置多款主流大模型可切换（国内版支持 GLM / DeepSeek / Kimi / MiniMax）；**通过安全网关代理规避内网穿透风险**；支持接入企业微信 / QQ / 飞书 / 钉钉 | 已核实（SR-28） |
| 架构特点 | Agent 智能体架构在 CodeBuddy（编码）与 WorkBuddy（办公）之间复用；安全网关代理替代内网穿透；统一后台、账号与计费体系，具备安全审计 | 已核实（SR-28） |
| 部署形态 | 插件 / IDE / CLI 三形态 + 桌面客户端（WorkBuddy） | 已核实（SR-28） |
| 集成方式 | CodeBuddy Agent SDK 支持在 TypeScript / JavaScript 或 Python 应用中直接调用具备工具能力的 Agent（读改文件、执行 Shell、搜代码库、MCP 交互、多轮推理）；与 CloudBase / Supabase / CloudStudio 打通 | 已核实（SR-27、SR-28） |
| 定价模式 | 腾讯云账号与计费体系（商业） | 已核实（SR-28） |
| 优势 | **与本项目目标形态最接近的国内产品**：桌面 Agent 客户端 + Skills + MCP + 多 Agent 并行 + 消息通道（IM）接入 + 安全网关代理。其「安全网关代理替代内网穿透」是本项目 T10 的直接可借鉴方案 | 已核实（SR-28） |
| 局限 | 强绑定腾讯云生态与国内模型生态；内部效能数据（90% 工程师使用、编码时间缩短 40%+、AI 生成代码占比超 50%）为厂商自述，未经第三方验证 | 已核实 + 推断（SR-28，厂商自述） |
| 对本项目的参考价值 | 中高。借鉴其 **Skills 组织方式、多 Agent 并行、IM 通道接入与安全网关代理**；是否复用其 Agent SDK 需先确认供应商绑定与双协议支持程度（见 §5.2 U-04） | 推断 |

### 2.3 关键技术能力横向事实

> 不评分、不排序，仅按能力维度横陈各方案事实。

| 能力维度 | B1 Claude Code / Agent SDK | B3 Cursor | B6 OpenHands | B9 Devin | B11 CodeBuddy / WorkBuddy | 说明 / 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| 桌面客户端形态 | 否（CLI / 库，宿主自建） | 是（VS Code fork 桌面 IDE） | 否（Web + CLI，Docker 沙箱） | 否（纯云端 Web / Slack） | 是（WorkBuddy 桌面 Agent） | SR-02、SR-07、SR-14、SR-22、SR-28 |
| 嵌入编辑器 | 否（由宿主提供） | 是（自身即编辑器） | 否（Web 前端 + 沙箱内编辑器） | 是（Devbox 内含 VS Code 风格编辑器） | 是（IDE 形态） | SR-07、SR-14、SR-23、SR-27 |
| 嵌入终端 | 是（Bash 内置工具，宿主渲染） | 是（终端执行工具） | 是（CmdRunAction，沙箱内 shell） | 是（Devbox shell） | 是（Shell 命令执行） | SR-02、SR-07、SR-15、SR-23、SR-27 |
| MCP 支持 | 是（client + 进程内 server，stdio / 外部 server） | 是（仅 tools，40 工具上限，stdio + SSE） | 未覆盖 | 是（消费 MCP server） | 是（MCP 协议内置） | SR-02、SR-07、SR-22、SR-28 |
| Skill / 技能包 | 是（Claude Skills，SKILL.md 格式发起方） | 是（Rules 机制，非 SKILL.md） | 是（AgentSkills 库，Python 函数） | 未覆盖 | 是（20+ Skills 技能包） | SR-02、SR-07、SR-16、SR-28 |
| 本地沙箱 | 部分（Hooks + permissionMode，OS 级沙箱由宿主补） | 部分（terminal 安全确认） | 部分（LocalRuntime 直跑本机，DockerRuntime 隔离） | 否（执行全在云端） | 未覆盖 | SR-02、SR-07、SR-14、SR-23 |
| 云端沙箱 | 未覆盖 | 是（Background Agents 跑隔离 Ubuntu VM） | 是（E2BRuntime / DockerRuntime） | 是（Devbox，单租户 VPC 可选） | 是（CloudStudio 沙盒环境一键部署） | SR-09、SR-14、SR-22、SR-28 |
| 消息通道 | 未覆盖 | 是（Slack @Cursor 触发后台 Agent） | 未覆盖 | 是（Slack / Teams / Jira / Linear + REST API） | 是（企微 / QQ / 飞书 / 钉钉） | SR-07、SR-22、SR-28 |
| 并行子 Agent | 是（Subagents / Agent teams，teams 为 CLI 特性） | 是（最多 8 个并行 + Git worktree 隔离） | 是（AgentDelegateAction，Spawn / Delegate） | 是（Sub-Agent + Manager Devin） | 是（多窗口多 Agent 并行，动态任务分解） | SR-02、SR-07、SR-16、SR-23、SR-28 |
| 长程任务 / 检查点 | 部分（SessionStart / SessionEnd / PreCompact Hooks，会话可恢复） | 部分（会话 + worktree 变更） | 是（EventStream 全量事件 + Condensation） | 是（长上下文记忆 + 回放时间线） | 部分（动态任务分解） | SR-02、SR-07、SR-14、SR-23、SR-28 |
| 部署与预览 | 未覆盖 | 未覆盖 | 未覆盖 | 未覆盖 | 是（CloudStudio 一键部署到沙盒环境，生成可分享链接） | SR-28 |
| 双协议 / 多供应商 | 否（仅 Anthropic 原生） | 是（Claude / GPT / Gemini / Grok 多供应商路由） | 是（LiteLLM 对接任意 completion 模型） | 未公开 | 是（GLM / DeepSeek / Kimi / MiniMax 可切换） | SR-02、SR-07、SR-14、SR-28 |
| 开源 / 可自持 | 否（SDK 开源，运行时闭源） | 否 | 是（Apache-2.0，可自托管） | 否 | 否 | SR-14、SR-16、SR-22 |

---

## 3. 对比：对比矩阵与加权评分

> **四段式「对比」段**。在 §2 的事实基础上建立对比矩阵，赋予权重并打分。

### 3.1 对比矩阵

#### 3.1.1 对标产品借鉴价值矩阵

> **每行权重之和 = 1.00**。评估对象为「该标杆对 openINTJ 的可借鉴程度」，非「该产品本身的优劣」。

| 评估维度 | 权重 | 权重理由 | B1 得分 | B3 得分 | B6 得分 | B9 得分 | B11 得分 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 场景契合度 | 0.30 | 本项目为「一体化桌面客户端 + 本地 Agent + 云端 Agent + 沙箱 + 长程编排」，形态与能力模型的重合度是首要判据 | 5 | 4 | 4 | 4 | 4 |
| 技术成熟度 | 0.20 | 绿地项目需站在已验证路径上，未经验证的架构会放大返工风险 | 5 | 5 | 4 | 4 | 4 |
| 集成难度（反向，分越高越易集成） | 0.15 | 团队为 TS 全栈，闭源 / Python 实现 / 生态绑定都会显著抬高集成成本 | 4 | 3 | 5 | 2 | 3 |
| 成本（反向，分越高成本越低） | 0.15 | 无预算输入（资料摘要 Q4），成本可控性比绝对金额更重要 | 3 | 3 | 5 | 2 | 3 |
| 合规可控性 | 0.20 | 涉及许可证可自持性与数据驻留，属不可妥协项 | 3 | 2 | 5 | 2 | 3 |
| **加权总分** | **1.00** | — | **4.15** | **3.50** | **4.50** | **3.00** | **3.50** |

**评分标尺**：每项 1~5 分，1 = 严重不符合，3 = 基本满足但存在明显局限，5 = 完美契合。

#### 3.1.2 关键技术选型论题评分矩阵（T1~T12）

> 维度与权重按主理人指定维度设定，权重之和 = 1.00。其中「工程复杂度」「许可证风险」「成本」三项为**反向计分**（分越高分别代表复杂度越低、风险越低、成本越低），以保证加权总分可直接横向比较。

| 评估维度 | 权重 | 权重理由 |
| --- | --- | --- |
| 能力覆盖度 | 0.20 | 能否覆盖该论题对应的能力原子，是选型的第一判据 |
| 工程复杂度（反向） | 0.15 | 绿地项目、TS 全栈团队，超出主栈的工程量会直接转化为工期风险 |
| 生态成熟度 | 0.15 | 决定遇到问题能否找到现成解法与社区支持 |
| 许可证风险（反向） | 0.10 | 涉及商标与再分发限制，权重不高但属一票否决型因素 |
| 跨平台一致性 | 0.15 | 用户明确要求覆盖 Windows 与 macOS，跨平台行为差异会放大测试与运维成本 |
| 长期可维护性 | 0.15 | 14 项能力原子属长周期工程，依赖的存活与可控性决定总持有成本 |
| 成本（反向） | 0.10 | 无预算输入，故按相对量级比较而非绝对值 |

| 论题 | 候选方案 | 能力覆盖 | 工程复杂度（反向） | 生态成熟度 | 许可证风险（反向） | 跨平台一致性 | 长期可维护性 | 成本（反向） | 加权总分 | 主要证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 桌面壳层 | Electron | 5 | 5 | 5 | 4 | 5 | 4 | 4 | **4.65** | SR-30、SR-31、SR-36 |
| T1 桌面壳层 | Tauri 2 | 3 | 2 | 3 | 5 | 3 | 3 | 5 | **3.25** | SR-32、SR-33、SR-36 |
| T1 桌面壳层 | Neutralinojs | 2 | 3 | 2 | 5 | 3 | 2 | 5 | **2.90** | SR-32 |
| T2 IDE 嵌入形态 | VS Code 扩展形态 | 3 | 5 | 5 | 5 | 5 | 4 | 5 | **4.45** | SR-34、SR-35 |
| T2 IDE 嵌入形态 | 内嵌 VSCodium（MIT + Open VSX） | 5 | 2 | 4 | 4 | 4 | 3 | 3 | **3.65** | SR-34、SR-35 |
| T2 IDE 嵌入形态 | 内嵌 code-server | 4 | 3 | 4 | 4 | 5 | 3 | 3 | **3.75** | SR-34 |
| T2 IDE 嵌入形态 | fork Code-OSS 定制发行版 | 5 | 1 | 3 | 2 | 4 | 2 | 2 | **2.90** | SR-34、SR-35 |
| T3 终端嵌入 | xterm.js + node-pty 自研集成 | 5 | 4 | 5 | 5 | 4 | 4 | 5 | **4.55** | SR-36、SR-37、SR-38 |
| T3 终端嵌入 | 复用 VS Code 集成终端 | 4 | 2 | 4 | 4 | 4 | 3 | 4 | **3.55** | SR-37 |
| T3 终端嵌入 | 系统终端唤起 | 2 | 5 | 3 | 5 | 2 | 3 | 5 | **3.35** | SR-36 |
| T4 Agent SDK 形态 | 自研 TS SDK（薄内核 + 自研工具循环） | 5 | 2 | 2 | 5 | 5 | 4 | 4 | **3.85** | SR-02、SR-39 |
| T4 Agent SDK 形态 | Vercel AI SDK（多供应商）+ 自研编排 | 4 | 4 | 5 | 5 | 5 | 4 | 4 | **4.40** | SR-39、SR-40 |
| T4 Agent SDK 形态 | 复用 Claude Agent SDK | 4 | 5 | 4 | 3 | 4 | 3 | 3 | **3.80** | SR-02、SR-39 |
| T4 Agent SDK 形态 | 复用 OpenAI Agents SDK | 3 | 4 | 3 | 3 | 3 | 3 | 3 | **3.15** | SR-39、SR-40 |
| T5 LLM Provider 抽象 | 自研双协议适配层（Anthropic Messages 原生 + OpenAI Chat Completions 兼容） | 5 | 3 | 4 | 5 | 5 | 5 | 5 | **4.55** | SR-41、SR-42、SR-43 |
| T5 LLM Provider 抽象 | 单一 OpenAI 兼容层 + 网关转换 | 3 | 5 | 5 | 4 | 5 | 3 | 4 | **4.10** | SR-41、SR-43 |
| T5 LLM Provider 抽象 | 完全依赖第三方网关 | 3 | 5 | 3 | 2 | 5 | 2 | 2 | **3.25** | SR-43 |
| T6 MCP 实现 | 官方 TS SDK，client + server 双角色，stdio + Streamable HTTP（保留 SSE 兼容） | 5 | 4 | 5 | 5 | 5 | 5 | 5 | **4.85** | SR-44、SR-45、SR-46 |
| T6 MCP 实现 | 仅 client 角色 + stdio | 3 | 5 | 4 | 5 | 5 | 4 | 5 | **4.30** | SR-44、SR-45 |
| T6 MCP 实现 | 自研 JSON-RPC（不兼容 MCP） | 3 | 3 | 1 | 5 | 5 | 2 | 4 | **3.15** | SR-44 |
| T7 Skill 协议 | 采用 agentskills.io 开放标准（SKILL.md + YAML frontmatter 六字段） | 5 | 5 | 5 | 5 | 5 | 5 | 5 | **5.00** | SR-47、SR-48、SR-49 |
| T7 Skill 协议 | 自定义 YAML frontmatter 目录约定 | 3 | 4 | 2 | 5 | 5 | 2 | 4 | **3.45** | SR-47 |
| T7 Skill 协议 | 仅采用 Claude Skills 私有格式 | 3 | 4 | 3 | 3 | 4 | 2 | 4 | **3.25** | SR-48 |
| T8 云端沙箱 | E2B（Firecracker microVM，Apache-2.0 可自托管） | 5 | 4 | 5 | 5 | 4 | 4 | 3 | **4.35** | SR-50、SR-51、SR-52 |
| T8 云端沙箱 | Daytona（容器优先，sub-90ms） | 4 | 4 | 4 | 2 | 4 | 3 | 4 | **3.65** | SR-50、SR-51、SR-52 |
| T8 云端沙箱 | Modal（gVisor + GPU） | 4 | 4 | 4 | 3 | 4 | 4 | 3 | **3.80** | SR-51、SR-52 |
| T8 云端沙箱 | 自建 Firecracker / microVM 或容器 | 5 | 1 | 3 | 5 | 3 | 4 | 3 | **3.45** | SR-50、SR-51 |
| T9 本地沙箱与权限 | 平台原生原语分级 + 统一策略层（Seatbelt / 受限令牌 + AppContainer + Job Object / Landlock + seccomp） | 5 | 3 | 4 | 5 | 3 | 4 | 5 | **4.10** | SR-04、SR-05、SR-52 |
| T9 本地沙箱与权限 | WSL2 / 容器统一封装 | 4 | 3 | 4 | 4 | 3 | 3 | 4 | **3.55** | SR-04、SR-52 |
| T9 本地沙箱与权限 | 仅进程白名单 + 审批（不做 OS 隔离） | 2 | 5 | 3 | 5 | 5 | 2 | 5 | **3.65** | SR-02、SR-04 |
| T10 消息通道接入 | 统一 Channel Adapter（Webhook + IM Bot + MQ 三适配器，事件归一化） | 5 | 4 | 4 | 5 | 5 | 5 | 5 | **4.70** | SR-22、SR-28、SR-53 |
| T10 消息通道接入 | 仅 Webhook | 2 | 5 | 4 | 5 | 5 | 4 | 5 | **4.10** | SR-53 |
| T10 消息通道接入 | 仅单一 IM Bot | 3 | 4 | 3 | 4 | 4 | 3 | 4 | **3.50** | SR-28 |
| T11 长程任务编排 | 自研 EventStream + Checkpoint（借鉴 OpenHands 与 LangGraph，TS 实现） | 5 | 2 | 3 | 5 | 5 | 4 | 4 | **4.00** | SR-14、SR-15、SR-54、SR-55 |
| T11 长程任务编排 | 复用 LangGraph（JS 版）+ Postgres Checkpoint | 4 | 3 | 4 | 4 | 4 | 4 | 4 | **3.85** | SR-54、SR-55 |
| T11 长程任务编排 | 复用 Claude Agent SDK 的 Task / Subagent | 3 | 4 | 4 | 3 | 4 | 3 | 3 | **3.45** | SR-02 |
| T11 长程任务编排 | 自研 Goal-Milestone-Task + Critic，不带检查点 | 4 | 4 | 2 | 5 | 5 | 3 | 4 | **3.80** | SR-14、SR-54 |
| T12 部署与预览 | 本地 preview server + 云端托管预览双轨 | 5 | 3 | 5 | 4 | 5 | 4 | 3 | **4.25** | SR-56、SR-57、SR-58 |
| T12 部署与预览 | 仅本地 preview server | 3 | 5 | 4 | 5 | 5 | 4 | 5 | **4.30** | SR-56 |
| T12 部署与预览 | 自建 ephemeral preview（泛域名 + 反代 + 自动回收） | 4 | 2 | 3 | 5 | 4 | 3 | 4 | **3.50** | SR-57、SR-58 |

> **矩阵裁决状态说明（2026-10-20 更新）**：上表中 **T1、T2、T8 三个论题已经过中间确认并由用户裁决**（见 §0.1）——T1 定为 Electron（4.65）、T2 定为内嵌 VSCodium + Open VSX（3.65）、T8 定为自建（3.45）。这三个论题**所有候选方案的分值（含裁决项本身）一律保留不动**——评分是对方案本身的事实记录，不因用户裁决而修改；其中裁决项以外的候选分值仅作为已记录的评估证据保留，不再参与后续选型。各论题的最终裁决结果：T1 = Electron（4.65）、T2 = 内嵌 VSCodium + Open VSX（3.65）、T8 = 自建（3.45，云厂商腾讯云）。其对应的推荐结论见 §3.2 与 §4.3 中标注「经中间确认，已冻结」的行。需要特别指出：T2 与 T8 的最终裁决均**不取矩阵最高分**（T2 最高分为 VS Code 扩展形态 4.45、T8 最高分为 E2B 4.35），原因是打分维度无法覆盖「产品形态承诺」与「数据自持」这两项用户侧约束——这正说明加权打分是评估而非授权。

### 3.2 评分结论

> 基于 §3.1 加权总分，形成分层结论。每层结论引用得分作为依据。

**对标产品（§3.1.1）**

- **优先借鉴**：`B6 OpenHands` — 适用度评分：**4.50**。理由：场景契合度 4/5（EventStream + Runtime 抽象 + 多 Agent 委派，正好覆盖长程编排与沙箱两项核心能力）、集成难度 5/5（Apache-2.0 开源、架构文档完整，可完全借鉴甚至移植）、合规可控性 5/5（可自托管，无供应商绑定）、成本 5/5（无许可费用）。是本轮唯一在「合规可控性」上得满分的标杆。
- **优先借鉴**：`B1 Claude Code / Claude Agent SDK` — 适用度评分：**4.15**。理由：场景契合度 5/5（其 Hooks 事件体系、Subagents、permissionMode 分档、Skill 目录约定与本项目 14 项能力原子重合度最高）、技术成熟度 5/5。**限定借鉴范围**：只借鉴其能力模型与协议约定，不借鉴其闭源运行时与单供应商绑定（合规可控性仅 3/5、成本仅 3/5）。
- **部分借鉴**：`B3 Cursor` — 借鉴点：**并行 Agent 的 Git worktree 隔离范式**、**本地交互 + 云端委派的双层分工**、**语义 diff 自愈**。不借鉴的部分：闭源编辑器内核、VS Code fork 的维护体系。理由：形态最接近（场景契合度 4/5、技术成熟度 5/5），但合规可控性仅 2/5、集成难度 3/5，内核不可复用。评分 3.50。
- **部分借鉴**：`B11 CodeBuddy / WorkBuddy` — 借鉴点：**Skills 技能包组织方式**、**多窗口多 Agent 并行**、**IM 通道接入**、**安全网关代理替代内网穿透**。不借鉴的部分：腾讯云生态绑定、国内模型生态锁定。理由：形态贴近（场景契合度 4/5），但生态绑定使长期可维护性与合规可控性均受限（各 3/5）。评分 3.50。
- **不借鉴（否决）**：`B9 Devin` — 否决理由：**Brain 位于供应商云不可自持（合规可控性 2/5）**、**ACU 计费在开放式任务上不可预测（成本 2/5）**、**闭源且不可自托管（集成难度 2/5）**——三项同时触底，与本项目「一体化桌面客户端 + 数据自持」的定位冲突。评分 3.00。**仅保留其「Brain / Devbox 分离」这一架构思想的参考价值，不采纳其产品形态与供给方式。**

**关键技术选型（§3.1.2）各论题的首位候选（按加权总分）**

| 论题 | 首位候选 | 加权总分 | 次位候选 | 差值 | 结论形态 |
| --- | --- | --- | --- | --- | --- |
| T1 桌面壳层 | Electron（**实际裁决项 = Electron**，与打分首位一致） | 4.65 | Tauri 2（3.25） | 1.40 | **已冻结（经中间确认，用户裁决 2026-10-20）** — IC-01；Tauri 2 / Neutralinojs 退出候选 |
| T2 IDE 嵌入形态 | VS Code 扩展形态（**实际裁决项 = 内嵌 VSCodium + Open VSX（3.65），非打分首位**） | 4.45 | 内嵌 code-server（3.75） | 0.70 | **已冻结（经中间确认，用户裁决 2026-10-20）** — IC-02；不取矩阵最高分；扩展形态保留为降级路径 |
| T3 终端嵌入 | xterm.js + node-pty | 4.55 | 复用 VS Code 集成终端（3.55） | 1.00 | 建议 |
| T4 Agent SDK 形态 | Vercel AI SDK + 自研编排 | 4.40 | 自研 TS SDK（3.85） | 0.55 | 建议（差值小，需以抽象层兜底，见 §5.1 R-05） |
| T5 LLM Provider 抽象 | 自研双协议适配层 | 4.55 | 单一兼容层 + 网关（4.10） | 0.45 | 建议（用户已明确双协议，此项基本锁定） |
| T6 MCP 实现 | 官方 TS SDK，双角色 + stdio + Streamable HTTP | 4.85 | 仅 client + stdio（4.30） | 0.55 | 建议 |
| T7 Skill 协议 | agentskills.io 开放标准 | 5.00 | 自定义 YAML 约定（3.45） | 1.55 | 建议（差值最大，业界已事实上统一） |
| T8 云端沙箱 | E2B（**实际裁决项 = 自建（3.45），云厂商 = 腾讯云，非打分首位**） | 4.35 | Modal（3.80） | 0.55 | **已冻结（经中间确认，用户裁决 2026-10-20）** — IC-03 + IC-04；不取矩阵最高分；E2B 降为技术参考实现，Daytona 不参考 |
| T9 本地沙箱与权限 | 平台原生原语分级 + 统一策略层 | 4.10 | 仅进程白名单（3.65） | 0.45 | 建议 |
| T10 消息通道接入 | 统一 Channel Adapter | 4.70 | 仅 Webhook（4.10） | 0.60 | 建议 |
| T11 长程任务编排 | 自研 EventStream + Checkpoint | 4.00 | LangGraph（3.85） | 0.15 | 建议（差值极小，见 §3.3 组合分析） |
| T12 部署与预览 | 仅本地 preview server | 4.30 | 本地 + 云端双轨（4.25） | 0.05 | 建议（差值几乎为零，见 §3.3 组合分析） |

> **重要声明**：上表中 **T1、T2、T8 三行已由用户裁决并冻结**（见 §0.1），是「经中间确认」决策，下游可直接消费；**其余九行（T3、T4、T5、T6、T7、T9、T10、T11、T12）仍为建议**，需 business-architect 在 G3 阶段结合自身边界判断后确认，不得直接视为既定技术栈。加权打分最优不等于已冻结选型——T2 与 T8 的最终裁决均未取矩阵最高分，即为明证。

### 3.3 方案组合分析

> 调研发现：单一方案无法覆盖全部需求，需要组合。

| 组合方式 | 覆盖哪些能力 | 未覆盖能力 | 组合复杂度 | 总体成本估算 |
| --- | --- | --- | --- | --- |
| A：B1 能力模型（协议与约定）+ B6 架构骨架（EventStream / Runtime 抽象）+ B3 隔离范式（worktree）+ B11 通道与安全网关 | 能力原子 1、5、6、7、9、11、13；长程编排；并行子 Agent；消息通道 | 能力原子 3（IDE 嵌入）、4（终端嵌入）、10、12（本地沙箱与 OS 权限）——这四项无标杆可直接覆盖，必须自研或组合第三方组件 | 中（协议层统一，但需自建 TS 实现） | 中（无许可费，主要为自研人力） |
| B：组合 A + **自建**云端沙箱（腾讯云，经中间确认 IC-03 / IC-04） | 追加覆盖能力原子 8、9、14（云端工作区 / 云端沙箱 / 云端部署预览） | 需自建镜像 / 快照 / 预热池 / 回收 / 网络策略六项能力（见 §4.4）；隔离原语取决于 CVM 是否透传 /dev/kvm（待核实，见 §5.2 U-05） | **高**（原第三方方案为低，改自建后升至高） | 中高（人力为主 + 云资源费用；需预算基线） |
| C：组合 B + 平台原生本地沙箱 | 追加覆盖能力原子 10、12（本地 shell/CLI 调用、本地沙箱安全运行环境） | Windows 原生沙箱能力弱于 macOS（已核实：无法阻止 world-writable 目录写入） | 高（三套 OS 原语 + 统一策略层） | 中高（人力为主，无许可费） |
| D：组合 C + 双轨预览（本地 preview server + 云端 ephemeral preview） | 追加覆盖能力原子 13、14 的「预览」部分 | 云端预览依赖组合 B 的账号基线（资料摘要 Q4 部分补齐：厂商已定为腾讯云，账号 / Region / 配额 / 预算仍缺） | 中 | 中（云端按资源计费） |

**组合结论（建议，2026-10-20 依 IC-03 / IC-04 修订）**：组合 B 已改写为**自建于腾讯云**（CVM / COS / VPC / CAM），其组合复杂度由「低」重估为「**高**」、成本估算由「低」重估为「**中高**」（人力为主 + 云资源费用，需预算基线）——这直接改变了 MVP 的工期假设，business-architect 在 G3 冻结边界时必须把 §4.4 的六项自建能力纳入工作量评估。

**「以组合 C 为 MVP 目标形态」是否仍成立的核对结论（2026-10-20）：需下修，不成立**。核对依据：组合 C = 组合 B（含云端自建）+ 平台原生本地沙箱；而云端自建在 Q4 剩余四项补齐前无法完成容量与成本设计（§4.2 已判定能力原子 8 / 9 / 14 不进 MVP）。故修订为：**MVP 实际落地形态 = 组合 A + 平台原生本地沙箱（即组合 C 去掉「云端自建」这一增量）；组合 C 仍作为架构演进的目标形态；组合 D 为最终演进目标**。该修订与 §4.2 的判定一致，仍属建议，最终 MVP 边界由 business-architect 在 G3 冻结。

**补充结论**：由于自建沙箱在 Q4 完全补齐前无法完成容量与成本设计，建议 MVP 阶段采用「**本地沙箱为主、云端自建为 Phase 5 交付项**」的两阶段落地顺序。该顺序同时满足 IC-03 中用户已知悉并接受的前提（「Q4 未解则无法设计」）。

---

## 4. 建议：取舍决策支持

> **四段式「建议」段**。基于 §2 事实 + §3 对比，给出可被 `business-architect` 直接采用的建议。本节是建议而非最终裁决，最终边界由业务架构师冻结。

### 4.1 自研 / 采购 / 复用边界建议

> 逐项对齐 material_digest.md D2 §2.4 的 14 项能力原子。

| 能力原子编号 | 能力项 | 建议方式 | 建议依据 | 候选方案 / 系统 | 关键前提 |
| --- | --- | --- | --- | --- | --- |
| 1 | Agent SDK | 复用 + 自研（混合） | Vercel AI SDK 在生态成熟度 5/5、跨平台一致性 5/5 上领先（4.40），但工具循环、权限、Hooks 需自研；Claude Agent SDK 因仅支持单供应商被排除 | Vercel AI SDK 作为模型调用层 + 自研编排层 | 必须加一层内部 AgentRuntime 抽象，避免被任一 SDK 绑定（见 §5.1 R-05） |
| 2 | TS 全栈桌面 AI Agent 客户端 | 自研（壳层复用成熟框架） | 无标杆可直接复用；Electron 在能力覆盖 5/5、跨平台一致性 5/5 上领先（4.65）。**经中间确认 IC-01 已冻结为 Electron** | Electron（**经中间确认，2026-10-20**） | 需接受 80–150MB 包体与 150–300MB 空闲内存（用户已明确接受）；需手动配置 contextIsolation 与 CSP；包体与内存的优化方向见 §5.1 R-09 |
| 3 | 嵌入 VSCode 作为 IDE | 复用（组件）+ 自研（集成） | 内嵌 VSCodium 是唯一同时满足「一体化内嵌」与「MIT 可再分发」的形态。**经中间确认 IC-02 已冻结为该形态**（该形态打分 3.65 低于扩展形态 4.45，但扩展形态无法达成一体化客户端承诺） | VSCodium 二进制（Code-OSS 的 MIT 构建，遥测已移除）+ Open VSX 扩展源（**经中间确认，2026-10-20**） | 必须规避微软商标，对外表述只能用 VSCodium / Code-OSS；必须接受微软专有扩展不可用（C/C++、C# Dev Kit、Remote-SSH / Remote-Containers / Remote-WSL），需准备开源替代或列入 Out-of-Scope；需评估 Open VSX 覆盖度缺口。详见 §5.1 R-01 |
| 4 | 嵌入终端 | 复用（组件）+ 自研（集成） | xterm.js + node-pty 得 4.55，是 VS Code 集成终端的同款底座，跨平台支持 Windows ConPTY 与 POSIX forkpty | xterm.js + node-pty | 原生模块需按 Electron ABI 重新编译；需配置 asar unpack |
| 5 | 工作区（Workspace） | 自研 | 无标杆提供可直接复用的工作区抽象；VS Code Agent HQ 的 Git worktree 隔离是可借鉴范式 | 自研 Workspace + Git worktree 隔离 | 需先确定本地工作区与云端工作区的主副本关系（资料摘要 Q10 未解） |
| 6 | MCP 标准协议实现 | 复用（SDK）+ 自研（双角色） | 官方 TS SDK 得 4.85；MCP 已形成事实标准（2025-06-18 版规范定义 stdio 与 Streamable HTTP 两种标准传输，SSE 已废弃） | `@modelcontextprotocol/sdk`（TS） | 需同时实现 client 与 server 双角色；需保留对旧 HTTP+SSE 的兼容探测 |
| 7 | Skill 标准协议实现 | 复用（标准）+ 自研（加载器） | **业界已事实上统一**：agentskills.io 的 SKILL.md 格式得 5.00，已被 30+ 平台支持（含 Codex CLI、Claude Code、Gemini CLI、GitHub Copilot、Cursor、VS Code、Roo Code、Goose） | 采纳 agentskills.io 开放标准 | frontmatter 严格限定六字段（name / description / license / compatibility / metadata / allowed-tools），扩展字段放入 metadata |
| 8 | 云端工作区 | **自研（自建于腾讯云）** | 用户裁决为自建，数据完全自持、无第三方合同、隔离强度与成本自控。**经中间确认 IC-03 已冻结为自建**；云厂商经 IC-04 定为腾讯云（CVM / 轻量应用服务器 + COS + VPC + CAM） | 腾讯云 CVM（或轻量应用服务器）+ COS + VPC + CAM | **必须先补齐 Q4 剩余项（账号 / Region / 配额 / 预算）**；用户已知悉并接受「Q4 未解则无法设计」这一前提；自建需建设的**最小能力集**见 §4.4：镜像与快照（SB-01）、预热池（SB-02）、实例回收（SB-03）、网络策略（SB-04）、配额与成本模型（SB-05）、多租户隔离（SB-06） |
| 9 | 云端沙箱 + 接入消息通道 | **自研（沙箱自建）+ 自研（通道适配）** | 沙箱侧经 IC-03 冻结为自建；通道侧无标杆可直接复用，需自研统一适配器（4.70）。技术参考实现为 E2B（Apache-2.0 可自托管，Firecracker microVM 路线），Daytona 因 2026-06 转闭源不参考 | 自建沙箱编排层 + 自研 Channel Adapter | 隔离原语取决于腾讯云 CVM 是否透传 /dev/kvm（待核实，§5.2 U-05）；需先明确「消息通道」外延（资料摘要 Q5 未解）；自建后代码不出本机 / 落自有云，不构成第三方上传 |
| 10 | 调用本地 shell / CLI / 应用程序 | 自研 | 无标杆可直接复用；需与能力原子 12 的沙箱策略合并设计 | 自研执行器 + 平台原生沙箱原语 | 需定义命令白名单与危险命令拦截（参照 Claude Agent SDK 的 PreToolUse Hooks 范式） |
| 11 | 本地系统权限 | 自研 | 无标杆可直接复用；Codex CLI 的 approval_policy × sandbox_mode 双轴模型是直接可借鉴蓝本 | 自研权限分级（读 / 工作区写 / 全局写 / 网络） | 需先界定权限项范围（资料摘要 Q6 未解：文件系统读写、进程管理、剪贴板、辅助功能、管理员提权） |
| 12 | 本地沙箱安全运行环境 | 自研（基于 OS 原语） | 平台原生原语分级得 4.10，是唯一能在无容器依赖下实现跨平台隔离的路径 | macOS Seatbelt + sandbox-exec；Windows 受限令牌 / AppContainer + Job Object；Linux Landlock + seccomp + bwrap | 需接受 Windows 侧能力弱于 macOS（已核实限制：无法阻止 world-writable 目录写入） |
| 13 | 本地部署、测试、预览服务 | 自研（本地）+ 采购（云端） | 仅本地 preview server 得 4.30，成本与复杂度最优；云端预览依赖账号基线 | 本地 preview server 为主 | 需定义预览的生命周期与端口分配策略 |
| 14 | 云端部署、测试、预览服务 | **自研（自建于腾讯云）** | 因 T8 已裁决为自建，云端预览随之落在自有云而非第三方托管 PaaS；自建 ephemeral preview 得 3.50，工程量高（泛域名证书 + 临时反代路由 + 用完自动回收三要素） | 腾讯云自建 ephemeral preview（COS + VPC + 自建反代） | 依赖 Q4 剩余项（账号 / Region / 配额 / 预算）；自建后代码不出自有云，无需评估「上传第三方」问题 |

### 4.2 MVP 范围建议

> 对 14 项能力原子给出**调研侧的 MVP 可行性判断**。最终 MVP 边界由 `business-architect` 在 G3 冻结，本节不构成范围裁决。

| 能力原子编号 | 能力项 | 建议进 MVP？ | 理由 |
| --- | --- | --- | --- |
| 1 | Agent SDK | ✅ | Vercel AI SDK 生态成熟（5/5），模型调用层可快速打通；编排层可先做单 Agent 单循环 |
| 2 | TS 全栈桌面客户端 | ✅ | Electron 生态成熟（5/5），无技术不确定性；形态已由主理人裁决为桌面客户端 |
| 5 | 工作区 | ✅ | 本地工作区为其余能力的前置依赖，无外部不确定性 |
| 6 | MCP 标准协议实现 | ✅ | MCP 已形成事实标准，官方 TS SDK 得 4.85；可先做 stdio + client 角色 |
| 7 | Skill 标准协议实现 | ✅ | 开放标准已统一（5.00），实现成本极低（本质是目录扫描 + YAML 解析 + 渐进加载） |
| 4 | 嵌入终端 | ✅ | xterm.js + node-pty 是 VS Code 同款底座（4.55），技术确定性高 |
| 10 | 调用本地 shell / CLI | ✅ | 与终端共用执行器底座，增量成本低 |
| 11 | 本地系统权限 | ✅ | 可先实现「审批在环 + 命令白名单」，OS 级隔离后置 |
| 12 | 本地沙箱安全运行环境 | ⚠️（分级进入） | macOS Seatbelt 与 Linux Landlock 可进 MVP；Windows 原生沙箱为实验特性且有已知限制，建议 MVP 阶段降级为「审批 + 白名单」，正式版再补 OS 隔离 |
| 3 | 嵌入 VSCode 作为 IDE | ⚠️（可进，但为最小集成） | IC-02 已冻结为内嵌 VSCodium + Open VSX，工程量高于扩展形态。建议 MVP 只做「内嵌启动 / 打开 / 聚焦 / 工作区同步 / Open VSX 扩展安装」的最小集成，深度定制后置；微软专有扩展的缺口按 §5.1 R-01 处理 |
| 13 | 本地部署、测试、预览 | ⚠️（可进） | 本地 preview server 确定性高（4.30），但优先级低于上述能力 |
| 8 | 云端工作区 | ❌（MVP 不进；Phase 5 交付） | **部分阻塞**：IC-04 已把云厂商定为腾讯云，但账号 / Region / 配额 / 预算仍缺，容量与成本模型无法设计。且 IC-03 为自建，需先完成 §4.4 六项能力 |
| 9 | 云端沙箱 + 消息通道 | ❌（MVP 不进；Phase 5 交付） | **双重阻塞**：Q4 剩余项缺失 + Q5 消息通道外延未定义。**注意：数据出境这一阻塞项已因 IC-03 自建而解除**——自建后代码不出本机 / 落自有云，不构成第三方上传 |
| 14 | 云端部署、测试、预览 | ❌（MVP 不进；Phase 5 交付） | **阻塞**：同能力原子 8；且依赖 8/9 先落地 |

**调研侧小结（2026-10-20 依 IC-01~IC-04 修订）**：14 项能力原子中，**8 项无外部阻塞可进 MVP**（1、2、4、5、6、7、10、11），**2 项需分级 / 最小集成进入**（3、12），**1 项可进但优先级靠后**（13），**3 项因 Q4 剩余输入缺失不进 MVP、转为 Phase 5 交付**（8、9、14）。

与 v1.0 的差异：能力原子 8/9/14 的阻塞原因由「云基线全缺 + 数据出境未定」收窄为「**仅 Q4 剩余项缺失**」——数据出境这一阻塞因 IC-03 自建裁决而解除；但同时因自建取代采购，这三项的**工程量显著上升**，business-architect 在 G3 不应按原「采购接入」的工作量估算。这一变化与资料摘要 X3 冲突仍保持一致：能力项已确定、落地输入部分补齐但核心账基仍缺。

**云厂商已定为腾讯云后，能力原子 8 / 9 / 14 是否解锁的重新判断（2026-10-20）：仍未解锁，维持「不进 MVP」**。判断理由：解锁这三项所需的可设计输入是**数值与位置**而非**厂商名称**——没有配额与预算上限就无法定出容量与成本模型（§4.4 SB-05），没有 Region 就无法定部署位置与 VPC 网段设计，没有账号就无法开始任何实测（含 U-05 的嵌套虚拟化核实）；且 IC-03 为自建，还需先交付 §4.4 的 SB-01~SB-06 六项能力。因此 IC-04 的作用是把阻塞从「云基线全缺 + 数据出境未定」**收窄**为「账号 / Region / 配额 / 预算四项仍缺 + 自建能力集未交付」，**并不构成解锁条件**。若后续补齐这四项，三项能力可从 Phase 5 提前，但仍须以 §4.4 的交付为前提。

**本地沙箱（能力原子 12）在 MVP 阶段的定位，因 IC-03 而进一步上升**：在云端自建沙箱落地之前，能力原子 12 是唯一可用的强隔离执行环境，建议 MVP 阶段明确「本地沙箱为主、代码不出本机」作为基线承诺。

### 4.3 技术栈参考建议

| 论题 | 推荐方案 | 替代方案 | 选择理由 |
| --- | --- | --- | --- |
| T1 桌面壳层 | **Electron**（经中间确认，已冻结）「IC-01，2026-10-20 用户裁决」 | （已退出候选）Tauri 2 / Neutralinojs | 4 项能力原子依赖 Node 原生能力（node-pty、内嵌编辑器、shell 调用、系统权限），Electron 是唯一能把它们放进同一 TS 进程树的方案；VS Code 与 Cursor 均基于此，属已验证路径。代价（包体 80–150MB、内存 150–300MB、冷启动 1–3 秒）已由用户明确接受，优化方向见 §5.1 R-09 |
| T2 IDE 嵌入形态 | **内嵌 VSCodium（Code-OSS 的 MIT 构建，遥测已移除）+ Open VSX 扩展源**（经中间确认，已冻结）「IC-02，2026-10-20 用户裁决」 | 降级路径：连接用户已安装的 VS Code（非并列候选，仅在 VSCodium 内嵌失败时启用） | 打分最高的扩展形态（4.45）无法达成「一体化桌面客户端内嵌 IDE」这一产品承诺，故在形态约束下选择内嵌 VSCodium（3.65）。**必须遵守三条不可回避约束**：对外表述只用 VSCodium / Code-OSS 不得用 VS Code 商标；微软专有扩展不可用需给替代或列入 Out-of-Scope；扩展源为 Open VSX 需评估覆盖度缺口。详见 §5.1 R-01 |
| T3 终端嵌入 | **xterm.js + node-pty 自研集成** | 复用 VS Code 集成终端（若 IDE 形态定为内嵌 VSCodium 则可复用其终端） | 4.55 领先；与 VS Code 集成终端同底座，跨平台支持 Windows ConPTY 与 POSIX forkpty；可控性最强 |
| T4 Agent SDK 形态 | **Vercel AI SDK（模型调用层）+ 自研编排层** | 完全自研 TS SDK（3.85） | 4.40 领先；多供应商统一 API 正好匹配双协议要求，TS 原生体验最佳。但必须加内部 AgentRuntime 抽象层，且工具循环 / 权限 / Hooks 自研 |
| T5 LLM Provider 抽象 | **自研双协议适配层** | 单一 OpenAI 兼容层 + 网关转换（4.10） | 4.55 领先；用户已明确要求 Anthropic 原生 + OpenAI 兼容双协议并行。两套协议的 system 字段位置、content block 结构、tool_use / tool_result 形态、流式事件类型、stop_reason 取值、缓存机制（Anthropic 显式 cache_control vs OpenAI 隐式前缀缓存）均有实质差异，必须各自原生实现再做归一化 |
| T6 MCP 实现 | **官方 TS SDK，client + server 双角色，stdio + Streamable HTTP（保留 SSE 兼容探测）** | 仅 client + stdio（4.30） | 4.85 领先；2025-06-18 规范已把 SSE 传输标记为废弃，Streamable HTTP 是当前标准远程传输；本地场景 stdio 仍为规范推荐 |
| T7 Skill 协议 | **采纳 agentskills.io 开放标准** | 自定义 YAML frontmatter 约定（3.45） | 5.00 满分且差值最大（1.55）；该格式由 Anthropic 发起并于 2025 年 12 月作为开放标准发布，已被 30+ 平台支持。自定义方案在本论题上无任何收益 |
| T8 云端沙箱 | **自建（自有云账号）**，云厂商 = **腾讯云**（CVM / 轻量应用服务器 + COS + VPC + CAM）（经中间确认，已冻结）「IC-03 + IC-04，2026-10-20 用户裁决」 | 技术参考实现：E2B（Apache-2.0 可自托管，Firecracker microVM 路线）。不参考：Daytona（生产平台 2026-06 转闭源） | 用户裁决为自建，收益是数据完全自持、无第三方合同、隔离强度与成本自控、数据驻留境内。**用户是在明知「工程量最大，需自建镜像 / 快照 / 预热池 / 回收 / 网络策略，且必须先补齐云账号基线（Q4 未解则无法设计）」的前提下做出的选择，该前提不作淡化**。自建最小能力集见 §4.4；隔离原语（microVM vs 容器）取决于腾讯云 CVM 是否透传 /dev/kvm，待核实见 §5.2 U-05 |
| T9 本地沙箱与权限 | **平台原生原语分级 + 统一策略层** | WSL2 / 容器统一封装（3.55） | 4.10 领先；无容器依赖、启动无开销。参照 Codex CLI 的双轴模型：审批策略（untrusted / on-request / on-failure / never）× 沙箱模式（read-only / workspace-write / danger-full-access）正交可配 |
| T10 消息通道接入 | **统一 Channel Adapter（Webhook + IM Bot + MQ，事件归一化）** | 仅 Webhook（4.10） | 4.70 领先；用户未定义通道类型（Q5），三适配器并行是唯一不返工的做法。借鉴 WorkBuddy 的「安全网关代理」替代内网穿透，避免暴露本地端口 |
| T11 长程任务编排 | **自研 EventStream + Checkpoint（TS）** | LangGraph JS 版 + Postgres Checkpoint（3.85） | 4.00 略领先但差值仅 0.15，建议**混合**：持久化语义（checkpoint / time travel / interrupt + resume）参照 LangGraph 已稳定能力，事件模型（Action / Observation 封闭词表 + 事件流即内存）参照 OpenHands，二者在 TS 中自研实现。理由：LangGraph 为 Python 优先，且其 graph 抽象与本项目 Screen A 的「执行图谱 / 实时执行流」UI 语义不完全一致 |
| T12 部署与预览 | **本地 preview server 为主，云端预览为演进项（自建于腾讯云，非第三方托管 PaaS）** | 自建 ephemeral preview（泛域名 + 反代 + 自动回收，3.50） | 仅本地方案 4.30 与双轨方案 4.25 差值仅 0.05，但双轨方案强依赖 Q4 云基线。建议 MVP 只做本地，云端预览在 Q4 剩余项补齐后纳入。因 IC-03 裁决为自建，云端预览不再评估 Vercel / Netlify 等第三方托管 PaaS |

### 4.4 自建云端沙箱最小能力集（供 platform-architect 在 G5 直接消费）

> 因 IC-03 裁决为自建，原「采购第三方沙箱即服务」所隐含的能力全部转为自研。以下六项是自建路线的**最小必要能力集**，缺任一项都无法构成一个可用的沙箱服务。本表由主理人指定产出，供 platform-architect 在 Phase 5 直接消费。

| 编号 | 自建能力 | 需要建设的内容 | 关键设计决策 | 上游依赖 |
| --- | --- | --- | --- | --- |
| SB-01 | 镜像与快照 | 基础镜像构建与版本管理；实例快照（内存 + 进程 + 文件系统）与恢复；镜像分发与缓存 | 根文件系统格式建议选支持 reflink 的文件系统（如 btrfs）以实现零拷贝快照与秒级克隆；ext4 可回退但克隆较慢；**避免网络文件系统** | 依赖 SB-06 的存储选型 |
| SB-02 | 预热池 | 常驻预热实例池以消除冷启动；池的容量水位与扩缩容策略 | 预热池直接决定冷启动时延量级：Firecracker 本身冷启动约 125~150ms，预热池可进一步降至数十毫秒级；池的规模与闲置回收策略是成本与延迟的权衡点 | 依赖 Q4 配额与预算 |
| SB-03 | 实例回收 | 会话结束 / 超时 / 异常时的实例销毁；孤儿实例巡检与强制回收；关联的临时存储与网络资源一并释放 | **回收必须覆盖「工作流中途失败」的情形**——仅依赖正常结束钩子会持续积累僵尸实例，需夜巡兜底 | 无 |
| SB-04 | 网络策略 | 出网白名单与默认禁网；入网仅暴露必要端口；DNS 策略；租户间网络隔离 | 沙箱默认应禁网，按需逐次授权；这是执行不可信生成代码的第一道防线 | 依赖腾讯云 VPC / 安全组配置 |
| SB-05 | 配额与成本模型 | 单租户并发上限；单实例存活时长上限；按秒或按 vCPU-时的计量与账单归集；预算熔断 | 需先有 Q4 的配额与预算输入才能定阈值；在 Q4 补齐前本项**只能出接口不能出数值** | **阻塞于 Q4 剩余项** |
| SB-06 | 多租户隔离 | 租户到实例的资源归属与强隔离；租户级凭据隔离；跨租户不可见 | 隔离原语取决于腾讯云 CVM 是否透传 /dev/kvm：若透传，走 Firecracker microVM（每沙箱独立内核，硬件级隔离）；若不透传，只能退回容器（共享内核，隔离弱于 microVM），**该降级必须在设计与对外文档中显式承认** | **阻塞于 U-05（待核实）** |

**腾讯云自建参考架构（依据 IC-04）**：

| 层 | 腾讯云产品 | 用途 | 状态 |
| --- | --- | --- | --- |
| 计算 | CVM / 轻量应用服务器 | 沙箱宿主实例（是否支持嵌套虚拟化待核实，见 U-05） | 产品已定，规格待定 |
| 对象存储 | COS | 镜像、快照、构建产物与预览静态资源 | 产品已定，桶与生命周期策略待定 |
| 网络 | VPC | 租户网络隔离、子网划分、出网白名单 | 产品已定，网段与安全组待定 |
| 身份与权限 | CAM | 子账号、角色、最小权限策略、审计 | 产品已定，策略粒度待定 |
| 地域 | **待定 · 假定 ap-guangzhou（广州）** | 主理人暂按此 Region 供后续推演 | **假定值，非已确认事实** |

> **Region 标注纪律**：本报告凡涉及 Region、实例规格、单价处一律标注「待定 · 假定 ap-guangzhou」。**不得**把该 Region 写成已确认事实，也**不得**据此推算任何具体单价或规格。

---

## 5. 风险与待确认项

> **四段式「风险」段**。列出调研中发现的主要风险、不确定信息、待业务架构师进一步裁决的依赖项，以及仍需人工补充调研的部分。

### 5.1 主要风险清单

| 编号 | 风险描述 | 触发条件 | 影响范围 | 严重程度 | 缓解建议 |
| --- | --- | --- | --- | --- | --- |
| R-01 | VS Code 商标与微软市场条款限制。**经中间确认 IC-02 冻结为内嵌 VSCodium 后，以下三条约束为不可回避项**：其一，产品不得使用 "Visual Studio Code" / "VS Code" 商标，对外表述只能用 VSCodium / Code-OSS；其二，C/C++、C# Dev Kit、Remote-SSH / Remote-Containers / Remote-WSL 等微软专有扩展不可用；其三，扩展源为 Open VSX 而非 Microsoft Marketplace，存在扩展覆盖度缺口 | **已发生**（IC-02 已冻结为内嵌 VSCodium + Open VSX） | 内嵌 IDE 的能力完整度；对外命名与合规；UserStory 的能力承诺表述 | 高 | 其一：产品命名、UI 文案、文档、关于页、许可证声明全部规避微软商标，仅用 VSCodium / Code-OSS 表述，并写入品牌合规检查清单；其二：为缺失能力准备开源替代（clangd 替代 C/C++、BasedPyright 替代 Python、Open Remote-SSH / Open Remote-WSL 替代 Remote 系列），无法替代的（如 C# Dev Kit）**明确列入 Out-of-Scope 并在对外文档中声明**；其三：见 R-11。降级路径：保留「连接用户已安装的 VS Code」作为 VSCodium 内嵌失败时的备选 |
| R-02 | 云端能力（能力原子 8 / 9 / 14）的云基线仍不完整：云厂商已由 IC-04 定为腾讯云，但**账号、Region、配额、预算上限四项仍缺**，容量与成本模型仍无法定出数值 | Q4 剩余四项在本阶段结束前仍未补齐 | 云端工作区、云端沙箱、云端部署预览三项能力；部署设计 §容量与 §成本 | 高 | 是否将 Q4 提升为 Phase 5 前置阻塞项，由主理人在 G2 审核时裁决（本报告不代为裁决）；MVP 明确排除这三项，转为 Phase 5 交付；架构上先定义 SandboxProvider 与 PreviewProvider 接口，使云端实现可后插；Region 暂按「待定 · 假定 ap-guangzhou」推演，**不得写成已确认事实** |
| R-03 | 自建沙箱取代第三方 SaaS 后，数据出境风险已解除，但替换为**自建工程量与运维负担**风险：需自建 §4.4 的六项能力（镜像与快照 / 预热池 / 实例回收 / 网络策略 / 配额与成本模型 / 多租户隔离），且隔离原语可能因宿主不支持嵌套虚拟化而降级 | **已发生**（IC-03 已冻结为自建） | 云端三项能力的工期与成本；沙箱隔离强度的对外承诺 | 高 | 明确两阶段落地：MVP 阶段本地沙箱为主、代码不出本机；云端自建作为 Phase 5 交付项，落自有腾讯云账号，不构成第三方上传、数据驻留境内。原「代码是否允许出境」策略开关不再需要。运维负担通过 §4.4 的能力集拆解到 platform-architect 逐项交付 |
| R-04 | Windows 本地沙箱能力显著弱于 macOS：Windows 原生实现基于受限令牌 + AppContainer，已知限制是无法阻止对 Everyone SID 具写权限目录（world-writable）的写入 / 删除 / 创建 | 在 Windows 上启用能力原子 12 的 OS 级隔离 | Windows 平台的沙箱承诺；安全设计的可信边界 | 高 | Windows 侧 MVP 降级为「审批在环 + 命令白名单 + 工作区根目录约束」，OS 级隔离后置；UI 与文档不得对 Windows 用户承诺与 macOS 等价的隔离强度；评估 WSL2 作为 Windows 侧的补充隔离层 |
| R-05 | Agent SDK 选型被单一供应商或单一框架绑定：Claude Agent SDK 仅支持 Claude 模型，OpenAI Agents SDK 为 Python 优先，Vercel AI SDK 不含编排原语 | 直接以任一 SDK 作为 Agent 层全部实现 | 能力原子 1；双协议要求；长期可维护性 | 中高 | 强制引入内部 AgentRuntime 抽象层，SDK 仅作为模型调用层可替换件；工具注册、权限判定、Hooks、事件流全部自研并位于抽象层之上；在架构文档中把「更换 SDK 的成本上限」写成显式非功能需求 |
| R-06 | 长程任务的上下文膨胀：OpenHands 已核实「40 步调试会话可超过 100K tokens」，需依赖 condensation 缓解 | 启用长程任务编排（能力原子对应 Screen A 的长程任务执行图谱） | Agent 响应质量与成本；UI 的实时执行流渲染 | 中 | 事件流即内存，但必须实现分层压缩（近期原始事件 + 远期摘要）；关键节点（Goal / Milestone 边界）强制落 checkpoint，使上下文可从检查点重建而非全量回放 |
| R-07 | 桌面壳层的原生模块维护负担：node-pty 为 C++ 原生模块，必须针对 Electron 内嵌 Node 的 ABI 重新编译，且 native 模块与可执行文件不能打进 asar | 采纳 T1 的 Electron + T3 的 node-pty | 构建链；CI/CD；跨平台打包 | 中 | 使用 Electron Forge 或 electron-rebuild 固定重建流程；配置 asar unpack 规则覆盖 .node 与 spawn-helper；在 CI 中对 Windows / macOS / Linux 三平台分别做构建与冒烟验证 |
| R-08 | Skill 标准仍在演进，allowed-tools 字段被规范标注为实验性，各客户端支持度不一；部分客户端存在私有扩展字段（如 disable-model-invocation、agents/openai.yaml） | 采纳 T7 的开放标准并依赖其高级字段 | Skill 的跨产品可移植性 | 中 | frontmatter 严格限定在规范六字段内，扩展信息一律放入 metadata；不依赖 allowed-tools 做权限判定，权限一律由本项目的权限层裁决；跟踪 agentskills.io 规范版本并在架构文档中固定所依据的版本 |
| R-09 | **Electron 的包体与内存代价**（IC-01 冻结 Electron 的直接后果）：安装包 80–150MB、空闲内存 150–300MB、冷启动 1–3 秒，显著高于 Tauri 路线的 2–10MB / 30–50MB / 0.3–1 秒 | **已发生**（IC-01 已冻结为 Electron） | 用户端下载与启动体验；低配机器上的可用性；分发带宽成本 | 中 | 用户已明确接受该代价，故本项由「选型风险」转为「待优化项」。优化方向（供系统设计与部署设计消费）：其一，按需加载非首屏模块与延迟初始化重组件；其二，内嵌 VSCodium 二进制与语言包做裁剪与懒加载；其三，关闭不必要的 Electron 特性并对渲染进程做资源限额；其四，安装器做差分更新以降低分发带宽。建议在非功能需求中设一条可验收的启动与内存基线 |
| R-10 | **腾讯云 CVM 是否透传 /dev/kvm 待核实**：Firecracker 强依赖 /dev/kvm 与 CPU 虚拟化扩展（Intel VT-x 的 vmx 或 AMD-V 的 svm 标志），若 CVM 不提供嵌套虚拟化，自建沙箱只能退回容器路线 | IC-03 自建落地时（Phase 5）；若在未核实前提下按 microVM 路线做设计 | §4.4 的 SB-06 多租户隔离强度；沙箱隔离强度的对外承诺 | 高 | **本项为「待核实」，未取得腾讯云官方明确文档，不作臆断**。核实方式：在候选实例上实测三条命令——`ls -l /dev/kvm` 检查设备节点、`systemd-detect-virt` 确认虚拟化类型、`grep -cE 'vmx 或 svm' /proc/cpuinfo` 统计 CPU 虚拟化标志数（需大于 0），并以 `[ -r /dev/kvm ] && [ -w /dev/kvm ]` 验证可读写权限。**建议 platform-architect 在 Phase 5 通过官方渠道或实测确认**。若确认为不支持：其一，退回容器路线并在设计与对外文档中**显式承认隔离强度弱于 microVM**；其二，评估腾讯云裸金属服务器或专用集群类实例作为宿主；其三，评估 Kata Containers 等折中方案 |
| R-11 | **Open VSX 扩展覆盖度缺口**：IC-02 冻结后扩展源为 Open VSX 而非 Microsoft Marketplace，部分扩展可能缺失、版本滞后或需手工安装 .vsix | **已发生**（IC-02 已冻结） | 内嵌 IDE 的语言与工具链覆盖度；UserStory 的能力承诺 | 中 | 其一，在系统设计阶段按本项目目标语言与工具链做一次 Open VSX 覆盖度盘点，产出「已覆盖 / 有替代 / 缺失」三清单；其二，缺失项准备 .vsix 手工安装或自建扩展源镜像作为兜底；其三，把盘点结果写入 UserStory 的能力边界，避免对外承诺不可用的扩展 |
| R-12 | **自建沙箱取代采购导致工程量显著上升**：原「采购第三方沙箱即服务」的接入工作量（低）转为自建六项能力（高），工期假设需整体上调 | **已发生**（IC-03 已冻结为自建） | 能力原子 8 / 9 / 14 的工期与人力估算；G3 边界冻结的工作量依据 | 中高 | business-architect 在 G3 冻结边界时**不得沿用在 T8 未裁决前形成的工期直觉**，须以 §4.4 的六项能力集为工作量分解底稿；platform-architect 在 G5 逐项交付 SB-01~SB-06；建议在架构文档中把自建沙箱单列为一个可独立验收的里程碑 |

### 5.2 待确认项（需主理人 / 业务方反馈）

> 调研中因外部信息不可得而暂不能确认的事实。

| 编号 | 待确认项 | 不确定性说明 | 若无法确认的备选路径 |
| --- | --- | --- | --- |
| U-01 | 云账号、可用 Region、配额与预算上限（对应资料摘要 Q4）**——云厂商已于 2026-10-20 经 IC-04 定为腾讯云，本项收窄为剩余四项** | 云厂商已确定（腾讯云：CVM / 轻量应用服务器 + COS + VPC + CAM），但账号、Region、配额、预算上限仍缺，故容量与成本模型仍无法定出数值。主理人暂按 ap-guangzhou 作为假定 Region 供推演 | 由主理人协调用户补齐剩余四项；补齐前云端三项能力（8 / 9 / 14）不进 MVP、架构只出接口与抽象。Region 一律标注「待定 · 假定 ap-guangzhou」 |
| U-02 | 「消息通道」的外延（对应资料摘要 Q5） | 用户原话仅到「沙箱可以接入消息通道」，未定义是 IM 机器人、Webhook、MQ 还是三者皆有 | 按统一 Channel Adapter 三适配器并行设计；若始终无法确认，MVP 只实现 Webhook 适配器 |
| U-03 | 「本地系统权限」的边界（对应资料摘要 Q6） | 未界定是否含文件系统读写、进程管理、剪贴板、辅助功能、管理员提权 | 安全设计阶段按最小权限原则给出默认分级，并由用户在设置中逐项授权；调研侧建议 MVP 只开放文件读写与进程管理 |
| U-04 | 腾讯 CodeBuddy Agent SDK 是否支持用户自有模型凭据与 OpenAI 兼容协议 | 官方文档仅说明其支持在 TS / Python 应用中调用 Agent，未公开是否可接入外部供应商凭据与双协议 | 由主理人向腾讯云侧确认；未确认前不作为本项目的 Agent SDK 候选，仅作为形态与 Skills 组织方式的参考 |
| U-05 | **腾讯云 CVM 是否支持嵌套虚拟化 / 是否可运行 Firecracker microVM**（决定自建走硬件级隔离还是退回容器隔离；本项由原「E2B 与 Daytona 中国区可用性」改写而来，见下方编号沿革说明） | 未取得腾讯云官方文档对该问题的明确声明。已核实的事实：其一，Firecracker 强依赖 /dev/kvm 与 CPU 虚拟化扩展，无软件回退路径（无模拟模式）；其二，云 VM 上运行 Firecracker 的前提是 hypervisor 向 guest 透传虚拟化扩展（嵌套虚拟化），而是否透传因厂商与实例族而异，需实测；其三，腾讯云 CVM 底层基于 KVM 构建，且有第三方资料称部分实例族支持嵌套虚拟化、部分资料称大多数标准实例不开启，两说并存且均非官方来源，**不足以支撑结论**。**结论：待核实，建议 platform-architect 在 Phase 5 通过 CloudQ 或实测确认，本报告不作臆断** | 由 platform-architect 在 Phase 5 通过官方渠道或实测确认（命令见 §5.1 R-10）；若确认为不支持，退回容器路线并显式承认隔离强度降级，或评估裸金属 / 专用集群类实例 |
| U-06 | 内嵌 VSCodium 的具体版本锁定策略与升级合并成本 | VSCodium 为社区构建，发布节奏与 VS Code 上游不同步；未取得本项目目标版本的构建可用性证据 | 在系统设计阶段固定一个 LTS 版本并建立上游升级评估流程；先做一次可行性 spike 验证内嵌与 IPC 打通 |
| U-07 | Windows 侧是否接受 WSL2 作为补充隔离层 | 用户明确 windows 与 macOS，但未提 Linux / WSL（对应资料摘要 Q8） | 由主理人确认 WSL 是否在范围内；若不在范围内，Windows 沙箱能力需按 R-04 降级并对外明示 |
| U-08 | 14 项能力原子的优先级与 MVP 边界（对应资料摘要 Q7） | 用户为并列枚举，无优先级；本报告 §4.2 给出的是调研侧可行性判断，非范围裁决 | 由主理人 + business-architect 在 G2 / G3 阶段冻结 |

> **编号沿革说明（2026-10-20）**：原 U-05「E2B 与 Daytona 在中国区的可用性与数据驻留选项」已因 IC-03 裁决为自建而**消解**——E2B 与 Daytona 已退出供给方式候选，无需再核实在华可用性；E2B 仅保留为自建方案的**技术参考实现**（Firecracker microVM 路线，Apache-2.0 可自托管），Daytona 因生产平台 2026-06 转闭源不作为参考。该编号位随即改写为自建路线下的核实项，即上表 U-05（腾讯云 CVM 嵌套虚拟化）。该核实项在 v1.1 期间曾暂用 U-09 这一临时编号，现统一并入 U-05，全文引用（§3.3、§4.1 原子 9、§4.3 T8、§4.4 SB-06、§5.1 R-10、§5.3 D-09、§6 SR-60 / SR-62 / SR-64、附录 B.1.1、附录 B.3）已同步更新。

### 5.3 需业务架构持续关注的依赖项

> 调研中发现但不由 `research-analyst` 裁决的下游问题。

| 编号 | 依赖项 | 说明 | 建议关注阶段 |
| --- | --- | --- | --- |
| D-01 | 桌面壳层形态与 Agent 进程模型的耦合 | 若 T1 定为 Electron，node-pty、内嵌编辑器、Agent 主循环可同进程树；若改为 Tauri 则需 sidecar 编排，进程模型与 IPC 设计完全不同 | 系统设计 §进程与模块设计 |
| D-02 | IDE 嵌入形态决定产品形态的对外表述 | 扩展形态 vs 内嵌形态，会改变 UserStory 的主交互路径与「一体化客户端」这一产品承诺的成立与否 | 高层架构设计 §业务边界 + UserStory |
| D-03 | 双协议适配层的归一化契约 | 两套协议在 system 字段位置、content block 结构、tool_use / tool_result 形态、流式事件类型、stop_reason 取值、缓存机制上均有实质差异，归一化契约需在系统设计阶段冻结为接口 | 系统设计 §接口清单 |
| D-04 | 自建云端沙箱的六项能力交付（SB-01~SB-06） | IC-03 冻结为自建后，原「SaaS 迁移退出通道」的设计目标已不存在，替换为**自建能力集的逐项交付**。platform-architect 需直接消费 §4.4：镜像与快照、预热池、实例回收、网络策略、配额与成本模型、多租户隔离 | 高层架构设计 §方案决策 + 部署设计（G5） |
| D-09 | 腾讯云 CVM 嵌套虚拟化的实测结论 | 决定自建沙箱走 Firecracker microVM 还是容器路线，进而决定沙箱隔离强度的对外承诺。本报告标注为「待核实」（U-05），不得在设计阶段当作已确认 | 部署设计（G5）+ 安全设计（G5） |
| D-05 | 本地沙箱三套 OS 原语的统一策略层 | macOS / Windows / Linux 三套原语能力不等价，需要一层统一策略 + 能力降级声明，且该降级是对外可感知的 | 安全设计 §威胁建模 + 系统设计 |
| D-06 | 长程编排的检查点粒度与 UI 映射 | Screen A 的「长程任务执行图谱 / 实时执行流 / 检查点与恢复」与 Screen B 的「Goal-Milestone-Task 三层拆解 / Critic 评分趋势 / 干预记录」需要落到具体的事件与状态模型上 | 系统设计 §领域模型 + UserStory |
| D-07 | 云端预览与工作区数据的主副本关系 | 本地工作区与云端工作区是同步、独立还是主备（资料摘要 Q10 未解），直接决定数据一致性设计 | 高层架构设计 §业务架构 + 系统设计 §数据设计 |
| D-08 | API Key 与凭据的存储与隔离粒度 | 双协议直连真实 API 意味着至少两套凭据；存储位置、隔离粒度与轮换方式未定（资料摘要 Q11） | 安全设计 §密钥管理 |

---

## 6. 关键来源目录

> 集中列出全部调研所使用的公开资料、官方文档、社区仓库、分析报告等。每条来源不低于 URL 粒度，关键来源给出具体章节或段落。

**硬指标**：≥ 3 条来源，至少覆盖每家标杆；关键数据（版本号、性能基准、定价）指定来源段落。

| 编号 | 来源类型 | 标题 / 名称 | URL / 路径 | 相关章节 | 最后访问日期 |
| --- | --- | --- | --- | --- | --- |
| SR-01 | 官方文档 | Intercept and control agent behavior with hooks（Claude Agent SDK） | https://platform.claude.com/docs/en/agent-sdk/hooks | B1, §2.2.1, §3.1.2 T4/T11 | 2026-10-20 |
| SR-02 | 官方文档 | Use Claude Code features in the SDK | https://claude-wiki.com/use-claude-code-features-in-the-sdk.html | B1, §2.2.1, §2.3 | 2026-10-20 |
| SR-03 | 社区仓库（官方 SDK 参考转述） | Agent SDK — TypeScript（@anthropic-ai/claude-agent-sdk 完整 API 与 Hook 事件清单） | https://github.com/uvsmile9/claude-code-system-prompts/blob/main/system-prompts/data-agent-sdk-reference-typescript.md | B1, §2.2.1 | 2026-10-20 |
| SR-04 | 官方文档 | Agent approvals & security（Codex CLI 审批策略与沙箱模式、各 OS 沙箱实现） | https://learn.chatgpt.com/docs/agent-approvals-security | B2, §2.2.2, §3.1.2 T9 | 2026-10-20 |
| SR-05 | 官方文档（镜像） | sandbox.md（Codex 沙箱与审批，含 Windows 受限令牌限制说明） | https://github.com/saurabhjha1/codex/blob/main/docs/sandbox.md | B2, §2.2.2, §5.1 R-04 | 2026-10-20 |
| SR-06 | 包管理页（历史版本） | @openai/codex npm v0.2.0（macOS Seatbelt / Linux Landlock+seccomp / Windows WSL2 说明） | https://www.npmjs.com/package/@openai/codex/v/0.2.0 | B2, §2.2.2 | 2026-10-20 |
| SR-07 | 技术分析 | Cursor 2.0: Agent-First Architecture Complete Guide（8 并行 Agent、worktree 隔离、MCP 40 工具上限） | https://dev.to/digitalapplied/cursor-20-agent-first-architecture-complete-guide-31h0 | B3, §2.2.3, §2.3 | 2026-10-20 |
| SR-08 | 产品档案 | Cursor（Fusion Tab 模型、Background Agents、模型路由） | https://mstone.ai/tools-wizard/cursor/ | B3, §2.2.3 | 2026-10-20 |
| SR-09 | 行业分析 | Cursor AI: How Development Teams Are Shipping 3x Faster in October 2025（Background Agents 上线时间与 AWS 基础设施） | https://www.grow-fast.co.uk/blog/cursor-ai-development-teams-shipping-3x-faster-october-2025 | B3, §2.2.3, §2.3 | 2026-10-20 |
| SR-10 | 官方文档 | Cline Overview（Agent Core / SDK / CLI / Kanban / JetBrains / ACP 模式） | https://docs.cline.bot/ | B4, §2.2.4 | 2026-10-20 |
| SR-11 | 产品主页 | Cline（VS Marketplace，5,559,735 installs；人工在环审批与 MCP） | https://cline.bot/install | B4, §2.2.4 | 2026-10-20 |
| SR-12 | 竞品对比 | Roo Code vs Cline: AI Coding Extensions Battle 2026（模式、Cloud Agents、定价） | https://serenitiesai.com/articles/roo-code-vs-cline-ai-coding-2026 | B4, B5, §2.2.4, §2.2.5 | 2026-10-20 |
| SR-13 | 产品档案 | Roo Code - Autonomous Multi-Mode AI Coding Assistant (2026) | https://www.needaitool.com/tools/roo-code | B5, §2.2.5 | 2026-10-20 |
| SR-14 | 开源仓库 | OpenHands/opendevin Architecture README（EventStream、Runtime、Session 类定义） | https://github.com/OpenHands/OpenHands/tree/7cd9c92116ecda1840cccde6f7c1cd460ac6d060/opendevin | B6, §2.2.6, §3.1.2 T11 | 2026-10-20 |
| SR-15 | 技术拆解 | Inside OpenHands: How an AI Coding Agent Executes Code（动作封闭词表、上下文膨胀数据） | https://blog.sandbase.ai/inside-openhands-ai-coding-agent-architecture | B6, §2.2.6, §5.1 R-06 | 2026-10-20 |
| SR-16 | 论文解读 | OpenHands: An Open Platform for AI Software Developers as Generalist Agents（AgentSkills、委派、SWE-bench 数据） | https://lacuna.tiptreesystems.com/paper/opendevin-an-open-platform-for-ai-software-developers-as-generalist-agents/art_6538a86bc4c0cadacde9f58b13bae82e | B6, §2.2.6, §2.3 | 2026-10-20 |
| SR-17 | 行业综述 | 多智能体协同软件工程：架构与实践（OpenHands 事件流架构、SWE-bench Verified 72%、A 轮融资） | https://cloud.tencent.com.cn/developer/article/2718759 | B6, §2.2.6 | 2026-10-20 |
| SR-18 | 官方发版说明 | November 2025 (version 1.107)（Agent HQ、后台 Agent、Git worktree 隔离、复用 Claude Skills） | https://code.visualstudio.com/updates/v1_107 | B7, §2.2.7, §2.3 | 2026-10-20 |
| SR-19 | 行业媒体 | VS Code 1.107 (November 2025 Update) Expands Multi-Agent Orchestration | https://visualstudiomagazine.com/articles/2025/12/12/vs-code-1-107-november-2025-update-expands-multi-agent-orchestration-model-management.aspx | B7, §2.2.7 | 2026-10-20 |
| SR-20 | 技术分析 | The Agentic Shift: A Comprehensive Analysis of Next-Generation AI Development Environments (2025)（Windsurf Cascade、Zed GPUI） | https://thesoftix.com/the-agentic-shift-a-comprehensive-analysis-of-next-generation-ai-development-environments-2025/ | B8, B10, §2.2.8, §2.2.10 | 2026-10-20 |
| SR-21 | 官方对比页 | Zed vs. Windsurf | http://zed.dev/compare/windsurf | B8, B10, §2.2.8, §2.2.10 | 2026-10-20 |
| SR-22 | 产品档案 | Devin（AI Agents Wiki：Brain/Devbox、ACU 计费、集成列表） | https://aiagents.wiki/agents/cognition-devin | B9, §2.2.9, §2.3 | 2026-10-20 |
| SR-23 | 评测拆解 | Devin AI Review 2026（沙箱环境、Slack 集成、Sub-Agent 架构） | http://techvernia.com/pages/reviews/coding/devin.html | B9, §2.2.9 | 2026-10-20 |
| SR-24 | 官方文档 | 企业级部署 - Devin Docs（Brain 与 Devbox 架构、Enterprise Cloud 与客户专属 VPC） | https://docs.devin.ai/zh/enterprise/deployment/overview | B9, §2.2.9 | 2026-10-20 |
| SR-25 | 官方站点 | Zed（Rust + GPUI、Agentic 编辑、CRDT 协同、Ollama 支持） | https://zed.dev/ | B10, §2.2.10 | 2026-10-20 |
| SR-26 | 行业报道 | 60 万行 Rust 打造的高性能 AI 代码编辑器（Zed Agentic Editor 发布） | https://www.163.com/dy/article/JV4K85970511D3QS.html | B10, §2.2.10 | 2026-10-20 |
| SR-27 | 官方博客 | CodeBuddy Agent SDK：为你的应用注入 AI Agent 能力 | https://www.codebuddy.cn/blog/33 | B11, §2.2.11 | 2026-10-20 |
| SR-28 | 官方 / 媒体报道 | 腾讯版 WorkBuddy 发布（桌面 Agent、Skills、MCP、多 Agent 并行、安全网关代理、IM 接入） | https://huacheng.gz-cmc.com/pages/2026/03/10/b5ab698388a34da0a103431d3011333d.html | B11, §2.2.11, §2.3 | 2026-10-20 |
| SR-29 | 官方站点 | 腾讯云代码助手 CodeBuddy 首页（插件 / IDE / CLI 三形态、CloudStudio 沙盒部署） | https://www.codebuddy.cn/home/ | B11, §2.2.11, §2.3 | 2026-10-20 |
| SR-30 | 基准测试 | Bundle Size Benchmark: Tauri 2.0 vs. Electron 32 vs. Flutter 4.0 | https://johal.in/bundle-size-benchmark-tauri-20-vs-electron-32 | §3.1.2 T1 | 2026-10-20 |
| SR-31 | 技术对比 | Tauri vs Electron - Building Desktop Apps in 2025 | https://bnowdev.com/blog/tauri-vs-electron--building-desktop-apps-in-2025/ | §3.1.2 T1 | 2026-10-20 |
| SR-32 | 技术对比（中文） | 构建跨平台桌面应用选择 Electron 还是 Tauri？ | https://juejin.cn/post/7517497597886103589 | §3.1.2 T1 | 2026-10-20 |
| SR-33 | 技术对比（中文） | 比较 Electron 和 Tauri 在桌面应用开发中的选择（2025 年末权衡） | https://blog.openreplay.com/zh/比较-electron-tauri-桌面 | §3.1.2 T1 | 2026-10-20 |
| SR-34 | 官方文档（中文镜像） | VSCodium 使用文档（与 VS Code 差异、市场条款、Open VSX、不可用扩展清单） | https://www.vscodium.cn/docs.html | §3.1.2 T2, §5.1 R-01 | 2026-10-20 |
| SR-35 | 社区文档 | ArchWiki - Visual Studio Code（Code-OSS MIT 与官方专有二进制的许可差异、市场限制） | https://wiki.archlinux.org.cn/title/Visual_Studio_Code | §3.1.2 T2, §5.1 R-01 | 2026-10-20 |
| SR-36 | 开源仓库 | node-pty README（forkpty 绑定、ConPTY、真实使用者列表） | https://github.co.uk/scryptedapp/node-pty | §3.1.2 T1/T3 | 2026-10-20 |
| SR-37 | 源码解读（中文） | vscode 内置终端集成原理与源码解读（三层架构与 node-pty 跨平台实现） | https://www.php.cn/faq/1683413.html | §3.1.2 T3 | 2026-10-20 |
| SR-38 | 工程实践 | Electron Forge + node pty（原生模块重建、asar unpack 约束） | https://thomasdeegan.medium.com/electron-forge-node-pty-9dd18d948956 | §3.1.2 T3, §5.1 R-07 | 2026-10-20 |
| SR-39 | 三方对比 | AI Agent SDK Comparison: Features and Pricing（Claude / OpenAI / Google ADK / Vercel 的工具调用、MCP 深度、多语言支持） | https://www.autolearningagents.com/ai-agent-sdks/comparison.php | §3.1.2 T4, §5.1 R-05 | 2026-10-20 |
| SR-40 | 实践对比 | What are the differences between Anthropic Agent SDK, OpenAI Agents SDK, and Vercel AI SDK? | https://docs.bswen.com/blog/2026-04-29-agent-sdk-comparison-anthropic-openai-vercel/ | §3.1.2 T4 | 2026-10-20 |
| SR-41 | 协议对比 | Claude API vs OpenAI API: Complete Technical Comparison 2026（tool_use 形态、流式事件、prompt caching 差异） | https://dev.to/claudeguide/claude-api-vs-openai-api-complete-technical-comparison-2026-3i4d | §3.1.2 T5, §5.3 D-03 | 2026-10-20 |
| SR-42 | 接口对比 | OpenAI and Anthropic TypeScript interfaces compared | https://n4n.ai/blog/openai-and-anthropic-typescript-interfaces-compared | §3.1.2 T5 | 2026-10-20 |
| SR-43 | 协议对比 | Comparing OpenAI and Anthropic request bodies with curl（system 字段位置、content block、SSE 事件、缓存） | https://n4n.ai/blog/comparing-openai-and-anthropic-request-bodies-with-curl | §3.1.2 T5 | 2026-10-20 |
| SR-44 | 官方规范 | Transports - Model Context Protocol（2025-06-18 版：stdio 与 Streamable HTTP，SSE 已废弃） | https://modelcontextprotocol.io/specification/2025-06-18/basic/transports | §3.1.2 T6 | 2026-10-20 |
| SR-45 | 规范解读 | Transports: stdio vs Streamable HTTP（命名差异与安全要求） | https://github.com/techwolf-ai/ai-first-toolkit/blob/main/plugins/tool-build-kit/skills/build-mcp/references/transports.md | §3.1.2 T6 | 2026-10-20 |
| SR-46 | 技术解读 | What are MCP Tools? How They Work & How to Use Them（两种传输的适用场景与会话管理） | https://agen.co/learning-center/what-are-mcp-tools | §3.1.2 T6 | 2026-10-20 |
| SR-47 | 官方规范 | Agent Skills Specification（SKILL.md 目录结构与 frontmatter 六字段约束） | https://agentskills.io/specification | §3.1.2 T7, §5.1 R-08 | 2026-10-20 |
| SR-48 | 官方站点 | Agent Skills Overview（渐进式披露三阶段、开放标准来源、支持方） | https://agentskills.io/ | §3.1.2 T7 | 2026-10-20 |
| SR-49 | 生态分析 | The Codex CLI Skills Ecosystem: agentskills.io and Community Skills（30+ 平台支持、扫描顺序、信任分层） | https://codex.danielvaughan.com/2026/03/27/codex-cli-skills-ecosystem | §3.1.2 T7 | 2026-10-20 |
| SR-50 | 技术对比 | Integrating E2B and Daytona: Cloud Sandboxes for Next-Gen AI Development（Firecracker microVM vs 容器、冷启动数字） | https://rohitk06.in/blogs/e2b-daytona-integration | §3.1.2 T8 | 2026-10-20 |
| SR-51 | 行业综述 | AI Agent 的沙箱战争：从 chroot 到 Firecracker（E2B / Daytona / Modal / Cloudflare / Northflank 横向对比） | https://wujiaming88.github.io/2026/06/12/ai-agent-sandbox.html | §3.1.2 T8, §5.2 编号沿革说明 | 2026-10-20 |
| SR-52 | 研究报告 | Remote Tool Execution and Cloud Sandbox Platforms for AI Agents（平台对比表、冷启动量级、MCP 授权演进、agent-outside tools-inside 架构） | https://zylos.ai/research/2026-06-13-remote-tool-execution-cloud-sandbox-platforms | §3.1.2 T8/T9 | 2026-10-20 |
| SR-53 | 技术解读 | Remote MCP Server（远程 MCP 的鉴权形态与托管约束） | https://serp.fast/glossary/remote-mcp-server | §3.1.2 T10 | 2026-10-20 |
| SR-54 | 技术分析 | LangGraph 1.0 GA: Checkpoints, Durability, and Why Voice Teams Care（检查点持久化、暂停恢复、时间旅行） | https://callsphere.ai/blog/vw1g-langgraph-1-stable-checkpoints-production | §3.1.2 T11 | 2026-10-20 |
| SR-55 | 最佳实践（中文） | LangGraph 持久化执行与人机协同最佳实践（durability 三模式、checkpointer 选型、thread_id） | https://blog.hotdry.top/posts/2025/11/15/langgraph-durable-execution-hitl | §3.1.2 T11 | 2026-10-20 |
| SR-56 | 工程实践（中文） | 自托管也想要 Vercel 那种 Per-PR 预览环境？（动态子域名、临时反代路由、自动回收三要素） | https://blog.csdn.net/qq_43635619/article/details/162350022 | §3.1.2 T12 | 2026-10-20 |
| SR-57 | 工程指南 | Ephemeral PR Preview Environments, Self-Hosted: Every Pull Request Gets a Live URL | http://www.bigiron.cc/guides/ephemeral-pr-preview-environments-self-hosted | §3.1.2 T12 | 2026-10-20 |
| SR-58 | 平台对比 | Preview Environments for Every PR, Without YAML（Temps / Coolify / Vercel / Netlify 预览能力对比） | https://temps.sh/blog/how-to-set-up-preview-environments-every-pull-request | §3.1.2 T12 | 2026-10-20 |
| SR-59 | 上游文档 | material_digest.md（资料摘要 v1.0，14 项能力原子、X1~X4 冲突、Q1~Q13 待澄清项） | `.workbuddy/output/material_digest.md` | §1.1, §4.1, §5.2 | 2026-10-20 |
| SR-60 | 技术实践 | 您的 VPS 能运行 Firecracker microVM 吗？（Firecracker 强依赖 /dev/kvm；三项检查命令：设备节点、systemd-detect-virt、vmx/svm 标志计数；可读写权限验证） | https://www.ssdnodes.com/learn/lang/zh-hans/firecracker-microvms-on-a-vps | §5.1 R-10, §5.2 U-05, §4.4 SB-06 | 2026-10-20 |
| SR-61 | 技术资料 | System Requirements（Firecracker 宿主硬性要求：Linux x86_64/aarch64 + KVM + /dev/kvm 可访问；无软件回退；建议 btrfs 以支持 reflink 零拷贝快照；存储与网络要求） | https://www.noid.one/docs/getting-started/system-requirements | §4.4 SB-01, §5.1 R-10 | 2026-10-20 |
| SR-62 | 行业分析 | The AI-Native Cloud: The Agent Stack（容器 / Firecracker microVM / 常规 VM 的隔离强度与启动耗时对比表；自建 microVM 的硬性前提；各云厂商嵌套虚拟化支持差异） | https://aispectrum.io/ai-native-cloud-agent-stack | §4.4 SB-06, §5.1 R-10, §5.2 U-05 | 2026-10-20 |
| SR-63 | 工程实践 | Best bare metal providers for Firecracker in 2026（Firecracker 仅运行于具备 KVM 的 x86_64/aarch64 Linux 宿主；三种获得 /dev/kvm 的途径；三类常见失败模式） | https://www.pandastack.ai/blog/best-bare-metal-providers-for-firecracker-2026 | §4.4 SB-06, §5.1 R-10 | 2026-10-20 |
| SR-64 | 技术综述 | KVM 虚拟化——企业级硬件虚拟化的基石（腾讯云 CVM 基于 KVM 构建；提及 GPU 虚拟化、嵌套虚拟化等增强功能；KVM 需 Intel VT-x / AMD-V 硬件虚拟化扩展） | https://cloud.tencent.com/developer/article/2493605 | §5.2 U-05（注：该来源为腾讯云开发者社区文章，非官方产品规格文档，不足以单独支撑 CVM 嵌套虚拟化的结论） | 2026-10-20 |
| SR-65 | 官方站点 | 腾讯云基础设施与可用地域（含 South China Guangzhou、Shenzhen、Qingyuan 等地域，用于核对假定 Region 的可用性） | https://cloud.tencent.com/ | §4.4, §5.1 R-02 | 2026-10-20 |

---

## 7. 硬指标清单

> 汇总本模板所有章节的硬指标，供自动校验与人工审核使用。

| 章节 | 硬指标项 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| §1 | 调研问题已收敛为 ≥ 3 条可执行问题 | ✅ | 已收敛为 RQ1~RQ5 共 5 条，每条含调研对象 / 目标 / 预期产出 / 关联种子 |
| §2.1 | 标杆系统 ≥ 3 家，含 ≥ 1 家头部 SaaS | ✅ | 实际 11 家；头部 SaaS 代表 5 家（B3 Cursor、B7 Copilot Agent HQ、B8 Windsurf、B9 Devin、B11 CodeBuddy / WorkBuddy） |
| §2.1 | 标杆系统 ≥ 1 家开源或自研代表 | ✅ | 开源代表 6 家（B1 Agent SDK、B2 Codex CLI、B4 Cline、B5 Roo Code、B6 OpenHands、B10 Zed） |
| §2.2 | 每家标杆有独立详述卡片 | ✅ | B1~B11 共 11 张卡片，每张覆盖产品定位 / 目标用户 / 核心能力 / 架构特点 / 部署形态 / 集成方式 / 定价模式 / 优势 / 局限 / 参考价值 10 个维度并逐行标注置信度 |
| §2.3 | 关键能力横向事实无遗漏 | ✅ | 13 个能力维度 × 5 家标杆，逐格标注来源编号 |
| §3.1 | 对比矩阵含 5 维度 + 权重 + 评分 | ✅ | §3.1.1 五维度权重和 = 1.00；§3.1.2 七维度权重和 = 1.00，覆盖 T1~T12 全部论题共 35 个候选 |
| §3.2 | 评分结论含优先/部分/不借鉴三层 | ✅ | 优先借鉴 B6（4.50）与 B1（4.15）；部分借鉴 B3（3.50）与 B11（3.50）；不借鉴 B9（3.00），均引用得分 |
| §4.1 | 自研/采购/复用边界有明确建议 | ✅ | 14 项能力原子逐项给出建议方式、依据、候选方案与关键前提；其中原子 2、3、8、9、14 已按 IC-01~IC-04 更新为已裁决结论 |
| §4.2 | MVP 范围建议与用户诉求对齐 | ✅ | 逐项对齐 14 项能力原子，并显式声明最终边界由 business-architect 冻结；已按 IC-03 修订阻塞原因 |
| §4.4 | 自建云端沙箱最小能力集可供 platform-architect 消费 | ✅ | SB-01~SB-06 六项，每项含建设内容、关键设计决策与上游依赖；附腾讯云自建参考架构表 |
| §5.1 | 主要风险 ≥ 3 条，有缓解建议 | ✅ | R-01~R-12 共 12 条，每条含触发条件 / 影响范围 / 严重程度 / 缓解建议 |
| §6 | 关键来源可追溯（URL / 章节） | ✅ | SR-01~SR-65 共 65 条，全部到 URL 粒度，关键数据标注具体段落 |
| 全文 | 明确区分事实 / 推断 / 建议 / 风险 | ✅ | 文档开头给出五类标注约定；§2 逐行标置信度；§4 全章标注为建议；§5 全章为风险 |
| 全文 | 不存在编造来源或占位符 | ✅ | 全部来源为 2026-10-20 可访问公开资料；无模板残留标记 |
| 全文 | 中间确认协议已履行 | ✅ | T1 / T2 / T8 三项命中触发标准并逐题发起，已全部收到回注并冻结（IC-01~IC-03），另含 Q4 部分补齐裁决（IC-04）；其余 9 项自检结论见附录 B |
| 全文 | 已裁决论题标注为「经中间确认」 | ✅ | §0.1 独立裁决记录表（含裁决人、时间、用户已知悉的代价），§3.1.2 / §3.2 / §4.1 / §4.3 各处均带「经中间确认」标注与裁决编号 |
| 全文 | 未虚构云基线数据 | ✅ | 凡涉及 Region 处一律标注「待定 · 假定 ap-guangzhou」；未出现任何实例规格、单价、配额数值；CVM 嵌套虚拟化标注为「待核实」 |

---

## 附录 A：调研方法与执行流程

### 流程总览

| 步骤 | 动作 | 落入章节 |
| --- | --- | --- |
| Step0 | 读取模板 + 完整读取上游 `material_digest.md` 全文 | — |
| Step1 | 从 14 项能力原子 + 12 个主理人论题提取调研种子，收敛为 5 条可执行调研问题 | §1 |
| Step2 | 按「学术论文 / 工程博客 / 权威对比 / 开源生态」四路检索，逐家标杆取证 | §2.1、§2.2 |
| Step3 | 建立横向事实表（不评分不排序） | §2.3 |
| Step4 | 设维度与权重，两套矩阵分别打分 | §3.1 |
| Step5 | 形成三层结论 + 组合分析 | §3.2、§3.3 |
| Step6 | 输出自研 / 采购 / 复用边界、MVP 可行性判断、技术栈建议 | §4 |
| Step7 | 识别风险、待确认项与下游依赖 | §5 |
| Step8 | 汇总来源、逐项核验硬指标、执行自动校验 | §6、§7 |

### 检索策略

| 检索路径 | 关键词方向 | 用途 |
| --- | --- | --- |
| 官方文档 / 官方仓库 | Claude Agent SDK hooks、Codex CLI sandbox、VS Code 1.107 release notes、VSCodium docs、MCP transports、agentskills.io specification、node-pty README | 取已核实事实（版本号、事件名、字段约束、许可条款） |
| 权威对比 / 基准测试 | Electron vs Tauri bundle size benchmark、E2B vs Daytona sandbox、AI Agent SDK comparison、Claude API vs OpenAI API | 取性能与成本的可比较数字 |
| 大厂工程实践 / 架构拆解 | OpenHands EventStream、LangGraph checkpoint、Devin Brain/Devbox、CodeBuddy / WorkBuddy 架构 | 取架构范式 |
| 生态与合规 | VS Code 商标与市场条款、Open VSX、Daytona 许可证变更、Cloud sandbox 数据驻留 | 取合规与许可证事实 |

### 调研原则

1. **不跨源归并**：每格事实直接映射回单一来源编号，多源一致的才标综合归纳。
2. **推断必须标注**：凡无单一来源直接声明的推论，一律标「推断」并给出所依据的已核实事实编号。
3. **冲突保留**：与上游 `material_digest.md` 的 X1~X4 冲突保持并列，不做裁决（如 X3 在 §4.2 中同时体现为「能力项已确定」与「输入全缺致不可进 MVP」）。
4. **来源不编造**：凡未取得公开证据的（云定价中国区可用性、CodeBuddy SDK 的双协议支持度、VSCodium 目标版本构建可用性），一律进 §5.2 待确认项。
5. **打分不等于决策**：§3 全部结论在 §4 中以「建议」形式落地，并显式声明最终裁决权归 business-architect。

---

## 附录 B：中间确认自检报告

> 依据 `skills/aicoding-team-bootstrap/protocols/intermediate_confirmation.md` §2.4，对 12 个技术选型论题逐项完成 §2.1 判定与 §2.3 反向验证三问。

### B.1 已命中并发起中间确认的论题（3 项，均已回注并冻结）

| 论题 | §2.1 判定 | §2.3 Q1（3 个月后推翻的工程返工成本） | §2.3 Q2（用户 / 客户 / 监管是否可感知） | §2.3 Q3（与用户原始诉求是否一致） | 命中条款 | 处置与裁决结果 |
| --- | --- | --- | --- | --- | --- | --- |
| T1 桌面壳层 | 命中：三方案均合理（Electron 原生能力最强、Tauri 体积最优、Neutralinojs 最轻），且影响系统设计进程模型与部署设计打包分发；用户原话仅到「ts 全栈」，未指定壳层 | **不可控**。返工范围：主进程 / IPC 层、node-pty 集成方式（Electron 直接引用 vs Tauri sidecar）、内嵌编辑器进程编排、打包与自动更新链路、asar unpack 规则，共涉及系统设计 §进程模型与 §模块设计、部署设计 §打包分发两章约 5 张表；切换成本估 **1.5~2 人月** | **用户可感知**：安装包体积（80–150MB vs 2–10MB）、冷启动时长（1–3 秒 vs 0.3–1 秒）、空闲内存占用（150–300MB vs 30–50MB），均为用户直接体验指标 | **不一致（部分）**：用户原话为「ts 全栈」，Tauri 2 的后端为 Rust，与「TS 全栈」口径存在偏差；用户诉求未显式提及壳层 | §2.1 + §2.2(1) + §2.2(2) + §2.2(3) | **已发起并获用户裁决，已回注冻结（2026-10-20）** — IC-01。用户最终选择：**Electron**。**是否为打分首位：是**（4.65 为矩阵首位，次位 Tauri 2 为 3.25）。Tauri 2 / Neutralinojs 退出候选；代价（包体 / 内存 / 冷启动）已由用户明确接受，由选型风险转为待优化项 R-09 |
| T2 IDE 嵌入形态 | 命中：三方案均合理（扩展形态工程量最低但形态不符、内嵌 VSCodium 形态符合但受限、fork 能力上限最高但成本最高），且影响业务边界与 UserStory 交互路径；用户原话仅到「嵌入 vscode」，未指定形态 | **不可控**。返工范围：IDE 宿主进程编排、工作区与编辑器状态同步、扩展源管理、产品对外命名与合规声明，涉及高层架构 §业务边界、系统设计 §模块设计、UserStory 主交互路径约 4 个章节；从扩展形态改内嵌形态的切换成本估 **2 人月以上**，从内嵌改 fork 更高 | **用户与监管均可感知**：用户侧感知产品形态（是「一体化客户端」还是「编辑器插件」）与可用扩展范围（微软专有扩展不可用）；监管 / 法务侧感知商标使用与许可证再分发合规 | **未显式提及且改变产品形态**：用户原话「嵌入 vscode 作为 IDE」未指定形态，而该决策直接决定产品是否为一体化桌面客户端 | §2.1 + §2.2(1) + §2.2(2) + §2.2(3) | **已发起并获用户裁决，已回注冻结（2026-10-20）** — IC-02。用户最终选择：**内嵌 VSCodium + Open VSX**。**是否为打分首位：否**（3.65，矩阵首位为 VS Code 扩展形态 4.45），因打分维度无法覆盖「一体化客户端」的形态承诺；三条不可回避约束固化入 R-01 |
| T8 云端沙箱供给方式 | 命中：三方案均合理（E2B 隔离最强可自托管、Daytona 启动最快、自建数据自持），且影响部署设计资源清单与成本模型、安全设计数据驻留；用户原话仅到「有云端的工作区、沙箱」，未指定供给方式 | **不可控（原判定为供应商绑定）**。返工范围：SandboxProvider 适配层、部署设计 §资源清单与 §成本、安全设计 §数据驻留与出境评估、platform-architect 云基线核对项，约 4 张表 | **用户 / 客户 / 监管均可感知**：用户侧感知代码是否上传第三方；客户侧感知数据驻留位置与服务范围条款；监管侧感知数据出境与隐私保护方式 | **未显式提及**：用户原话「沙箱可以接入消息通道」未涉及供给方式；且资料摘要 X3 明确记录该项「落地输入全缺」 | §2.1 + §2.2(1) + §2.2(2) | **已发起并获用户裁决，已回注冻结（2026-10-20）** — IC-03 + IC-04。用户最终选择：**自建（自有云账号），云厂商 = 腾讯云**。**是否为打分首位：否**（3.45，矩阵首位为 E2B 4.35），因打分维度无法覆盖「数据自持」这一用户侧约束。用户是在**明知**「工程量最大、需自建六项能力、且必须先补齐云账号基线（Q4 未解则无法设计）」的前提下做出的选择，该前提已在 §0.1 保留、未作淡化。随后 **IC-04 把云厂商定为腾讯云**，Q4 由「全缺」收窄为「账号 / Region / 配额 / 预算四项仍缺」；E2B 降为技术参考实现，Daytona 不参考 |

### B.1.1 Q4 云基线补齐后的重新判定（2026-10-20 追加自检）

> IC-04 回注后，本报告对 T8 做了一次重新判定，以确认是否因新信息而再次命中触发标准。

| 检查项 | 判定 | 依据 |
| --- | --- | --- |
| 云厂商选择是否构成新的方案分歧？ | **否** | 用户已明确指定腾讯云，不存在 ≥2 种待用户拍板的候选方案（已由 IC-04 裁决，无需再发起） |
| 云厂商选择是否属不可逆 / 跨界感知？ | **是，但已由用户本人裁决** | 选择云厂商属长期商务绑定且决定数据驻留位置；此项已通过 IC-04 由用户本人裁决，无需重复发起（协议 §6 禁止就已裁决论题重复发起） |
| 自建沙箱的隔离原语（microVM vs 容器）是否构成新的方案分歧？ | **是，但当前不可裁决** | 该分歧的裁决前提是「腾讯云 CVM 是否透传 /dev/kvm」，而此事实**未取得官方来源确认**（U-05）。在事实未明前发起确认等于让用户裁决一个技术事实，不具可操作性 |
| 处置 | **不发起新的中间确认；改为列入待核实项 U-05 + 风险 R-10** | 建议 platform-architect 在 Phase 5 通过官方渠道或实测确认；若确认为不支持嵌套虚拟化，则由 platform-architect 就「microVM 降级为容器」这一对外可感知的隔离强度变更发起中间确认 |
| Q4 是否提升为 Phase 5 前置阻塞项 | **不由本报告裁决** | 主理人明示该项由其在 G2 审核时裁决；本报告仅如实记录「账号 / Region / 配额 / 预算四项仍缺」这一事实 |

### B.2 未命中、不发起中间确认的论题（含反向验证证据）

| 论题 | §2.1 判定 | §2.3 Q1 答案与证据 | §2.3 Q2 答案与证据 | §2.3 Q3 答案与证据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| T3 终端嵌入 | 未命中：xterm.js + node-pty 以 4.55 显著领先次位 3.55（差值 1.00），且是 VS Code 集成终端同底座，专业判断可单方裁决 | **可控**。返工范围仅限终端面板模块的渲染层与 pty 绑定层，约 1 个模块、1 张接口表；因 node-pty 的 API 面积极小（spawn / onData / write / resize / kill），替换成本估 **不超过 2 人周** | **感知不到**。判断依据：三种方案对外呈现均为「客户端内可交互的终端面板」这一同一交互路径，功能集合（命令执行、输出回显、resize）与用户操作步骤完全一致，差异仅在内部分层 | **一致**。用户原话「嵌入 ternimal」（主理人已裁决为 terminal），三方案均满足「嵌入终端」这一能力 | 不发起 |
| T4 Agent SDK 形态 | 未命中（**以强制抽象为前提**）：Claude Agent SDK 仅支持 Claude 模型、OpenAI Agents SDK 为 Python 优先，二者已被用户「双协议」要求从专业上排除；剩余 Vercel AI SDK 与自研之间可通过抽象层收敛 | **可控（以强制内部 AgentRuntime 抽象层为前提）**。返工范围：模型调用层（约 1 个模块）+ 工具 schema 映射层；工具循环、权限判定、Hooks、事件流全部位于抽象层之上不受影响。切换成本估 **3~4 人周**，低于 1 人月阈值。**该结论已作为 R-05 写入 §5.1，抽象层为硬约束而非可选项** | **感知不到**。判断依据：对外呈现均为「Agent 执行任务并流式返回结果」这一同一交互路径，能力集合（工具调用、流式输出、多轮推理）与操作步骤不变 | **一致**。用户原话「包含 agent SDK」——无论自研还是内置第三方 SDK，产品均「包含 agent SDK」，不冲突 | 不发起（抽象层已作为强制约束写入风险清单） |
| T5 LLM Provider 抽象 | 未命中：用户已**显式**要求「同时支持 Anthropic + OpenAI 双协议」（D2 §2.2），该决策点本身已被用户诉求明确选择，自研双协议适配层是实现该要求的唯一路径 | **可控**。返工范围：Provider 适配层内部的两个 Adapter 实现与归一化契约，约 1 个模块、1 张接口表；上层编排与工具层不受影响。切换成本估 **不超过 2 人周** | **感知不到**。判断依据：对外呈现均为「可配置模型供应商与协议」，用户可见的能力集合与操作步骤由用户诉求本身决定，不因适配层内部实现而变化 | **一致**。用户原话「直接接入真实大模型 API」且「同时支持 Anthropic + OpenAI 双协议」，自研双协议适配层正是该诉求的直接落地 | 不发起（用户诉求已显式指定） |
| T6 MCP 实现 | 未命中：MCP 已是事实标准，官方 TS SDK 以 4.85 领先，专业判断可单方裁决 | **可控**。返工范围：MCP 传输层实现（stdio / Streamable HTTP / 旧 SSE 兼容），约 1 个模块；工具注册与调用面不受影响。切换成本估 **不超过 2 人周** | **感知不到**。判断依据：对外呈现均为「可添加 MCP server 并使用其工具」，能力集合与操作步骤不变；传输方式属内部实现细节 | **一致**。用户原话「有 MCP」「标准协议的实现」，采用官方标准 SDK 正是该诉求的直接落地 | 不发起 |
| T7 Skill 协议 | 未命中：业界已事实上统一于 agentskills.io 开放标准，以 5.00 满分且领先次位 1.55，无方案分歧 | **可控**。返工范围：Skill 扫描与 frontmatter 解析模块，约 1 个模块；进度式加载逻辑不受影响。切换成本估 **不超过 1 人周** | **感知不到**。判断依据：对外呈现均为「技能目录 + 触发时加载」，能力集合与操作步骤不变 | **一致**。用户原话「skill 标准协议的实现」，采纳已统一的开放标准正是「标准协议」的直接落地 | 不发起 |
| T9 本地沙箱与权限 | 未命中：三种 OS 原语（macOS Seatbelt、Windows 受限令牌 / AppContainer、Linux Landlock / seccomp）是各平台的唯一可行解，不存在跨方案选择；「是否做 OS 隔离」由用户诉求本身决定（已明确要求「本地沙箱安全运行环境」） | **可控**。返工范围：统一策略层的 OS 分支实现，约 1 个模块 + 1 张权限分级表；上层策略与审批 UX 不受影响。切换成本估 **2~3 人周** | **感知不到**。判断依据：无论采用哪种原语，对外呈现均为「本地沙箱 + 权限授权弹窗」这一同一交互路径，功能集合与用户操作步骤完全一致；差异仅体现在隔离强度的承诺措辞上（已作为 R-04 写入风险清单并要求对外明示） | **一致**。用户原话「本地沙箱安全运行环境」与「本地系统权限」，本决策是该诉求在同一交互路径下的实现选择 | 不发起 |
| T10 消息通道接入 | 未命中：统一 Channel Adapter 以 4.70 领先，且用户未定义通道类型（Q5），三适配器并行是唯一不返工的解，专业判断可单方裁决 | **可控**。返工范围：适配器模块内的三个子适配器实现与事件归一化契约，约 1 个模块；上层派发逻辑不受影响。切换成本估 **不超过 2 人周** | **感知不到**。判断依据：对外呈现均为「沙箱事件可推送到外部通道」，用户可见的能力集合由 Q5 的答案决定，与本决策（是否做统一抽象）无关 | **一致**。用户原话「沙箱可以接入消息通道」，统一适配器正是该诉求的完备实现 | 不发起 |
| T11 长程任务编排 | 未命中：自研 EventStream + Checkpoint（4.00）与复用 LangGraph（3.85）差值仅 0.15，且已在 §4.3 给出「混合」结论——持久化语义参照 LangGraph、事件模型参照 OpenHands，二者在 TS 中自研实现，不构成二选一的分歧 | **可控**。返工范围：持久化后端选型（内存 / SQLite / Postgres）与事件模型实现，约 2 个模块；上层 Goal-Milestone-Task 与 Critic 逻辑不受影响。切换成本估 **3~4 人周** | **感知不到**。判断依据：对外呈现均为 Screen A 的「长程任务执行图谱 / 实时执行流 / 检查点与恢复」与 Screen B 的「Goal-Milestone-Task 三层拆解 / Critic 评分趋势」，交互路径与操作步骤由既有设计资产决定，不因编排实现选型而变化 | **一致**。Screen A / Screen B 的模块清单（D3 §3.2 / §3.3）已显式定义这些能力，本决策是其实现方式选择，未改变能力集合 | 不发起 |
| T12 部署与预览 | 未命中：仅本地（4.30）与双轨（4.25）差值仅 0.05，且已在 §4.3 给出「MVP 只做本地、云端预览待 Q4 补齐后纳入」的分阶段结论，不构成歧 | **可控**。返工范围：PreviewProvider 接口的一个实现，约 1 个模块；本地预览逻辑不受影响。切换成本估 **不超过 2 人周** | **感知不到**。判断依据：对外呈现均为「一键预览服务」，能力集合与操作步骤一致；差异仅在预览环境位于本地还是云端，属部署位置而非交互路径变化 | **一致**。用户原话「能够在本地以及云端部署、测试和预览服务」，分阶段落地正是该诉求在输入缺失条件下的唯一可行实现顺序 | 不发起 |

### B.3 自检完整性说明

- 本轮共完成 **12 个论题**的 §2.1 判定与 §2.3 反向验证三问，无遗漏。
- §2.1 命中并发起中间确认：**3 项**（T1、T2、T8），均按协议 §3.2 五项必填字段（论题 / 候选方案 / 推荐项 / 阻塞范围 / 期望回答时限）逐题单发，未合并；三项均已收到主理人回注的用户裁决原文并冻结（IC-01 / IC-02 / IC-03），另收到 Q4 部分补齐裁决（IC-04）。
- **已裁决论题不再重复发起**：T1、T2、T8 三项在本阶段内已完成裁决，依协议 §6「禁止就已裁决论题在同一阶段内重复发起」，不再就壳层、IDE 形态、沙箱供给方式重复打扰用户。若后续出现新的分歧（如 U-05 核实后需降级隔离原语），应由承接该阶段的成员发起，而非本报告。
- **T2 与 T8 的最终裁决均未取矩阵最高分**，已在 §3.1.2 与 §3.2 显式记录。这是对「加权打分最优 ≠ 已冻结选型」的直接印证：打分维度无法覆盖「产品形态承诺」与「数据自持」两类用户侧约束，裁决权属于用户而非矩阵。
- §2.1 未命中且经反向验证确认无 §2.2 命中：**9 项**（T3、T4、T5、T6、T7、T9、T10、T11、T12），每项均已给出三问的**具体证据**（返工范围具体到章节 / 模块 / 表，成本具体到周数或月数；感知判断给出具体依据；一致性判断引用用户原话或标注未显式提及），未使用「可控 / 感知不到 / 一致」等无证据结论。
- T4 的「不发起」结论附带**硬约束**：必须引入内部 AgentRuntime 抽象层（已作为 R-05 写入 §5.1）。若 business-architect 在 G3 决定放弃该抽象层，则 T4 的 Q1 答案将转为「不可控」，须由 business-architect 重新发起中间确认。
- 本报告全部评分结论均为**建议**，未以任何形式表述为「已选定 / 已冻结」，下游 business-architect 保留完整裁决空间。
