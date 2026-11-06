# OpenINTJ — TS Monorepo (v3)

> 代码仓库：<https://github.com/oceancolor/openINTJ>
>
> 这是 OpenINTJ v3 的 TypeScript 重写仓。Python 实现冻结在 tag `v2.0-python-reference`，详见 [../docs/architecture/python-reference.md](../docs/architecture/python-reference.md)。

## 仓库结构

```
ts/
├── packages/
│   ├── core/                    核心：循环 + 钩子 + 类型
│   ├── runtime/                 RFC-011 统一装配面：PromptSection 注册表 + run 前置链
│   ├── profile/                 RFC-014 个人档案：结构化、可编辑的记忆控制面
│   ├── scheduler/               RFC-015 调度器：持久定时任务 + 补跑/安静时段/通知事件
│   ├── connectors/              RFC-016/017 连接器：笔记/日历/邮件（含可写面）+ MCP 客户端
│   ├── channels/                RFC-018 消息/移动通道：调度事件 → webhook 出站通知
│   ├── voice/                   RFC-019 语音输入：STT 提供者（HTTP 兼容端点 / mock）
│   ├── sync/                    RFC-019 跨端同步：档案 + 定时任务 Bundle v1 合并
│   ├── shared/                  跨包工具
│   ├── planes/
│   │   ├── control/             控制平面：GoalParser + Planner DAG
│   │   ├── execution/           执行平面：ToolHub + AgentPool + 线程原语
│   │   ├── memory/              记忆平面：HybridRetriever + ShaderPipeline + 三层记忆
│   │   └── governance/          治理平面：Policy + Audit + Quota
│   ├── llm/
│   │   ├── runtime/             统一 ModelRuntime（provider 选择 / 指纹 / 健康状态）
│   │   ├── ollama/              Ollama 本地适配
│   │   ├── hunyuan/             腾讯混元适配
│   │   └── openai-compat/       OpenAI 兼容占位（尚未接入 runtime）
│   └── storage/
│       ├── lance/               LanceDB 向量库
│       └── sqlite/              SQLite 元数据/审计
├── apps/
│   ├── desktop/                 Electron 客户端
│   ├── cli/                     终端版（开发/调试）
│   └── server/                  可选：HTTP 服务（云端模式）
└── docs/
    ├── rfcs/                    架构设计 RFC
    └── architecture/            架构图、ADR
```

## 快速开始

```bash
# 安装依赖
pnpm install

# 启动 CLI（最小验证）
pnpm cli

# 启动桌面客户端开发模式
pnpm desktop:dev

# 跑测试
pnpm test

# 类型检查
pnpm typecheck

# 格式化 + lint 修复
pnpm lint:fix
```

## 工程要求

- Node ≥ 20.10
- pnpm ≥ 9.0
- 强类型：`strict: true` + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`
- 强 lint：Biome 默认配置 + 手工增强（详见 [biome.json](./biome.json)）
- 强测试：每个 package 必须有 Vitest 单测；plane 级别需要"行为对齐测试"

## 路线图

当前实现与设计边界见 [docs/architecture/next-session.md](../docs/architecture/next-session.md) 和
[docs/rfcs/](../docs/rfcs/)：

1. [RFC-001 TAO + ReAct 双层循环](../docs/rfcs/RFC-001-tao-react-loop.md)
2. [RFC-002 函数钩子系统](../docs/rfcs/RFC-002-hooks-system.md)
3. [RFC-003 多线程 / 任务池 / 钝化记忆三方向](../docs/rfcs/RFC-003-three-architecture-directions.md)
4. [RFC-004 桌面客户端 IPC 协议](../docs/rfcs/RFC-004-desktop-ipc-protocol.md)
5. [RFC-005 本地模型运行时](../docs/rfcs/RFC-005-local-model-runtime.md)
6. [RFC-006 产品行为契约](../docs/rfcs/RFC-006-product-behavior-contract.md)
7. [RFC-007 TaskPool 编排](../docs/rfcs/RFC-007-task-orchestration.md)
8. [RFC-008 输入结构化与消歧](../docs/rfcs/RFC-008-structured-input-and-disambiguation.md)
9. [RFC-009 Session Journal](../docs/rfcs/RFC-009-session-journal.md)
10. [RFC-010 斜杠命令与 Agent Handle](../docs/rfcs/RFC-010-agent-handle-and-commands.md)
11. [RFC-011 统一 Runtime 与提示词段](../docs/rfcs/RFC-011-unified-runtime-and-prompt-sections.md)
12. [RFC-012 压缩 / 能力缝 / Subagent](../docs/rfcs/RFC-012-compaction-seams-and-subagents.md)（§1 压缩+检查点、§3 Subagent 已落地；§2 Fs/Shell 缝仍 Deferred）
13. [RFC-013 交互审批与能力授权](../docs/rfcs/RFC-013-interactive-approval-and-grants.md)
14. [RFC-014 个人档案](../docs/rfcs/RFC-014-personal-profile.md)
15. [RFC-015 Scheduler 与通知](../docs/rfcs/RFC-015-scheduler-and-notifications.md)
16. [RFC-016 连接器框架与 MCP](../docs/rfcs/RFC-016-connector-framework-and-mcp.md)
17. [RFC-017 可写连接器](../docs/rfcs/RFC-017-writable-connectors.md)
18. [RFC-018 消息/移动通道](../docs/rfcs/RFC-018-channels.md)
19. [RFC-019 语音输入与跨端同步](../docs/rfcs/RFC-019-voice-and-sync.md)

> RFC-005–RFC-019 的核心实现已落地（RFC-012 §2 与各 RFC 明示的 Deferred 项除外）。
> 生产收口项与真实环境验收不在本 README 重复维护，统一以 `next-session.md` 文首工作队列为准。
