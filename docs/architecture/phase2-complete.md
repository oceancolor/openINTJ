# Phase 2 收尾报告 —— TS 端 v3.0-alpha 第二阶段

> 阶段目标：在 v2.0-python-reference 之上完成 TS 重写的核心装配 + 持久化 + 客户端 + RFC-003 三方向原型。
>
> 状态：**完成**（typecheck 全绿 / 全量测试 286 / 286 passed）
>
> 完成日期：2026-04-29

---

## 一、阶段范围

Phase 2 在 Phase 1（4 平面骨架 + TAO/ReAct + Hooks）落地的基础上，向四个方向纵深推进：

```
                                Phase 2
   ┌──────────────────────────────────────────────────────────┐
   │  2.1  Memory Shader Pipeline（V→G→F + ContextEngine）     │
   │  2.2  EmbeddingProvider 抽象（Ollama / Xenova / simple）   │
   │  2.3  持久化层（LanceDB 向量 + SQLite 元数据）             │
   │  2.4  应用形态（apps/server + apps/desktop Electron）     │
   │  2.5  RFC-003 三方向原型（concurrency / taskpool / dormant）│
   └──────────────────────────────────────────────────────────┘
```

每一项都遵循"对齐 Python v2.0 → TS 新增 → 单测 → 集成测试"的节奏。

## 二、新增包清单（v3.0-alpha）

| 包 | 角色 | 关键能力 |
|---|---|---|
| `@openintj/embed-ollama` | 嵌入 | 调 Ollama `/api/embeddings`，自动维度推断 |
| `@openintj/embed-xenova` | 嵌入 | 本地 `@xenova/transformers`（peer dep），sentence-transformers 系列 |
| `@openintj/storage-lance` | 存储 | LanceDB 向量存储 + 内存兜底 `InMemoryVectorStore` |
| `@openintj/storage-sqlite` | 存储 | better-sqlite3 元数据 / 审计 / session + 内存兜底 |
| `@openintj/concurrency` | RFC-003 多线程 | Mutex / Semaphore / Channel / ConditionVariable / AgentPool / ForkJoin / TokenBucket / BackpressureGate |
| `@openintj/taskpool` | RFC-003 任务池 | SharedContext / HybridRetriever (vec+BM25+RRF) / TaskQueue (DAG) / ObjectPool (hot/warm/cold) |
| `@openintj/dormant` | RFC-003 蛰伏记忆 | PassiveStore / PatternMiner (n-gram + LLM) / InternalizationManager（用户审批回路）|

加上 Phase 1 已有的 `core / shared / planes/* / llm/*`，目前 workspace 一共 **20 个项目**（19 packages + 1 root）。

## 三、应用形态

### apps/server（Hono HTTP API）

- 路由：`/healthz` / `/api/status` / `/api/chat`（SSE 流式）/ `/api/memory` / `/api/audit`
- 装配：`assembleServerAgent` 复用 CLI 的 `assembleAgent` 模式 + `PersistentMemoryStore`
- 测试：`apps/server/__tests__/routes.spec.ts`（7 个）

### apps/desktop（Electron + React）

- 主进程：`assembleDesktopAgent` 装配 4 平面 + IPC handler（RFC-004 协议）
- Preload：`contextBridge` 暴露 `window.openintj` 类型安全 API
- Renderer：React 18 + Tailwind + shadcn 风格三栏布局（聊天 / 轨迹 / 状态栏）
- 通道：所有 invoke + 流式事件（`tao.*` / `react.*` / 审计）经 zod 校验
- 测试：`apps/desktop/__tests__/ipc-handlers.spec.ts`（7 个）

## 四、关键技术决策

| 议题 | 决策 |
|---|---|
| 持久化形态 | 双轨：LanceDB（向量）+ SQLite（元数据 / 审计 / 会话），由 `PersistentMemoryStore` 编排 dual-write 与 hydrate |
| 重型依赖 | `@lancedb/lancedb`、`better-sqlite3`、`@xenova/transformers` 一律 **peer dependency**，避免 CI / 桌面构建强依赖 |
| 嵌入抽象 | `EmbeddingProvider` 接口同时支持同步 / 异步；`MemoryStore` 暴露 `addShortTermAsync` 等异步 API |
| 内存语义 | `MemoryFragment` 显式带 `memoryType: short_term | working | long_term`；short→long 升级要主动调用 `reassignMemoryType` |
| 着色器策略 | 严格对齐 Python `ShaderPipeline`：vertex 决 LOD / geometry 过滤 / fragment 摘要；预算耗尽时发 `event.CONTEXT_COMPACTED` |
| RFC-003 实现位置 | 单独的 3 个包，**不强制装配进默认 Agent**；通过示例 / 测试展示用法，避免主链路膨胀 |
| 工具链 | pnpm workspaces + Turborepo + Biome + Vitest + Playwright（desktop） |

## 五、对齐 Python v2.0 的清单

| Python 模块 | TS 对应 | 是否对齐完成 |
|---|---|---|
| `agent_loop.OpenINTJFramework` | `core/loop/tao.ts` + `core/loop/react.ts` | ✅ |
| `context_engine.ContextEngine` | `planes/memory/context-engine.ts` | ✅ |
| `memory_plane.MemoryStore/Retriever` | `planes/memory/store.ts` + `retriever.ts` | ✅（含异步） |
| `memory_plane.ShaderPipeline` | `planes/memory/shader/{vertex,geometry,fragment,pipeline}.ts` | ✅ |
| `control_plane` | `planes/control/` | ✅ |
| `execution_plane` | `planes/execution/` | ✅（含真正重试） |
| `governance_plane` | `planes/governance/` | ✅ |
| `llm_client (Hunyuan)` | `llm/hunyuan/` | ✅ |
| 简易 SHA256 embedding | `core/types/embedding.ts` `SimpleEmbedder` | ✅（保留为兜底） |
| FastAPI 路由 | `apps/server/routes.ts` | ✅ |
| 静态前端 | `apps/desktop/src/renderer/*` | ✅（升级为 Electron + React） |

## 六、Python v2 已知遗留 → TS 端解决情况

来自 [`python-reference.md` 第三节](./python-reference.md#三已知遗留问题仅-ts-端修复python-端不动)：

1. ✅ 死重试代码 → `Executor` 状态机已含真实重试 + 状态机合法转换表（`packages/planes/execution/__tests__/executor.spec.ts`）
2. ✅ 状态机非法转换 → 显式转换表 + 单测覆盖
3. ✅ 伪 embedding → 保留 `SimpleEmbedder` 作为兜底；新增 Ollama / Xenova / 可注入 provider
4. ✅ recency 半衰期错配 → `ShaderConfig.recencyHalfLifeHours` 独立字段
5. ✅ 测试缺失 → **286 个单测 / 集成测试**全绿（详见下文）
6. ✅ 依赖未锁版本 → pnpm-lock.yaml
7. ✅ 全局单例 → desktop 单进程天然单例；server 由 Hono lifespan 持有
8. ✅ 预加载演示耦合 → seed 由 `assembleAgent`/`assembleServerAgent`/`assembleDesktopAgent` 显式注入

## 七、CI 验证结果

```
pnpm -r typecheck       → 19 / 19 项目 PASS
pnpm -r --workspace-concurrency=1 test
                       → 17 个 test package + 2 个无测试包（shared / llm-openai-compat）
                          286 tests passed
```

各包测试条数（运行顺序，sequential）：

| Package | Tests |
|---|---:|
| `@openintj/core` | 67 |
| `@openintj/plane-control` | 12 |
| `@openintj/plane-execution` | 15 |
| `@openintj/plane-governance` | 15 |
| `@openintj/plane-memory` | 56 |
| `@openintj/llm-hunyuan` | 9 |
| `@openintj/llm-ollama` | 5 |
| `@openintj/llm-openai-compat` | — (no tests) |
| `@openintj/embed-ollama` | 7 |
| `@openintj/embed-xenova` | 4 |
| `@openintj/storage-lance` | 9 |
| `@openintj/storage-sqlite` | 7 |
| `@openintj/concurrency` | 21 |
| `@openintj/taskpool` | 16 |
| `@openintj/dormant` | 15 |
| `@openintj/cli` (含 RFC-003 集成) | 14 |
| `@openintj/desktop` (IPC) | 7 |
| `@openintj/server` (HTTP) | 7 |
| **总计** | **286** |

> Windows 下并行 esbuild 服务会偶发 "service was stopped"；Phase 2 收尾确认用 `--workspace-concurrency=1` 串行运行可稳定全绿，issue 留作 CI 优化项。

## 八、目录与依赖快照

```
ts/
├─ apps/
│  ├─ cli/        # 终端入口（含 RFC-003 集成测试）
│  ├─ server/     # Hono HTTP + SSE
│  └─ desktop/    # Electron + React + Tailwind
├─ packages/
│  ├─ core/       # 类型 / hooks / TAO / ReAct / Agent
│  ├─ shared/     # 通用工具
│  ├─ planes/{control, execution, memory, governance}
│  ├─ llm/{hunyuan, ollama, openai-compat}
│  ├─ embed/{ollama, xenova}
│  ├─ storage/{lance, sqlite}
│  ├─ concurrency/
│  ├─ taskpool/
│  └─ dormant/
└─ docs/
   ├─ architecture/
   │  ├─ python-reference.md     # v2.0 冻结说明
   │  └─ phase2-complete.md      # 本文档
   └─ rfcs/RFC-001..004.md
```

## 九、未完成 / 后续路线

> Phase 2 不在范围内，但已识别的下一阶段任务：

1. ~~**真实持久化 e2e**~~：✅ **已于 2026-05-09 完成**（Phase 3.1）。`apps/server` 通过 `OPENINTJ_DATA_DIR` env、`apps/desktop` 通过 `app.getPath('userData')` 默认启用 LanceDB + SQLite 写盘；新增 `createPersistentMemoryStore` 工厂；写入 → 关闭 → 重启 → 向量检索/审计读回 e2e 全绿。详见 [phase3-1-persistence.md](./phase3-1-persistence.md)。
2. **嵌入基准**：完成 `simple` vs `xenova` vs `ollama` 在固定语料上的 nDCG 基准，写入 `packages/embed/__tests__/benchmark.spec.ts`。
3. **Desktop E2E**：使用 Playwright 启 Electron + 真渲染器，跑 mock chat 路径。
4. ~~**RFC-003 装配**：把 `concurrency`/`taskpool`/`dormant` 接入主 Agent 的可选模式（默认关闭，环境变量 / 配置启用）。~~ ✅ **已于 2026-05-11 完成**（Phase 3.3）。`apps/server` + `apps/desktop` 都加了三个 opt-in：`enableDormant` / `retrievalMode` / `rateLimit`，HTTP `/api/dormant/*` 与 `/api/memory?mode=hybrid` 已上线，desktop IPC 同步扩展。详见 [phase3-3-rfc3-wiring.md](./phase3-3-rfc3-wiring.md)。
5. **行为对齐测试**：按 `python-reference.md` §五的"对齐测试模式"，对每个核心组件补一条 Python v2 → TS 的事件序列对比。
6. **打包发布**：`electron-builder` Win/macOS 双平台产物 + `electron-updater` 升级链路。
7. **可观测性**：把 hooks 输出接入 OpenTelemetry / Datadog（生产场景）。
8. ~~**CI 工作流**：GitHub Actions 跑 typecheck/test~~ ✅ **已于 2026-05-09 完成**（Phase 3.2）。仓库根 `.github/workflows/ci.yml` 三个 job：lint+typecheck（Node 20/22）/ test（ubuntu/win/mac）/ e2e-persistence（ubuntu）。turbo cache 已识别 `OPENINTJ_*` env。

## 十、阶段交付物索引

- TS 源码主入口：[`ts/`](../../ts/)
- 核心 RFC：
  - [RFC-001 TAO ↔ ReAct 双层循环](../rfcs/RFC-001-tao-react-loop.md)
  - [RFC-002 Hooks 系统](../rfcs/RFC-002-hooks-system.md)
  - [RFC-003 三大架构方向](../rfcs/RFC-003-three-architecture-directions.md)
  - [RFC-004 桌面 IPC 协议](../rfcs/RFC-004-desktop-ipc-protocol.md)
- Python 冻结说明：[python-reference.md](./python-reference.md)
- 架构论证：[../agent-architecture-research_20260422.md](../agent-architecture-research_20260422.md)
- CHANGELOG：[../../CHANGELOG.md](../../CHANGELOG.md)

---

**Phase 2 完成于 2026-04-29。下一阶段 Phase 3 启动前，建议先做 §九 的"行为对齐测试"与"真实持久化 e2e" 。**
