# Python v2.0 参考实现说明

> 本文档记录 OpenINTJ 仓库中 Python 实现（位于仓库根目录）的"语义参考实现"地位。
>
> Tag: `v2.0-python-reference`
> Freeze 日期：2026-04-29
> 状态：**冻结，不再接收新功能；仅修复严重 bug**

---

## 一、定位

OpenINTJ 仓库下一阶段（v3.0）切换到 TypeScript + Electron 桌面客户端技术栈
（详见 [../rfcs/](../rfcs/) 中四份 RFC 与 [`.cursor/plans/`](../../.cursor/plans/) 中的路线图）。

为避免双轨开发的协调成本，**Python 端从本 tag 起冻结**：

- ✅ 作为 TS 端的"语义参考实现"——遇到行为不一致时以 Python v2.0 的输出为准对齐
- ✅ 作为 4 平面分层架构、记忆着色器管线、Agent Loop 5 阶段闭环的"可运行规格说明"
- ✅ 作为对接腾讯混元大模型的最小工作样例
- ❌ 不再添加新功能（多线程、任务池、钝化记忆等都在 TS 端落地）
- ❌ 不再追加 API（除非为了 TS 端行为对齐而需要补"探针"）

## 二、Python 仓库的内容清单

| 模块 | 文件 | 关键能力 | TS 端对应 |
|---|---|---|---|
| 入口 | [main.py](../../main.py) | FastAPI + 6 路由 + 静态服务 | `apps/server/`（可选） |
| 类型/配置 | [framework_core.py](../../framework_core.py) | ErrorCode/Command/Event/ShaderMode/LODLevel/Budget/ToolDescriptor/FrameworkConfig | `packages/core/types/` |
| 主循环 | [agent_loop.py](../../agent_loop.py) | PERCEIVE→DECIDE→ACT→OBSERVE→REFLECT 5 阶段 + OpenINTJFramework | `packages/core/loop/tao.ts`（合并为 TAO 3 阶段） |
| 上下文 | [context_engine.py](../../context_engine.py) | Token 预算、JIT 加载、Session Compaction、多模态消息 | `packages/core/context/`（待 Phase 1） |
| 记忆平面 | [memory_plane/__init__.py](../../memory_plane/__init__.py) | MemoryStore（短/工/长）+ MemoryRetriever + ShaderPipeline 三阶段 | `packages/planes/memory/` |
| 控制平面 | [control_plane/__init__.py](../../control_plane/__init__.py) | GoalParser + Planner DAG + Dispatcher | `packages/planes/control/` |
| 执行平面 | [execution_plane/__init__.py](../../execution_plane/__init__.py) | StepStateMachine + ToolHub + CircuitBreaker + 4 内置工具 | `packages/planes/execution/` |
| 治理平面 | [governance_plane/__init__.py](../../governance_plane/__init__.py) | PolicyEngine + AuditTrail + QuotaGuard | `packages/planes/governance/` |
| LLM 客户端 | [llm_client.py](../../llm_client.py) | 腾讯混元 chat + vision_chat + mock 降级 | `packages/llm/hunyuan/` |
| 前端 | [static/index.html](../../static/index.html) + [static/main.js](../../static/main.js) | IDE 风格 SPA（对话/文件树/推理/输出） | `apps/desktop/src/renderer/` |
| 部署 | [Dockerfile](../../Dockerfile) / [docker-compose.yml](../../docker-compose.yml) / [nginx.conf](../../nginx.conf) / [deploy.sh](../../deploy.sh) | Linux 服务部署链路 | `apps/desktop/electron-builder.yml` 替代 |

## 三、已知遗留问题（仅 TS 端修复，Python 端不动）

以下问题在上一阶段代码审查中发现，作为 TS 端实现的"必修项"清单：

1. **死重试代码** — [execution_plane/__init__.py:336-349](../../execution_plane/__init__.py)
   - `Executor.execute` 写了 `if step.retry_count < step.max_retries:` 分支，但下面无条件跟一句 `transition(FAILED)`，永远不会真正重试
   - TS 端：实现真正的指数退避重试 + 状态机合法转换

2. **状态机非法转换** — [execution_plane/__init__.py:75-115](../../execution_plane/__init__.py) 的 `StepStateMachine.transition`
   - 在重试分支会触发 `FAILED → FAILED`，未列入合法转换集
   - TS 端：状态机合法转换表显式覆盖所有路径，并写单测

3. **伪 embedding** — [memory_plane/__init__.py:41-54](../../memory_plane/__init__.py) 的 `simple_embedding`
   - 用 SHA256 摘要构造伪向量，没有任何语义信息
   - TS 端：用 Ollama `nomic-embed-text` 或 Transformers.js 的 `bge-small`，落到 LanceDB

4. **recency 半衰期被错配** — [memory_plane/__init__.py:185-187](../../memory_plane/__init__.py) 的 `MemoryRetriever.retrieve`
   - `decay_importance(self.shader_config.max_summary_length / 10)` 把"摘要最大长度"当成半衰期小时数
   - TS 端：在 `ShaderConfig` 中拆出独立字段 `recencyHalfLifeHours`

5. **测试缺失** — 整个 Python 仓库无 `tests/` 目录
   - TS 端：每个 package 必须有 Vitest 单测；每个 plane 写"行为对齐测试"，对同一输入比对 Python v2.0 与 TS 新版的关键事件序列

6. **依赖未锁版本** — [requirements.txt](../../requirements.txt)
   - TS 端：用 pnpm-lock.yaml 强制锁定

7. **全局单例 + 多 worker 状态不共享** — [main.py:45](../../main.py) 的 `framework = OpenINTJFramework()`
   - TS 端：用 FastAPI 依赖注入或 lifespan / Electron 进程内单例（local-first 单进程天然避免）

8. **预加载演示记忆耦合在 main 模块** — [main.py:48-99](../../main.py) 的 `_preload_memories()`
   - TS 端：作为 seed 数据走显式 init API，不耦合到入口

## 四、Python 端唯一的"延寿活动"

允许做的事：
- 修复严重安全漏洞
- 修复阻塞 TS 端行为对齐的、Python 端的"事实错误"（极少数情况）
- 文档勘误

不允许做的事：
- 加新 API
- 加新平面/模块
- 改架构

## 五、行为对齐策略

TS 端每个核心组件落地时，需要对照 Python v2.0 写"行为对齐测试":

```
对齐测试模式：
  1. 准备一组固定输入（query / 状态快照 / 配置）
  2. 用 Python v2.0 跑，记录关键事件序列（events[]）
  3. 用 TS 新版跑相同输入，断言事件序列等价（顺序、类型、payload 关键字段）
  4. 不要求字面相等，但要求"语义等价"
```

具体测试位置：`packages/<name>/__tests__/parity/python-v2.spec.ts`

## 六、相关引用

- 项目顶层路线图：`.cursor/plans/openintj_ts_rewrite_roadmap_*.plan.md`
- 架构论证文档：[../agent-architecture-research_20260422.md](../agent-architecture-research_20260422.md)
- RFC-001 ~ 004：[../rfcs/](../rfcs/)
