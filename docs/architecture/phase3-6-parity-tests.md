# Phase 3.6 —— Python v2 ↔ TS 行为对齐测试

> 阶段编号：Phase 3.6（路线图 #1）
> 完成日期：2026-05-20
> Tag：`v3.0.0-alpha.6`

## 一、目标

为 TS 实现提供"行为级回归网"，把冻结的 Python v2.0 实现当作语义参考，
**在固定输入上断言 TS 输出与 Python 等价**（在显式记录的容差与已知偏差内）。

这是路线图 [#1 行为对齐测试](./next-session.md#三phase-3-候选路线按推荐顺序) 的落地。

## 二、设计

### 2.1 取证 + 回放，而不是双轨执行

不在 CI 里跑 Python，原因：

1. CI 装 Python + 维护依赖会拖慢 GitHub Actions。
2. Python v2 已冻结（`v2.0-python-reference`），其行为是稳定的"黄金值"，
   每次 CI 重新计算同样的结果是浪费。
3. fixture 化的预期输出可以人肉 code review，比"运行时比对"更透明。

具体做法：

- **Python 端**：[`scripts/python-parity/generate_fixtures.py`](../../scripts/python-parity/generate_fixtures.py)
  对 4 个 slice 跑预设输入，把结果固化为 JSON。**只读，绝不修改 Python 代码**。
- **TS 端**：4 个 `__tests__/parity/python-v2.spec.ts` 加载对应 JSON，把同样的输入喂给 TS 实现，
  断言输出在可控容差内等价。
- **CI**：只跑 TS 端 vitest，Python 与 fixture 都是 commit-in；fixture 升级靠 `schemaVersion` 守门。

### 2.2 覆盖范围

| Slice | 包 | 覆盖组件 | 新增 tests |
|---|---|---|---:|
| core | `@openintj/core` | `SimpleEmbedder` (SHA-256), `cosineSimilarity`, `decayImportance` | **23** |
| control | `@openintj/plane-control` | `GoalParser.parse`, `Planner.createPlan` | **21** |
| execution | `@openintj/plane-execution` | `StepStateMachine.transition`, `Executor.execute` 事件轨迹 | **17** |
| memory | `@openintj/plane-memory` | `MemoryStore` overflow, `MemoryRetriever` 评分组件 + 排序 | **3** |
| governance | `@openintj/plane-governance` | `PolicyEngine.check`（2026-07-08，见 next-session §9.6） | **9** |
| context | `@openintj/core` | `ContextBudget` 算术 + `ShaderConfig.get_shader_for_task`（ContextEngine 确定性内核，2026-07-08，§12.8） | **12** |
| taxonomy | `@openintj/core` | `EventType` / `CommandType` / `ErrorCode` 枚举契约（Hooks/事件最接近的对齐面，2026-07-08，§12.8） | **14** |

合计 **99 个 parity 测试**，全部跑在 CI 的 vitest 里（无需 Python）。

> **ContextEngine / HookBus 说明**：两端 `build_context` 整体架构不同（Python `ConversationMessage`+`token//4` vs
> TS `ShaderPipeline`+`estimateTokens`），全量 parity 代价大且脆，故只锁 **ContextBudget 算术 + task→shader 映射** 这一确定性
> 内核。HookBus 是 TS-only 抽象（Python v2 用局部 `events: List[Event]`，无等价物），因此**无「HookBus 行为」跨实现 parity
> 目标**；其行为由 core 单测 + `hook-bus-bench` + concurrency observability 守护，parity 仅锁其**事件/错误码分类**（taxonomy slice）。

### 2.3 容差策略

| 项 | 数值 | 原因 |
|---|---:|---|
| `SimpleEmbedder` 向量逐元素 | `1e-12` | 纯位运算（SHA-256 → byte → float），bit-identical |
| `cosineSimilarity` | `1e-12` | 纯浮点点积/sqrt，bit-identical |
| `decayImportance` | `1e-4` | Python 用 `0.693` 近似 `ln(2)`，TS 用 `Math.LN2`，相对误差 ~2e-4 |
| MemoryRetriever 评分组件 (relevance/keyword) | `1e-12` | 纯位运算 |
| MemoryRetriever 评分组件 (recency) + 最终 score | `1e-4` | 继承自 `decayImportance` |

容差是**单向**的：TS 比 Python 更精确（用了全精度 `Math.LN2`），允许 TS 略有改进，
但不允许相反方向偏离。

## 三、已知偏差矩阵

> 这些偏差**有意保留**——要么 TS 修复了 Python 已知 bug，要么 TS 做了功能扩展。
> 对齐测试要么显式接受，要么绕开（用兼容输入避免触发），不会把 TS 改回 Python 的错误行为。

### 3.1 `MemoryFragment.decay_importance` 半衰期口径（已修复）

| 端 | 代码 | 行为 |
|---|---|---|
| Python | `memory_plane/__init__.py:185-187` | `decay_importance(self.shader_config.max_summary_length / 10)` 把"摘要最大长度"当半衰期小时数 |
| TS | `ShaderConfig.recencyHalfLifeHours` (默认 24) | 独立字段，与 `framework_core.py:319 memory_half_life_hours` 对齐 |

**对齐方式**：fixture 生成器把 Python 的 `max_summary_length = 240` → Python 半衰期 = 24h，
与 TS 默认值对齐，使两边在同一组评分上严格可比。该 case 写在
`docs/architecture/python-reference.md` §三.4 已知遗留问题里。

### 3.2 `Planner` delete/execute 模板（TS 扩展）

| 端 | 行为 |
|---|---|
| Python | `control_plane/__init__.py:206-212`：`delete`/`execute` intent 都落回 general 分支（think/act/respond）|
| TS | `planner.ts`：`delete` 有专用 3 步（verify_existence/request_approval/delete），`execute` 有专用 3 步（validate_params/execute/report）|

**对齐方式**：parity 测试只对齐 5 个公共 intent（create/modify/query/plan/general）。
TS 扩展的 `delete`/`execute` 模板在 `control.spec.ts` 中有专门的非-parity 单测覆盖。

### 3.3 `Executor` 死重试 bug（TS 修复）

| 端 | 代码 | 行为 |
|---|---|---|
| Python | `execution_plane/__init__.py:336-349` | 写了 `if step.retry_count < step.max_retries:` 分支，但下面无条件跟一句 `transition(FAILED)`，永远不会真正重试，还会触发 FAILED→FAILED 非法转换 |
| TS | `executor.ts:runOne` | 用 `while(true)` 循环 + `canRetry(step)` 真正重试，状态机合法转换 |

**对齐方式**：fixture 只跑全成功路径（不触发失败重试），避免触发该已知差异。
TS 的真实重试在 `execution.spec.ts` 中有非-parity 单测覆盖。

### 3.4 `StepStateMachine` 错误码命名

| 端 | 错误码 |
|---|---|
| Python | `EXECUTION_FAILED`（粗粒度，凡是状态机问题都用这个）|
| TS | `STATE_TRANSITION_INVALID`（细粒度新增码）|

**对齐方式**：parity spec 接受两者之一为合法值。两者都属于"非法转换被拒绝"语义。

### 3.5 `StepStateMachine` 事件返回形态

| 端 | 返回类型 |
|---|---|
| Python | `framework_core.Event(event_type=EventType.STEP_STARTED/FINISHED/FAILED, source="step-state-machine", payload=...)` |
| TS | `{ stepId, from, to, timestampSec }`（无符号化 event_type）|

**对齐方式**：TS spec 在 `__tests__/parity/python-v2.spec.ts` 内重建一个 `EVENT_TYPE_FOR_TARGET` 映射表
（与 Python `agent_loop` 同口径），把 `to` 状态映射回 `STEP_STARTED/FINISHED/FAILED` 字符串后再断言。
执行平面已通过 hook bus 在更高层 emit 事件（`event.STEP_STARTED` / `event.STEP_FINISHED`），
TS 的事件分发协议是上层关注，状态机层只关心 `from`/`to`。

## 四、文件清单

### 新增

| 路径 | 用途 |
|---|---|
| `scripts/python-parity/generate_fixtures.py` | Python 端取证脚本 |
| `scripts/python-parity/README.md` | 工具使用说明 |
| `ts/packages/core/__tests__/parity/python-v2.spec.ts` | core slice TS spec |
| `ts/packages/core/__tests__/parity/fixtures/python-v2.json` | core fixture |
| `ts/packages/planes/control/__tests__/parity/python-v2.spec.ts` | control slice TS spec |
| `ts/packages/planes/control/__tests__/parity/fixtures/python-v2.json` | control fixture |
| `ts/packages/planes/execution/__tests__/parity/python-v2.spec.ts` | execution slice TS spec |
| `ts/packages/planes/execution/__tests__/parity/fixtures/python-v2.json` | execution fixture |
| `ts/packages/planes/memory/__tests__/parity/python-v2.spec.ts` | memory slice TS spec |
| `ts/packages/planes/memory/__tests__/parity/fixtures/python-v2.json` | memory fixture |
| `docs/architecture/phase3-6-parity-tests.md` | 本文档 |

### 改动

| 路径 | 改动 |
|---|---|
| `ts/biome.json` | 把 `**/__tests__/parity/fixtures/**` 加入 ignore（fixture 是 Python 产物，不走 biome formatter）|
| `CHANGELOG.md` | 新增 `3.0.0-alpha.6` 条目 |
| `docs/architecture/next-session.md` | 划掉 #1，新增本阶段记录 |

## 五、CI 数值

| 模式 | 总用例 | 通过 | skip | 备注 |
|---|---:|---:|---:|---|
| `pnpm exec turbo run test --concurrency=1`（CI） | 441 | 430 | 11 | 11 skip 来自 better-sqlite3 / LanceDB 真盘路径 |
| `OPENINTJ_E2E=1 pnpm exec turbo run test --concurrency=1`（E2E） | 441 | 441 | 0 | 真盘路径全跑 |

相比 Phase 3.5（366 / 377）净增 64 个 parity 测试。

## 六、维护提示

1. **改 Python 端任意 case 输入** → 必须重跑 `generate_fixtures.py` 才能让 TS 测试同步更新。
2. **TS 端有意改某个组件行为**（不是 bug 修复，而是 API 设计变更）→
   - 先在 `phase3-6-parity-tests.md` "已知偏差矩阵" 加一行说明；
   - 然后在对应 TS spec 里**显式接受新行为**，或在 fixture 注释里说明。
3. **新增 TS 组件** → 在 `generate_fixtures.py` 加对应 slice，重生成 fixture，写新 spec。
   保持"一个 TS 包 ↔ 一份 fixture ↔ 一个 spec"的对齐。
4. **fixture schema 不兼容升级** → 把 `schemaVersion` 从 1 改成 2，TS spec 也同步改断言，
   保证旧 fixture 在新 spec 上跑会立刻报错而不是误判。

## 七、参考

- [`docs/architecture/python-reference.md`](./python-reference.md) —— Python v2 冻结说明 + 已知遗留清单
- [`docs/architecture/phase2-complete.md`](./phase2-complete.md) §九 —— 路线图 #1 来源
- [`docs/architecture/next-session.md`](./next-session.md) —— 工作交接备忘
