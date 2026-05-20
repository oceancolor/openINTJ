# Python v2 ↔ TS 行为对齐测试 —— 工具说明

> 本目录给 [`docs/architecture/phase3-6-parity-tests.md`](../../docs/architecture/phase3-6-parity-tests.md)
> 中描述的 parity 测试方案提供"Python 端取证脚本"。

## 这是什么

`generate_fixtures.py` 是一个**只读**的取证脚本：

- 加载仓库根目录冻结的 Python v2.0 实现（`framework_core` / `memory_plane` / `control_plane` / `execution_plane`）。
- 在一组**固定输入**上跑核心组件，把可观察到的输出（向量 / 评分组件 / 事件类型 / 计划步骤）固化为 JSON。
- 输出到 4 个 TS 包的 `__tests__/parity/fixtures/python-v2.json`。

TS 端的 vitest spec（`packages/*/__tests__/parity/python-v2.spec.ts`）加载这些 JSON，
把同样的输入喂给 TS 实现，断言输出在可控容差内等价。

## 运行

```powershell
cd F:\openINTJ
py scripts/python-parity/generate_fixtures.py
```

输出（重新覆盖）：

```
ts\packages\core\__tests__\parity\fixtures\python-v2.json
ts\packages\planes\control\__tests__\parity\fixtures\python-v2.json
ts\packages\planes\execution\__tests__\parity\fixtures\python-v2.json
ts\packages\planes\memory\__tests__\parity\fixtures\python-v2.json
```

依赖：只需 Python 3.10+；Python v2 自身不依赖任何第三方包。

## 什么时候需要重跑

**几乎不需要**。Python v2 已冻结（tag `v2.0-python-reference`），fixture 一次生成、长期复用。

仅以下情况重跑：

1. 修改了 `generate_fixtures.py` 自己（加新测试 case / 改 case 输入）。
2. Python 端因为严重 bug 被允许修补（见
   [`docs/architecture/python-reference.md`](../../docs/architecture/python-reference.md) §四）。
3. fixture schema 升级（`schemaVersion` 跨版本变更）。

普通的 TS 端重构 / 行为对齐验证 **不需要重跑**，直接 `pnpm exec turbo run test` 即可。

## fixture 设计

每个 JSON 文件包含：

- `schemaVersion`: 当前为 `1`；TS spec 加载时会断言版本，跨大改时强制重新生成。
- `generatedFrom`: 对应 Python 源文件 / 类名，方便回溯。
- `notes` (可选): 关键设计选择与已知偏差说明。
- 数据节：按"输入→预期输出"成对组织。

TS spec 加载 fixture 时用 `JSON.parse` + 显式类型断言；如 schema 不匹配会直接抛错而非误判。

## 已知偏差（fixture 不覆盖的部分）

详见 [`phase3-6-parity-tests.md`](../../docs/architecture/phase3-6-parity-tests.md) "已知偏差矩阵" 一节。

简表：

| 组件 | 偏差 | 处理 |
|---|---|---|
| `decay_importance` | Python 用 `0.693` 当 `ln(2)` | TS 用 `Math.LN2`，测试容差 `1e-4` |
| `MemoryRetriever` 半衰期 | Python 写死 `max_summary_length/10` (v2 bug) | fixture 把 `max_summary_length=240` → 半衰期 24h 与 TS 默认对齐 |
| `Planner` 模板 | Python `delete`/`execute` 落回 general 分支 | parity 只跑公共 5 个 intent；TS 扩展行为在自家单测覆盖 |
| `Executor` 重试 | Python 有 FAILED→FAILED 死循环 bug | fixture 只跑全成功路径；TS 修复的真实重试有专属单测 |
| `StepStateMachine` 错误码 | Python `EXECUTION_FAILED` vs TS `STATE_TRANSITION_INVALID` | TS spec 接受两者之一 |
