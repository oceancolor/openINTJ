# RFC-008: 用户输入结构化与消歧

| 字段 | 值 |
|------|-----|
| 状态 | **Accepted** |
| 作者 | OpenINTJ |
| 关联 | RFC-006 Product Behavior、RFC-007 TaskPool、Desktop 任务工作台 |
| 依赖 | `@openintj/core` input-structuring、三端 `agent.run`、Desktop Workbench |

## 摘要

在 deterministic preflight 之后、classifier / Tao / TaskPool 之前增加一轮**自适应输入结构化**：

- 简单、明确或已可由 preflight 短路的输入不调用模型；
- 复杂或高歧义输入最多调用一次有界模型，产出结构化任务理解；
- 会实质改变执行结果的缺失信息暂停并追问 1–3 个问题；
- 桌面端展示可持久化的「任务理解卡」。

这不是人格扮演，而是把任务对话长期塑造成目标、关系、约束、交付物更清晰的形式。

## 双输入语义

| 字段 | 用途 |
|------|------|
| `originalInput` | 审计、Workbench/记忆落盘、Dormant、最终质量门禁校验 |
| `executionInput` | classifier、routing、Tao/ReAct、TaskPool 执行文本 |

禁止静默改写用户原意。优化器只能把低风险、可逆默认值写入 `assumptions`；会改变结果或造成外部副作用的缺失信息必须 `clarify`。

## 自适应策略

`OPENINTJ_INPUT_STRUCTURING=off|adaptive|always`，默认 `adaptive`。

- `off`：全部 pass-through。
- `adaptive`：本地门控（复杂度 / 歧义 / 关系线索）决定是否调用模型。
- `always`：除 preflight 短路外都走结构化。

额外配置：

- `OPENINTJ_INPUT_STRUCTURING_MAX_TOKENS`（默认 512）
- `OPENINTJ_INPUT_STRUCTURING_TIMEOUT_MS`（默认 8000）
- `OPENINTJ_INPUT_STRUCTURING_AMBIGUITY_THRESHOLD`（默认 0.62）

Product Behavior **control** 组强制关闭输入结构化，保证 RFC-006 A/B 可比。

## Schema

```json
{
  "action": "proceed|clarify",
  "executionInput": "...",
  "structure": {
    "goal": "...",
    "context": [],
    "relations": [],
    "constraints": [],
    "deliverables": [],
    "dependencies": [],
    "assumptions": []
  },
  "ambiguityScore": 0.0,
  "questions": []
}
```

`clarify.questions` 长度必须为 1–3。模型输出无效、超时或取消时 fail-soft 回到原始输入；用户 AbortSignal 传播而不静默继续。优化轮不得调用工具或提升权限。

## 执行链路

```text
user query
  → dormant run.input
  → deterministic preflight（算术 / 排序 / 记忆 / 安全 / material-clarification）
  → structureUserInput（adaptive）
       ├─ clarify → 返回任务理解卡，不进入 classifier/Tao
       └─ proceed → executionInput
  → classifier / decideRoute
  → Tao / TaskPool / self-consistency
  → enforceProductBehaviorAnswer（仍用 originalInput）
  → memory 记录 originalInput
```

TaskPool 子任务不重复做输入结构化。clarification skill 仍可作为执行期兜底，但输入阶段已澄清的问题不应重复追问。

## Desktop UX

- CHAT 响应携带 `inputStructure`
- Workbench `messages` 增加 `message_kind` 与 `input_structure_json`（schema v2，向后兼容迁移）
- Renderer 渲染可折叠「任务理解」卡：
  - `proceed`：显示「已自动继续」
  - `clarify`：展开「等待补充」与问题列表

低歧义不增加额外确认点击；高歧义明确暂停。

## 可观测性

Hooks（不记录用户原文）：

- `event.INPUT_STRUCTURE_STARTED`
- `event.INPUT_STRUCTURE_COMPLETED`
- `event.INPUT_STRUCTURE_CLARIFICATION`
- `event.INPUT_STRUCTURE_FALLBACK`

OTel counter：`openintj.input.structure`，attributes=`outcome,mode,ambiguity_band`。

## 与 RFC-006 T5 的关系

T5「必要澄清」拆为两类评测：

1. **不过度澄清**：明确短任务直接执行（如大小写转换）。
2. **关键歧义必须澄清**：缺少会改变结果的约束时暂停（如「部署到生产」）。

输入结构化是 T5 的输入侧执行面；输出侧 revision 与 Product Behavior prompt 仍然有效。

## 验收

- 简单明确输入不增加模型调用；
- 优化轮最多一次；
- 澄清问题 ≤ 3；
- 结构化失败不阻塞普通请求；
- 原始输入始终可追溯；
- 工具治理与工作区边界不被优化文本覆盖；
- control 组关闭结构化后与既有基线可比。
