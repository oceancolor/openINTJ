# ADR-001：ReAct 工具调用采用文本协议（Thought/Action/FINAL）而非 OpenAI function-calling

- 状态：**已采纳（Accepted）** — 2026-06-30
- 关联：RFC-001 §11 Q1（本 ADR 关闭该未决问题）
- 决策者：核心架构

## 背景

RFC-001 §11 Q1 留了一个未决问题：ReAct 微循环里 LLM 的"调用工具 / 给最终答案"是用
**prompt 模板约定的文本协议**，还是用 **OpenAI function-calling（tools/tool_calls）协议**？

当前实现（`packages/core/src/loop/react.ts`）事实上选了文本协议：

- `buildSystemPrompt()` 把工具说明拼进 system prompt，并约定输出格式：
  ```
  Thought: <推理>
  Action: <工具名>
  Action-Input: <JSON>
  ```
  或终止：
  ```
  Thought: <推理>
  FINAL: <答案>
  ```
- `parseLlmThought()` 用正则解析这段文本，提取 `action.tool` / `action.params` 或 `finalAnswer`。

此前这是"既成事实但未文档化的决策"，本 ADR 把它正式确定下来。

## 决策

**ReAct 微循环统一采用文本协议**，不依赖 OpenAI function-calling。理由：

1. **本地优先 / 多 provider 中立**：项目要同时支持混元、Ollama 等。function-calling 的
   JSON-Schema 严格度、`tool_calls` 字段语义在各家/各本地模型上兼容性参差；文本协议对
   "任何能续写的 LLM" 都成立，这与"本地优先"定位一致。
2. **可观测 / 可调试**：Thought/Action/FINAL 是纯文本，直接进 trajectory、进 UI、进 OTel span，
   人能一眼看懂；function-calling 的结构化调用要额外反序列化才能展示。
3. **与 Python v2 行为对齐**：parity 测试网以文本协议为基准；换协议会打破已冻结的对齐 fixture。
4. **实现简单、无供应商锁定**：解析逻辑集中在 `parseLlmThought` 一处，约 50 行，易测试、易替换。

## 代价 / 已知劣势

- **鲁棒性弱于原生 function-calling**：模型偶尔不守格式（漏 `Action-Input`、JSON 不合法）。
  现状用 `parseLlmThought` 的兜底分支处理（解析失败回传 `parseError`，无 Action/FINAL 时当隐式终止）。
- **token 开销**：工具说明每轮都进 system prompt。对纯对话场景，已用
  `TaoConfig.enableReact=false` 的**退化路径**（`ReactStateMachine.runSingle`）规避——不下发工具、单次调用。
- **无并行工具调用**：文本协议一轮一个 Action；function-calling 原生支持一次返回多个 tool_calls。
  目前不需要；若将来要并行工具，见下"重新评估触发条件"。

## 重新评估触发条件（什么时候应该回到 function-calling）

满足任一即重开此决策：

1. 主力 provider 的 function-calling 兼容性达标，且**格式错误率**成为线上主要失败源；
2. 需要**一轮并行多工具**（与 RFC-003 方向一/二的并行编排结合）；
3. 引入需要严格 JSON-Schema 校验的高风险工具（如金融/系统操作），文本解析的容错不再可接受。

## 迁移路径（若将来切换）

- 在 `LlmClient` 增加可选 `chatWithTools(messages, tools, opts)`，返回结构化 `tool_calls`；
- `ReactStateMachine` 增加 `protocol: "text" | "function-calling"` 开关，`function-calling` 时
  跳过 `buildSystemPrompt` 的格式约定段、改走结构化解析；
- parity 网为两种协议各留一组 fixture；
- 文本协议作为**降级兜底**保留（provider 不支持 function-calling 时回退）。
