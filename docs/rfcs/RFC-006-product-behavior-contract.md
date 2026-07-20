# RFC-006: INTJ-inspired 产品行为契约

| 字段 | 值 |
|------|-----|
| 状态 | **Accepted** |
| 作者 | OpenINTJ |
| 关联 | RFC-003、RFC-006 实施于 `@openintj/shared` |
| 依赖 | RFC-007（T1 战略分解执行面，首期仅契约引用） |

## 摘要

将 OpenINTJ 品牌落为**可观察、可评测**的工程行为契约（T1–T8），**不**实现 MBTI 类型判断，也不在 prompt 中声称「你是 INTJ」。

与 **Dormant User Persona** 严格分层：

| 层 | 范围 | 可撤销 | 覆盖治理 |
|----|------|--------|----------|
| Product Behavior | 全用户一致、版本化 | 否（仅 A/B 开关） | 不能覆盖工具治理/安全 |
| User Persona | 用户批准的偏好 | 是 | 不能修改 Product Behavior |

## 行为 trait（T1–T8）

1. **T1 战略分解**：复杂请求先列步骤/子目标（执行由 RFC-007 TaskPool 承接）。
2. **T2 结构化推理**：对比/规划/分析用编号或小标题。
3. **T3 证据优先**：时效/事实类先 `search` 再结论。
4. **T4 直言简洁**：去寒暄与重复。
5. **T5 必要澄清**：仅缺关键约束时追问。
6. **T6 独立执行**：权限内自主推进。
7. **T7 质量门禁**：交付前自检约束与核心问题。
8. **T8 工具治理尊重**：不覆盖安全策略与用户明确要求。

每项 trait 在 `trait-scenarios.ts` 含正例与 `judge` 条件。

## 实现（v1.1.0）

- `product-behavior.ts`：`buildProductBehaviorPrompt` / `assembleSystemPromptPrefix`
- v1.1 将契约从 prompt-only 提升为可执行边界：
  - 排序、大小写转换、简单算术约束、关键澄清与越权破坏请求走本地 deterministic preflight，
    不调用 LLM 或工具；
  - 对分阶段计划、结构化对比做一次有界 final-answer revision；
  - “一句话”要求做确定性单句收口。
- 三端拼装顺序：**Product Behavior → User Persona → Skills → Memory**
- `OPENINTJ_PRODUCT_BEHAVIOR=0` 关闭（A/B 基线组）
- CLI `chat` / `status` 可用 `--product-behavior treatment|control` 显式覆盖；未传时仍读取 env。
- server `/api/status`、desktop status 与 CLI status 暴露 `version/enabled/cohort`；server 启动日志打印 cohort。
- `classifier/routing.ts`：`planning` / `analysis` 永不 `single` 路由
- Skills：`planning`、`clarification` 能力包（按需命中）
- OTel：`event.PRODUCT_BEHAVIOR` → `openintj.product.behavior.injected`
- trait 可观测：`event.PRODUCT_TRAIT_SIGNAL` → `openintj.product.trait.signal`。只记录确定性事实：
  - T1 `plan_decomposed`：`tao.afterThink.plan.totalSteps > 1`，值为步骤数；
  - T5 `clarification_skill_selected`：技能选择器明确命中 `clarification`；
  - T3 `search_before_answer`：`tool.afterCall` 中 `search` 成功，且该生命周期点早于最终回答。
  这些 counter **不表示**模型意图、理解程度或 trait 最终通过，仅用于统计可观察执行信号。
- ReAct parser 对大小写不同的协议标记兼容，并拒绝 FINAL 中泄漏 Thought/Action；解析错误会回灌重试，
  max-iteration 时保留最佳有效 thought，而不是无条件丢弃为占位文本。

评测 runner 可选返回 `evidence.toolsUsed` / `evidence.trajectory`。T3 在结构化证据存在时要求真实
`search` 工具使用；旧的 `{ finalAnswer }` runner 保留兼容回退。T4 限制为单句、去寒暄、长度有界；
T7 同时验证算术结果与 `>3` 约束确认。

## 验收

- `evaluateTraits()` 各 trait 通过率 ≥ 60% 基线（`TRAIT_PASS_BASELINE`）
- `task-eval` / `longrun` 完成率不得显著回退
- Product Behavior 不能覆盖治理；User Persona 不能修改 Product Behavior 契约

## 基线状态（2026-07-19）

- normal CI 使用 scripted/stub runner 跑全部 8 traits（含 T5 contrast，共 9 cases），9/9 通过。
- 固定报告：`docs/architecture/rfc-006-deterministic-baseline.json`。该报告明确标为
  **deterministic，非真实模型分数**，只证明 harness 与 judges。
- 真实模型仍 gated：
  `RUN_TRAIT_EVAL=1 OPENINTJ_LLM_PROVIDER=ollama pnpm --filter @openintj/cli test -- trait.harness`
  （PowerShell 请用 `$env:...` 设置）。
- `qwen2.5:0.5b` 在 v1.1 上连续两次 treatment **9/9**，对应 control 分别 **5/9**、
  **4/9**；`baselineMet=true`，完成率增量 +44.4 / +55.6 个百分点。
- 每个 case 使用独立 Agent/memory，treatment/control 串行，避免跨 case 检索污染及本地 GPU 争用。
- 当前本机 `qwen2.5:7b` 因约 3.1GB CPU repack buffer 分配失败无法运行；这属于机器资源限制，
  不把 7B 的 0/9 provider failure 计为行为质量结果。
- T3 的当前本机 search 是 mock fallback；9/9 只证明“先调用 search”的行为契约，不证明
  Node.js 版本答案正确。事实正确性仍需配置真实 Tavily/Brave search 后单独验收。
