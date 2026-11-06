# RFC-011：统一 Runtime 装配与提示词段注册表

| 字段 | 值 |
|---|---|
| 状态 | **Accepted** |
| 起草日期 | 2026-11-02 |
| 决策日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 上游决策 | 吸收 DeepSeek Harness 的 profile / bundle 分层与 `ctx.systemPrompt.section()`；**用 TypeScript 工厂，不用 Cordis / YAML patch** |
| 影响包 | `@openintj/core` 或新建 `@openintj/runtime`；`apps/cli`、`apps/server`、`apps/desktop` 的 `agent.ts` |
| 依赖 | RFC-006 提示词顺序（**锁死**）；RFC-009 Journal；RFC-010 Handle/Commands |

---

## 摘要

CLI / server / desktop 各有一份 `assemble*Agent`，注释写着「三端共用」却是复制粘贴。提示词栈顺序写死在 [`assembleSystemPromptPrefix`](../../ts/packages/shared/src/product-behavior.ts)，plan-advisor 却把【规划】塞进 **user query**。

本 RFC：

1. 一个 `assembleOpenintj({ profile })`，三端只传差异。
2. 提示词 **段注册表**，默认顺序仍是 RFC-006：Product Behavior → User Persona → Skills → Memory。

不引入 Cordis、不引入 `cordis.patch.yml`、不把平面改成可热卸载插件内核。

---

## 1. 现状

| 文件 | 职责重叠 |
|---|---|
| [ts/apps/cli/src/agent.ts](../../ts/apps/cli/src/agent.ts) | HookBus、四平面、ToolHub+gate、skills、dormant、RFC-008、plan-advisor、`run()` |
| [ts/apps/server/src/agent.ts](../../ts/apps/server/src/agent.ts) | 同上 + 持久化、hybrid、OTel、TaskPool recovery |
| [ts/apps/desktop/src/main/agent.ts](../../ts/apps/desktop/src/main/agent.ts) | 同上 + workbench、多模型 Profile、rate limit |

差异应是 **profile 选项**，不是三份 `run()` 逻辑。

---

## 2. 装配面

推荐新包 `@openintj/runtime`（避免 `@openintj/core` 依赖 desktop/sqlite）。若评审觉得包过多，可先放 `packages/core/src/assemble.ts`，但 **禁止** core 依赖 Electron。

```typescript
type RuntimeProfile = "cli" | "server" | "desktop";

interface AssembleOpts {
  profile: RuntimeProfile;
  llm?: LlmClient;
  llmProvider?: LlmProviderId;
  workspace: WorkspaceConfig;
  hooks?: HookBus;
  persistence?: PersistenceOpts;     // server/desktop
  enableTaskPool?: boolean;
  enableDormant?: boolean;
  enableSkills?: boolean;
  // ...既有 opt-in 旗标，不发明第二套 env 名
}

interface OpenintjRuntime {
  handleFor(conversationId: string): AgentHandle;  // RFC-010
  commands: SlashCommandRegistry;
  journal: JournalFactory;                         // RFC-009
  prompt: PromptSectionRegistry;
  toolHub: ToolHub;
  hooks: HookBus;
  run(query: string, opts?: RunOpts): Promise<InputStructuredTaoResult>; // 兼容旧调用
  close(): Promise<void>;
}

export function assembleOpenintj(opts: AssembleOpts): Promise<OpenintjRuntime>;
```

Profile 差异（逻辑，不是 YAML）：

| 能力 | cli | server | desktop |
|---|---|---|---|
| Journal | 内存 | SQLite dataDir | workbench.sqlite |
| TaskPool recovery | 无 | 有 | 有 |
| OTel | 可选 | 默认开 | 默认开 |
| 多模型 Profile | 否 | 否 | 是 |
| Workbench | 否 | 否 | 是 |

三端 `agent.ts` 变成：解析 env/flags → `assembleOpenintj` → 暴露 CLI/HTTP/IPC。**禁止**再复制 `structureUserInput` → classifier → plan-advisor → `tao.run` 链。

---

## 3. 提示词段注册表

```typescript
interface PromptSection {
  id: string;
  /** 数值越小越靠前。RFC-006 预留 100/200/300/400 */
  order: number;
  render(ctx: PromptAssembleCtx): string | undefined | Promise<string | undefined>;
}

interface PromptAssembleCtx {
  query: string;
  conversationId?: string;
  history: ChatMessage[];          // 来自 RFC-009 deriveHistory
  taskType?: TaskTypeType;
}
```

默认注册（id 稳定，测试锁定顺序）：

| id | order | 来源 |
|---|---|---|
| `product-behavior` | 100 | RFC-006 `buildProductBehaviorPrompt` |
| `user-persona` | 200 | Dormant `personaSystemPrompt` |
| `skills` | 300 | `assembleSkillContext` 的 skill block |
| `plan-advice` | 350 | 规划顾问【规划】【已知】【未知】—— **从 user query 挪到 section** |
| `memory` | 400 | `ContextEngine` 的 `[记忆参考]` |

`assembleSystemPromptPrefix` 保留为默认段的纯函数实现，供单测；生产路径走 registry。`OPENINTJ_PRODUCT_BEHAVIOR=0` 时不注册或 render 为空。

**锁：不得调换 100→400 的相对顺序。** 新段只能插空档（如 350）或 >400。

plan-advisor 今日改 executionInput 字符串。改为 section 后，executionInput 保持用户/`structureUserInput` 文本；模型在 system 里看到规划块。RFC-008 `originalInput` 语义不变。

---

## 4. 与「万物插件」的边界

允许：

- 技能包、MCP（RFC-012）通过 `toolHub.register` / `prompt.section` 扩展
- 测试注入假 LLM / 内存 journal

不允许：

- 用 YAML 替换一行插件配置来换掉 TaoLoop
- 运行时卸载 `plane-governance`
- HMR 热替换平面（Desktop 仍用现有「保存配置 → 重装 assemble」）

这是 **有界工厂**，不是 Cordis 时空组合。

---

## 5. 验收

- CLI/server/desktop 的 `run()` 前置链（dormant → preflight → RFC-008 → 命令表 → classifier → plan-advisor → tao）在 **一个** 模块里，三端测试 spy 同一函数
- 提示词 snapshot 测试：默认四段顺序与 RFC-006 一致；`plan-advice` 在 skills 与 memory 之间
- `OPENINTJ_PRODUCT_BEHAVIOR=0` 行为与现网 A/B 一致
- 抽出装配后 Python parity 与 RFC-006 live 基线不回退（既有 gated harness）
- 包边界：runtime 不依赖 `electron`

---

## 6. 落地顺序

1. 把 `run()` 前置链收到 `@openintj/runtime`（或 core assemble），三端改为调用
2. PromptSectionRegistry 包装现有 prefix + ContextEngine
3. 挪 plan-advisor 出 query
4. 接线 RFC-009 journal factory 与 RFC-010 handle（可同迭代，但 PR 保持可回滚）

---

## 7. 落地记录（2026-11-05）

- 新包 `@openintj/runtime`（不依赖 Electron）：`PromptSectionRegistry` /
  `createDefaultPromptSections`（顺序锁 100/200/300/350/400）、`prepareRunPrelude`
  （preflight → RFC-008 → classifier → plan-advisor，三端共用）、
  `buildJournalTurnContext`。三端 `agent.ts` 的 contextProvider 走段注册表渲染
  前缀，ContextEngine 仍在 400 位追加 `[记忆参考]`；三端 agent 暴露 `prompt`。
- plan-advisor：新增 `formatPlanAdviceBlock`；建议块经 `TaoLoop` 的
  `planAdviceBlock` 按 run 透传到 contextProvider，渲染为 system 段。
  `executionQuery` 不再拼【规划】前缀，Journal 的 user 事件随之不再被规划块污染；
  `formatNextGoalQuery` 保留兼容。`metrics.planAdvice` 改由 prelude 的
  `usedPlanAdvice` 置位。
- 仍保留的宿主差异（profile 差异，不是复制的前置链）：CLI 懒分类器 hydrate、
  server/desktop 的持久化与 TaskPool recovery、desktop 的生图前置、
  artifact 请求跳过分类/规划、工作台 journal sink 与多模型解析。
- 验证：`@openintj/runtime` 单测（段顺序 snapshot、prelude 不污染 query）；
  CLI 断言【规划】出现在 system 且不出现在 journal user 事件；三端既有
  agent/routes/IPC 测试与 `tsc --noEmit` 通过。
