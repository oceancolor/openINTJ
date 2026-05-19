# Phase 3.3 — RFC-003 装配进主 Agent

> 把 `@openintj/concurrency` / `@openintj/dormant` / `@openintj/taskpool` 这三个原本只
> 在 `apps/cli/__tests__/rfc3-integration.spec.ts` 里被独立验证的包，正式接进
> `apps/server` 和 `apps/desktop` 的主 Agent 装配点。
>
> 目标：env / opts 一开关就能用；默认零开销；HTTP 与 IPC 两条接入都覆盖。

---

## 一、三个方向 × 三条入口

| RFC-003 方向 | 装配点 | env 开关 | opts 开关 |
| --- | --- | --- | --- |
| 1 LLM 速率限制 | `RateLimitedLlmClient` 包 `agent.llm` | `OPENINTJ_RATE_LIMIT_QPS` / `OPENINTJ_RATE_LIMIT_BURST` | `opts.rateLimit = { qps, burst? }` |
| 2 混合检索 | `agent.retrieveHybrid()` + `/api/memory?mode=hybrid` / IPC `MEMORY_QUERY{mode:'hybrid'}` | `OPENINTJ_RETRIEVAL_MODE=hybrid` | `opts.retrievalMode = 'hybrid'` |
| 3 蛰伏记忆学习 | `agent.dormant: DormantRuntime` + `/api/dormant/*` + IPC `DORMANT_*` | `OPENINTJ_DORMANT=1` | `opts.enableDormant = true` |

启用规则统一为："opts > env > 默认值"；显式 `false` / 0 / 非数字 等会被视作"不启用"。

---

## 二、新增组件

### 2.1 `@openintj/dormant` 的 `DormantRuntime`

三件套门面，避免每个 app 都自己拼装：

```ts
new DormantRuntime({
  maxPassiveEvents: 10000,              // PassiveStore 容量
  minerOpts: { ngramSize, minFrequency, minConfidence, llmExtract? },
  internalizationOpts: { mapToField? }, // category → PersonaConfig 字段映射
  initialPersona: {...},                 // 启动恢复
  eventIdPrefix: "server",
});
```

接口：

- `record(text, source, metadata?)`：写入 PassiveStore，自动生成 eventId
- `mine()`：触发 PatternMiner → 自动 proposeBatch；返回 `{ patterns, proposals, scannedEvents }`
- `listProposals(status?)`：UI 列表
- `approve(id)` / `reject(id)`：审批闸门
- `snapshot()`：当前 `PersonaConfig`
- `reset(initialPersona?)`：测试或用户清空

`agent.run()` 启用 dormant 后会自动喂事件（user input + final answer），元数据带 `stage` 标签便于追溯。

### 2.2 `@openintj/concurrency` 的 `RateLimitedLlmClient`

```ts
new RateLimitedLlmClient(innerLlm, { qps: 5, burst: 10 });
```

实现要点：

- 内部 TokenBucket（capacity=burst, refillRate=qps）
- `chat / visionChat` 前 `await bucket.acquire(1)`
- `getStatus` 透传，调用方无感
- 额外暴露 `rateLimitStatus()` 调试用

> 之前实现放在 `apps/server/src/rate-limited-llm.ts`，本阶段迁到
> `@openintj/concurrency`；server 端文件仅做向后兼容 re-export。

### 2.3 `retrieveHybrid` 助手

server 在 `apps/server/src/hybrid-retrieve.ts`，desktop 在 `apps/desktop/src/main/agent.ts` 内部，二者实现完全等价：

1. 取 `agent.persistentStore.all`（短期 + 工作 + 长期 fragments）
2. 按 `opts.memoryTypes` / `opts.taskTags` 过滤
3. 临时建 `HybridRetriever<HybridDoc + {memoryType, taskTags, importance}>`
4. 调 `embedder.embed(query)` 拿向量
5. 返回融合分数排序的 hits

设计取舍：

- 不维护持久化的混合索引；每次查询临时构建。中等规模（≤几千 fragments）够用；超大规模建议换 LanceDB 内建 FTS
- 与原有 `MemoryRetriever`（cosine + 朴素 keyword + recency 衰减）并存；用户按需切换

---

## 三、入口

### 3.1 server (HTTP)

| 路由 | 方法 | 说明 |
| --- | --- | --- |
| `/api/memory?q=&topK=&mode=&rrf=` | GET | 检索；`mode=hybrid` 切到 HybridRetriever；`rrf=true` 启 RRF 融合 |
| `/api/dormant/mine` | POST | 触发挖掘 + 生成 proposals |
| `/api/dormant/proposals?status=pending` | GET | 列出 proposals |
| `/api/dormant/proposals/:id/approve` | POST | 审批通过，写入 PersonaConfig |
| `/api/dormant/proposals/:id/reject` | POST | 拒绝 |
| `/api/dormant/persona` | GET | 当前 PersonaConfig 快照 |
| `/api/status` | GET | 含 `retrievalMode` 字段；若启用 dormant 还含 `dormant.{enabled, passiveSize, pendingProposals}` |

未启用 `enableDormant` 时所有 `/api/dormant/*` 一律返回 503 + `{ error: "dormant_not_enabled", hint }`。

### 3.2 desktop (IPC)

新增/扩展 channel：

```ts
IPC.MEMORY_QUERY    // 入参加 { mode?, rrf? }
IPC.DORMANT_MINE
IPC.DORMANT_LIST    // 入参 { status? }
IPC.DORMANT_APPROVE // 入参 { proposalId }
IPC.DORMANT_REJECT  // 入参 { proposalId }
IPC.DORMANT_PERSONA
```

入参用 zod schema 校验（`MemoryQueryRequestSchema` / `DormantListRequestSchema` / `DormantProposalDecisionSchema`）。

未启用 dormant 时所有 `DORMANT_*` 调用返回 `{ error: "dormant_not_enabled", hint }`。

UI 接入留到下一个 phase；本期只把"装配 + IPC + 校验"打通，保证 renderer 可以稳定调用。

---

## 四、文件清单

新增：

- `ts/packages/dormant/src/dormant-runtime.ts`
- `ts/packages/dormant/__tests__/dormant-runtime.spec.ts`
- `ts/packages/concurrency/src/rate-limited-llm.ts`
- `ts/apps/server/src/hybrid-retrieve.ts`
- `ts/apps/server/__tests__/dormant.spec.ts`
- `ts/apps/server/__tests__/hybrid-retrieve.spec.ts`
- `ts/apps/server/__tests__/rate-limited-llm.spec.ts`
- `docs/architecture/phase3-3-rfc3-wiring.md`

修改：

- `ts/packages/dormant/src/index.ts`（导出 DormantRuntime）
- `ts/packages/concurrency/src/index.ts`（导出 RateLimitedLlmClient）
- `ts/apps/server/{src,tsconfig.json,package.json}`：
  - `src/agent.ts` — opts 三连 / status 加字段 / run 喂 dormant / llm 速率限制装饰
  - `src/routes.ts` — `/api/memory` mode 切换；新增 `/api/dormant/*`
  - `src/rate-limited-llm.ts` — 改为兼容 re-export
- `ts/apps/desktop/{src,tsconfig.json,package.json}`：
  - `src/main/agent.ts` — 与 server 端对称的 opts / status / dormant / hybrid 实现
  - `src/main/ipc-handlers.ts` — 注册 dormant channel + memory query mode
  - `src/shared/ipc-protocol.ts` — 扩展 schema + IPC 常量
- `CHANGELOG.md`、`docs/architecture/next-session.md`

---

## 五、自检步骤

```powershell
# CI 模式
$env:NODE_OPTIONS = "--max-old-space-size=6144"
pnpm lint                       # 0 error，2 个 React 历史警告
pnpm exec turbo run typecheck --concurrency=1   # 33/33 OK
pnpm exec turbo run test --concurrency=1        # 312 passed / 7 skipped (e2e 跳过)

# E2E 模式
$env:OPENINTJ_E2E = "1"
pnpm exec turbo run test --concurrency=1        # 全部 318 passed
Remove-Item Env:\OPENINTJ_E2E
```

子项快速验证：

```powershell
# dormant only
pnpm --filter @openintj/server exec vitest run __tests__/dormant.spec.ts
# hybrid only
pnpm --filter @openintj/server exec vitest run __tests__/hybrid-retrieve.spec.ts
# rate-limit only
pnpm --filter @openintj/server exec vitest run __tests__/rate-limited-llm.spec.ts
# desktop ipc (含 RFC-003 装配)
pnpm --filter @openintj/desktop exec vitest run __tests__/ipc-handlers.spec.ts
```

---

## 六、已知边界 / 下一阶段

1. **PassiveStore 不持久化**：进程重启后 dormant 累积的事件丢失。下一阶段考虑用 SQLite 增量落盘 PassiveStore + PersonaConfig 单文件 JSON
2. **PatternMiner 默认 `category: "other"`**：未配 `llmExtract` 时所有 pattern 都是 "other"，`defaultMapToField` 不会落地 PersonaConfig，proposals 会为空。生产部署务必配 `llmExtract` 或使用自定义 `internalizationOpts.mapToField`
3. **HybridRetriever 每次重建索引**：N>10k 时建议改成 LanceDB FTS（一次性建索引、增量更新）
4. **rate-limit 装饰只覆盖 chat / visionChat**：没覆盖未来可能的 `stream` / `embeddings` 入口；按需扩展
5. **UI 未接入**：renderer 端的 dormant 审批 UI / retrieval mode 切换 / rate-limit 显示都尚未实现，留给前端阶段
