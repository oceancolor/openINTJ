# Phase 3.5 — Dormant 审批 UI（#9.B）收尾报告

> 更新时间：2026-05-19
> 仓库标签：`v3.0.0-alpha.5`
> 前序：[Phase 3.4 Dormant 持久化](./phase3-4-dormant-persistence.md)

## 一、目标

Phase 3.4 已经把 Dormant 持久化的"模型层"和"装配层"都做完了：

- `DormantPersistenceAdapter` / `SqliteDormantStore` / `DormantRuntime.hydrate()`
- `server` / `desktop` agent 自动挂 adapter + close 顺序
- 5 个 IPC channel (`DORMANT_MINE/LIST/APPROVE/REJECT/PERSONA`) 在主进程已注册

但桌面端 renderer **看不见** —— preload 没暴露 dormant 方法，UI 也没有审批入口。
Phase 3.5 收尾这最后一公里。

非目标：

- 不做 server 端 web UI（only Electron renderer）
- 不做 proposal 通知推送（用户主动打开 Dormant tab 就能看见 pending 数字角标，足够）
- 不做撤销已审批的 proposal（用户重新 mine 一次即可）

## 二、设计要点

### 2.1 协议层补全（`apps/desktop/src/shared/ipc-protocol.ts`）

**StatusResponseSchema** 补三个 optional 字段（与 main 进程 `agent.status()` 实际返回对齐）：

- `persistence?: { mode: 'memory' | 'real', dataDir?: string }`
- `retrievalMode?: 'vector' | 'hybrid'`
- `dormant?: { enabled: true, passiveSize, pendingProposals, persistence? }`

**Dormant 响应 DTO** 五件套：

```
DormantMineResponseSchema       (scannedEvents, patterns[], proposals[])
DormantListResponseSchema       (total, proposals[])
DormantDecisionResponseSchema   (proposalId, status, decidedAt?)
DormantPersonaResponseSchema    (preferences, phrases, habits, context, meta)
DormantProposalDtoSchema        (单条 proposal 在 IPC 上的精简形式)
```

**错误 schema** 两件：

- `DormantErrorSchema`：`{ error: 'dormant_not_enabled' }`（5 个 channel 都可能返）
- `DormantDecisionErrorSchema`：`{ error: 'not_found_or_already_decided' | 'invalid_request' }`

### 2.2 preload 暴露

`apps/desktop/src/preload/index.ts` 新增 5 个方法：

```ts
api.dormantMine()           → Promise<DormantMineResponse | DormantError>
api.dormantList(req?)       → Promise<DormantListResponse | DormantError>
api.dormantApprove({ id })  → Promise<DormantDecisionResponse | DormantDecisionError>
api.dormantReject({ id })   → Promise<DormantDecisionResponse | DormantDecisionError>
api.dormantPersona()        → Promise<DormantPersonaResponse | DormantError>
```

返回联合类型让 renderer 必须 narrow（`'error' in r`），无法把错误当数据用。

### 2.3 UI 结构

**App.tsx 布局**：从"chat | trajectory"两栏变成"chat | (tab: trajectory / dormant)"两栏单 tab：

```
┌──────────────────────────────────────┬─────────────────────────────┐
│ header                                                              │
├──────────────────────────────────────┼─────────────────────────────┤
│                                       │ [推理轨迹] [Dormant 🟡 N]   │
│           ChatPanel                   ├─────────────────────────────┤
│                                       │                             │
│                                       │   TrajectoryPanel /         │
│                                       │   DormantPanel              │
│                                       │                             │
├──────────────────────────────────────┴─────────────────────────────┤
│ StatusBar: LLM · 记忆 · 审计 · 检索 · 盘 · Dormant: N ev / M 待审 · 工具
└────────────────────────────────────────────────────────────────────┘
```

tab 角标：当 `status.dormant.pendingProposals > 0` 时显示黄色数字角标。

**DormantPanel.tsx** 内部结构（顶到底）：

1. **顶栏**：[Mine] 按钮（触发后端 mine，弹出本次扫描摘要）
2. **状态 filter**：pending / applied / rejected / all（默认 pending）
3. **错误条**（条件渲染）：last error message
4. **Mine 摘要条**（条件渲染）："扫描 N 事件 · M pattern · K proposals"
5. **列表**：每条 proposal 一张卡片
   - 状态徽章（yellow/green/red/blue）+ 频次 + 置信度 + 时间戳
   - pattern 描述（人类可读）
   - `targetField ← value` 一行（视觉对齐）
   - pending 时显示 [✓ 应用] [✗ 拒绝] 按钮
   - 已决策时显示决策时间
6. **底部折叠区**："当前 Persona"（点击展开 JSON）

**未启用态**：`status.dormant === undefined` 时面板显示居中提示，告诉用户加 `OPENINTJ_DORMANT=1` 或 `enableDormant: true`。

### 2.4 类型对齐

`StatusBar.tsx` 从本地定义的 `StatusSnapshot` 改为 `type StatusSnapshot = StatusResponse`（来自 protocol）。
这是 **Phase 3.1/3.3/3.4 累积的小漂移** —— main 进程一直返回 persistence / retrievalMode / dormant，但 renderer 端的类型早就过期，编辑器只是没报错而已（因为是结构性子集）。这一版顺便统一掉。

### 2.5 已知不做（留尾）

| 编号 | 项 | 理由 |
|---|---|---|
| L1 | dormant 推送通知 | 当 mine 在后台跑（未来 hooks 触发）时可以 send `EVT_DORMANT` 让 renderer 实时刷新；但当前 mine 只由 UI 主动触发，无需 |
| L2 | proposal 详情抽屉 | 现在卡片已经把所有关键信息塞进去了；如果以后 pattern.evidenceIds 也要展示，再单独做抽屉 |
| L3 | mine 任务进度条 | mine 通常 100ms 内完成；除非接入慢 LLM，才需要进度反馈 |
| L4 | persona 字段级编辑 | 当前 persona 是只读 JSON；用户唯一的写入路径是审批 proposals。直接编辑等于绕过审批闸门，不在本 phase 范围 |

## 三、测试矩阵

| 测试 | 数量 | 类型 |
|---|---:|---|
| `apps/desktop/__tests__/ipc-handlers.spec.ts` | 12 → 18 | 新增 6：StatusResponseSchema 校验 / DORMANT_MINE schema / DORMANT_LIST 默认 status / DORMANT_REJECT 不污染 persona / decide ghost id / persona 未启用错误 |

**为什么没有 renderer 单测**：当前 desktop 工作区只配了 main-process vitest（无 jsdom、无 @testing-library/react）。引入 React 测试栈成本不小，
而 DormantPanel 的逻辑分支已经被 IPC 层的契约测试基本覆盖；UI 渲染留给手动 / 未来 Playwright e2e（#4）。

## 四、自检脚本

```powershell
cd F:\openINTJ\ts
pnpm lint                                            # exit 0
pnpm exec turbo run typecheck --concurrency=1        # 33/33
pnpm exec turbo run test --concurrency=1             # CI 模式 366+11
$env:OPENINTJ_E2E="1"
pnpm exec turbo run test --concurrency=1             # 377/0
Remove-Item env:OPENINTJ_E2E

# 想本地真启动桌面端体验 Dormant 面板（需要 OPENINTJ_DORMANT=1 才看得见数据）
$env:OPENINTJ_DORMANT="1"
pnpm --filter @openintj/desktop dev
Remove-Item env:OPENINTJ_DORMANT
```

## 五、下一站候选

- ⭐⭐⭐ **#1 行为对齐测试**（Python v2 ↔ TS）
- ⭐⭐ **#3 嵌入基准** / **#4 Playwright Desktop E2E**（顺手能覆盖本 phase 的 UI）
- ⭐⭐ **#11 dormant 事件清理**（Phase 3.4 留尾）
- ⭐ **#6 打包发布** / **#7 OpenTelemetry**
