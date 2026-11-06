# RFC-013：交互审批与能力授权（治理）

| 字段 | 值 |
|---|---|
| 状态 | **Accepted** |
| 起草日期 | 2026-11-05 |
| 决策日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 依赖 | RFC-002 HookBus、RFC-004 IPC、RFC-010 AgentHandle、RFC-011 Runtime |
| 后继 | RFC-014 个人档案、连接器框架（可写连接器必须先过本 RFC） |

---

## 摘要

Personal Agent 要碰真实数据（日历、邮箱、文件、命令），前提不是模型更聪明，
而是**不可逆动作有人把关、且授权可限定范围、可撤销**。本 RFC 在现有
GovernancePlane + ToolHub gate 之上补两件事：

1. **交互审批**：需要审批的工具调用暂停在 gate 上，等人批准/拒绝再继续。
2. **能力授权（CapabilityGrant）**：把「总是允许」从口头承诺变成结构化、
   带范围与有效期、可撤销的记录；命中授权的调用不再逐次打扰。

不新增执行路径：审批仍走 `ToolHub.call → gate → GovernancePlane.checkToolCall`，
拒绝仍是不触发熔断的终态失败（与现有治理拒绝语义一致）。

---

## 1. 风险分级

按工具描述的 `permissions` 推导，无声明时按工具名兜底：

| 风险 | 判定 | 例 |
|---|---|---|
| `read` | `filesystem.read` / `network.read` / idempotent 只读 | `read_file`、`search` |
| `write` | `filesystem.write` / `network.write` | `write_file`、未来的 `mail.send` |
| `execute` | `system.execute` | `execute_command` |
| `external` | 跨出本机的不可逆外发（连接器声明） | 发消息、发布、付款 |

宿主用 `requiredRisks` 决定哪些级默认要审批（典型：`["write","execute","external"]`）。
默认空集 = 行为与今日完全一致，避免存量测试与工作流回退；桌面主进程显式开启。

## 2. 能力授权

```typescript
interface CapabilityGrant {
  id: string;
  tool: string;          // 工具名或 "*"
  risk: ToolRisk;        // 该授权覆盖的最高风险级
  scope?: string;        // write 看 params.path 前缀；execute 看命令首 token
  expiresAt?: number;    // epoch ms
  createdAt: number;
  label?: string;
}
```

- 授权存储接口 `GrantStore`（默认内存实现；2026-11-06 补 `JsonGrantStore` 文件实现，见 §8）。
- 审批通过时可附带「顺便给一个授权」（scope + 有效期），后续同范围调用免审批。
- `revoke(id)` 立即生效；过期授权视为不存在。

## 3. 交互审批

```typescript
interface ApprovalRequest {
  id: string; tool: string; risk: ToolRisk;
  params: Record<string, unknown>; requestedAt: number;
}
type ApprovalOutcome =
  | { decision: "approved"; grant?: { scope?: string; ttlMs?: number } }
  | { decision: "denied"; reason?: string };

interface ApprovalBroker {
  request(req: ApprovalRequest): Promise<ApprovalOutcome>;
}
```

- `PendingApprovalBroker`：无宿主接管时 **fail-closed（拒绝）**；宿主通过
  `decide(id, outcome)` 兑现 pending 请求（桌面 IPC / server HTTP / 测试注入）。
- 审批等待必须可取消：turn 的 `AbortSignal` 取消时，pending 请求以 denied 结算。
  本期通过 gate 抛错实现——ToolHub 已把 gate 拒绝转成终态失败结果；signal 在
  ToolHub 入口与工具执行层已有传播，审批超时/取消由 broker 的 `cancel(id)` 兜底。
- 事件：`event.APPROVAL_REQUESTED` / `event.APPROVAL_DECIDED` 进 HookBus，
  桌面经 IPC 推送，审计轨迹同步记录（allowed with `approved:true` / blocked）。

## 4. 宿主面

| 宿主 | 启用 | 审批通道 |
|---|---|---|
| Desktop | 主进程默认开 `write/execute`（可关） | `EVT_APPROVAL` 推送 + `approval.decide/list`、`grants.list/grant/revoke` IPC |
| Server | `OPENINTJ_APPROVAL_RISKS=write,execute` | `GET /api/approvals`、`POST /api/approvals/:id/decide`、`/api/grants` |
| CLI | opts/env 显式开 | 注入 broker；默认 PendingApprovalBroker（无人 decide 即拒绝） |

三端 agent 统一暴露 `approvals`（broker）与 `grants`（store），与 RFC-011 的
`prompt`/`commands`/`handle` 并列。

## 5. 验收

- 治理单测：无 broker 需审批 → 拒绝且审计 blocked；批准 → 执行且审计留痕；
  授权命中 → 不再请求审批；授权过期/撤销 → 恢复审批；只读工具永不触发审批（除非配置）。
- ToolHub 回归：审批拒绝不触发熔断、不发 `tool.onError`（沿用 gate 语义测试）。
- 桌面契约测试：pending 审批可经 IPC decide 后工具继续；grant 列表可查可撤。
- 默认关闭时三端既有测试全绿（零行为变化）。

## 6. 不做

- 不做跨进程审批队列持久化（进程重启 pending 即失效，按 denied 处理）。
- 不做细粒度字段级授权（如「只能发给某联系人」）——scope 先支持路径/命令前缀。
- 不改 PolicyEngine 的黑名单/白名单语义；审批是其后的第二道门。

---

## 7. 落地记录（2026-11-05）

- 治理平面新增 `approval.ts`：`classifyToolRisk`（permissions → read/write/execute/external）、
  `InMemoryGrantStore`（scope 前缀 + 过期 + 撤销）、`PendingApprovalBroker`
  （fail-closed、decide/cancel）、`createApprovalSetup`（opts > env
  `OPENINTJ_APPROVAL_RISKS` > 关）。`checkToolCall` 在策略后加第二道门，拒绝走既有
  POLICY_BLOCKED 审计路径，不触发熔断；HookBus 新增 `event.APPROVAL_REQUESTED/DECIDED`。
- 三端 agent 新增 opts `approvalRisks`/`approvalBroker`，统一暴露 `approvals`/`grants`。
  桌面主进程默认开启 write/execute（`OPENINTJ_APPROVAL=0` 关）；IPC 新增
  `approval.list/decide`、`grants.list/grant/revoke`、`EVT_APPROVAL`，renderer 新增
  `ApprovalBanner`（批准一次 / 批准并授权 1 小时 / 拒绝）。server 新增
  `/api/approvals`、`/api/grants`。授权当时为进程内存储，重启失效（2026-11-06 补齐，见 §7）。
- 验证：治理单测 6（fail-closed、批准/拒绝、grant 命中/撤销/过期、审批铸 grant、gate 风险透传）；
  CLI 端到端写文件被拒与 grant 放行；桌面 IPC grants 契约；三端既有测试全绿。

## 8. 落地记录（2026-11-06，grants 持久化）

- `CapabilityGrant` 新增 `source: "approval" | "manual"`（缺省 manual，向后兼容）；
  治理层由审批 outcome 铸造的 grant 标记 `approval`。
- 治理包新增 `JsonGrantStore`（`<dataDir>/grants.json`，镜像 profile/scheduler 的
  整文件同步写；损坏起空、加载丢弃过期、写入时 prune）。**两级规则**：只有显式
  （manual）grant 落盘，审批派生 grant 只留会话内——一次批准不跨重启变永久授权。
- 宿主：server/desktop 在 real 持久化模式自动注入（无需开关），三端新增
  `opts.grants` 注入点；CLI 无 dataDir，保持内存（可注入）。pending 审批队列仍不
  持久化（§6 不变，重启按 denied 处理）。
- 验证：治理单测新增 4（跨实例读回、审批派生不落盘、撤销/过期、损坏文件起空+
  审批铸 grant 带 source 标记）；server gated e2e 重装配读回 + `/api/grants` 契约；
  CLI 注入存储落盘读回；三端既有测试全绿。
