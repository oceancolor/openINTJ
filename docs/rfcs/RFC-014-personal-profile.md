# RFC-014：个人档案（记忆控制面）

| 字段 | 值 |
|---|---|
| 状态 | **Accepted** |
| 起草日期 | 2026-11-05 |
| 决策日期 | 2026-11-05 |
| 作者 | OpenINTJ Core |
| 依赖 | RFC-009 Journal、RFC-011 PromptSections、RFC-013 授权（档案属于敏感数据） |
| 后继 | Scheduler（按档案里的目标/时区做主动任务）、连接器（联系人/日历对齐档案） |

---

## 摘要

现有记忆有三套：Journal（情节日志）、MemoryPlane fragments（语义碎片）、
Dormant Persona（推断画像）。碎片和推断都有同一个问题：**用户不能系统地
查看、纠正、删除、导出关于自己的结构化信息**，新旧偏好冲突时也没有裁决规则。

本 RFC 在 Persona 之上加一层 **个人档案（Personal Profile）** 作为记忆控制面：

- 结构化条目：身份 / 偏好 / 目标 / 项目 / 关系 / 约束 / 习惯 / 备注；
- 每条带来源（`user` 显式填写 / `dormant` 推断导入 / `import` 导入）、
  出处、有效期；同 `section + key` 最新者胜，**用户显式条目不被推断覆盖**；
- 档案渲染进 RFC-011 的 `user-persona` 段（200 位），排在 Dormant 推断之前；
- 用户可 CRUD、导出 JSON、按条删除——这是 Personal Agent 的隐私底线。

档案不是第四套记忆：它只存**关于用户的稳定事实**，对话情节仍归 Journal，
可检索碎片仍归 MemoryPlane。Dormant 继续负责「发现」，档案负责「定稿」。

---

## 1. 数据模型

```typescript
type ProfileSection =
  | "identity" | "preferences" | "goals" | "projects"
  | "relationships" | "constraints" | "habits" | "notes";

interface ProfileEntry {
  id: string;
  section: ProfileSection;
  key: string;                 // 如 "name"、"editor"、"current-goal"
  value: string;               // v1 只存文本，结构化值 JSON.stringify 后存
  source: "user" | "dormant" | "import";
  provenance?: string;         // proposalId / 会话 id / 导入来源
  validUntil?: number;         // epoch ms，过期不渲染、不返回
  createdAt: number;
  updatedAt: number;
}
```

- 冲突裁决：同 `(section, key)` 只保留一条；写入时 `source=user` 永远胜过
  已有 `dormant` 条目，反之推断导入不得覆盖 `user` 条目（跳过并保留原值）。
- `importPersona(persona)`：把 Dormant PersonaConfig 的
  preferences/phrases/habits/context 幂等导入为 `dormant` 条目，渲染前调用，
  保证推断与档案不双写漂移。

## 2. 存储

`ProfileStore` 接口 + 两个实现：

- `InMemoryProfileStore`（CLI、测试）；
- `JsonProfileStore`（server/desktop：`<dataDir>/profile.json`，写时落盘）。
  v1 用单文件 JSON——档案量级是百条级，够用且天然可导出；量级涨了再换 SQLite，
  接口不变。

## 3. 宿主面

| 宿主 | 存储 | 入口 |
|---|---|---|
| Desktop | `profile.json`（userData 数据目录） | IPC `profile.list/upsert/remove/export` + 「档案」页 |
| Server | `profile.json`（dataDir）/ 内存 | `GET/POST/DELETE /api/profile`、`GET /api/profile/export` |
| CLI | 内存 | `agent.profile` 编程面 |

三端 agent 统一暴露 `profile`（ProfileService），与 `prompt`/`approvals` 并列。
档案默认启用（空档案渲染为空，零行为变化）；`OPENINTJ_PROFILE=0` 可关。

## 4. 验收

- 单测：同 key 冲突裁决（user 胜 dormant）、过期过滤、导入幂等、
  JSON store 往返、渲染分组与空档案为空。
- 接线：档案条目出现在 system 的 user-persona 段且排在 Dormant 推断之前；
  删除后下一轮不再出现。
- 桌面契约：profile IPC CRUD + export；renderer 可新增/删除条目。

## 5. 不做

- 不做档案变更历史/版本回滚（v1 只保留最新；Journal 仍是对话事实源）。
- 不做跨设备同步（P2）。
- 不把 MemoryPlane fragments 自动升格为档案条目——升格必须经用户或 Dormant 审批。

## 6. 落地记录（2026-11-05）

- 新包 `@openintj/profile`（零运行时依赖）：`types` / `store`
  （`InMemoryProfileStore`、`JsonProfileStore` 写 `<dataDir>/profile.json`）/
  `service`（冲突裁决、过期过滤、`importPersona`、分组渲染、导出/导入）。
  6 个单测全过。
- 删除语义补强：删除 `source=dormant` 条目会写入 dismissed 墓碑集（随 JSON
  持久化），`importPersona` 跳过墓碑 key——「删掉的推断不回潮」是硬保证；
  用户之后仍可手动写入同 key。Dormant 侧 revoke 不自动同步删除档案镜像，
  用户在档案页删除即可（v1 已知取舍）。
- 三端接线：persona provider 改为「profile 启用时档案接管 user-persona 段
  （渲染前 `importPersona(dormant.snapshot())`），禁用时回退原始 persona 注入」；
  三端 agent 统一暴露 `profile`，server/desktop real 模式用 JSON store，
  `OPENINTJ_PROFILE=0` 关闭。旧断言 `[用户画像]` 的三处顺序/注入测试已按新
  契约改断 `[个人档案]` + `key=value`（CLI / server / desktop 各一处）。
- 宿主面：桌面 IPC `profile.list/upsert/remove/export` + preload 四 API +
  renderer「档案」页（按分区列表、新增、删除、导出 JSON）；server
  `GET/POST/DELETE /api/profile` + `GET /api/profile/export`（zod 校验 section）。
- 验证：profile 6 / CLI 44 / server 68 / desktop 99 测试全过；四包
  `tsc --noEmit` 全过；biome check 已跑。
