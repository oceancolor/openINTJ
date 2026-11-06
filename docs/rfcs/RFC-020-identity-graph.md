# RFC-020 身份打通（Identity Graph）

## 背景

用户希望「给我一个邮箱 / 手机号 / 微信号，拉起我在各个应用、社区、工具上的
全部身份」。真实世界里这件事的边界必须先说清：

- **不存在合法的手机号 → 全网账号反查**。任何声称能做的服务都依赖灰产数据，
  本项目不接。手机号只能在**本机已授权语料**里关联（邮件签名、日历与会者、
  笔记文本里写过的号码）。
- **微信号同理**：没有公开反查 API。微信号只能命中本机语料 + 用户手动断言。
- **邮箱是唯一有公共面的种子**：Gravatar 提供公开的邮箱哈希查询
  （`api.gravatar.com/v3/profiles/{sha256}`，无需密钥），能返回头像、显示名、
  已验证的第三方账号与主页链接。

所以 v1 的诚实定位是：**以种子为锚的本机身份图谱** —— 公开面能查的查（邮箱
→ Gravatar），本机语料里能关联的关联，用户确认后才入图；查不到的允许手动
登记。绝不爬虫、绝不自动合并、绝不把「查不到」伪装成「没有」。

## 数据模型（`@openintj/identity`）

```ts
type IdentitySeedKind = "email" | "phone" | "wechat" | "username" | "url";
type LinkedAccountSource = "gravatar" | "mail" | "calendar" | "notes" | "manual";

interface LinkedAccount {
  id: string;
  platform: string;            // "gravatar" | "github" | "mail" | ...
  kind: IdentitySeedKind;
  value: string;               // 归一化后的值
  source: LinkedAccountSource;
  evidence: string;            // 一句可读证据（来自哪份语料 / Gravatar verified）
  confidence: number;          // 0..1
  verifiedAt: number | null;   // 公共面验证时间；手动断言为 null
  createdAt: number;
}

interface IdentityPerson {
  id: string;                  // 规范人（canonical），一个真人一条
  displayName: string;
  accounts: readonly LinkedAccount[];
  createdAt: number;
  updatedAt: number;
}

interface IdentityProposal {   // 候选，未经确认不得入图
  id: string;
  personId: string | null;     // 建议归属；null = 新建规范人
  displayName?: string;
  account: Omit<LinkedAccount, "id" | "createdAt">;
}
```

## Resolver

```ts
interface IdentityResolver {
  readonly id: string;
  resolve(seed: IdentitySeed, ctx: IdentityResolveContext):
    Promise<readonly IdentityProposal[]>;
}
```

v1 实现：

1. **GravatarResolver**（仅 email 种子）：`sha256(trim(lower(email)))` →
   Gravatar v3 profile。self 账号置信 0.95（`verifiedAt` 打标）；`accounts`
   里的 verified 第三方账号置信 0.9；`links` 置信 0.6。fetch 可注入，离线 /
   404 → 空候选（不报错冒充无结果，由服务层报告 resolver 状态）。
2. **LocalCorpusResolver**：消费宿主注入的只读语料
   `documents(): {source, title, text}[]`（邮件正文+签名、日历与会者行、笔记
   文本）。以种子为锚做精确匹配：种子是邮箱 → 找同文档中出现的手机号 /
   微信号 / @用户名；种子是手机/微信 → 找同文档中的邮箱与账号句柄。证据 =
   文档标题 + 命中片段（截断）。置信：同文档同现 0.6。
3. **手动断言**：不经 resolver，用户直接 `assertAccount`（source=`manual`，
   置信 1.0 但 `verifiedAt=null` —— 是「我声明」不是「已验证」）。

## 服务与不变量

`IdentityService`：`resolve(seed)` 跑全部 resolver 汇总候选；`confirm(ids)`
把候选落成 `LinkedAccount` 挂到建议的规范人（或新建）；`assertAccount` /
`unlink(accountId)` / `listPersons()`。

不变量：

- **提案-确认制**：任何 resolver 结果都不能绕过 `confirm` 进图谱。
- **唯一归属**：同一 `(platform, value)` 已挂在其他规范人名下时，
  `confirm` 返回 `conflict` 错误，绝不静默搬家；用户先解绑再确认。
- **自我种子优先**：图谱里第一个规范人默认是「本人」；profile 的 identity
  段可引用其主邮箱（本批不自动回写，避免双向同步复杂度）。

## 存储

`JsonIdentityStore`：`<dataDir>/identity.json` 整文件同步写，损坏 → 空图；
`InMemoryIdentityStore` 供测试 / 无持久模式。宿主有 `dataDir` 时自动注入。

## 宿主

- 三端 agent 均暴露 `identity` 服务（CLI 仅服务层）。
- server：`GET /api/identity/persons`、`POST /api/identity/resolve`、
  `POST /api/identity/confirm`、`POST /api/identity/assert`、
  `POST /api/identity/unlink`。
- desktop：`identity.*` IPC + preload + 右栏新增「身份」tab：输入种子 →
  解析候选（显示来源/证据/置信度）→ 勾选确认 → 规范人列表（可解绑）。

## 非目标

- 各平台 OAuth 登录 / 授权抓取（需要逐平台应用注册，后续 RFC）。
- 任何手机号 / 微信号「反查」服务、爬虫、灰产数据。
- 跨用户图谱、推荐、自动合并。
- 同步协议：identity.json 可随 RFC-019 bundle 手动携带，本批不做增量同步。

## 隐私

全部数据只存本机 `<dataDir>`；唯一出网是 Gravatar（只发邮箱 SHA-256 哈希，
不发明文邮箱给第三方之外的任何人）；本机语料不出机。

## 落地记录（2026-11-06）

已实现并通过测试：

- `@openintj/identity` 包：种子归一化/识别、Gravatar resolver（fetch 可注入，
  404 = 无公开资料）、本机语料 resolver（同文档共现抽取，邮箱内嵌 `@` 不再
  误报用户名）、`IdentityService`（resolve 只出候选；confirm 冲突抛
  `identity_conflict`；assert/unlink）、`JsonIdentityStore`（损坏文件读空图）。
- 宿主：三端 agent 暴露 `identity` 服务（real 持久化自动用
  `<dataDir>/identity.json`）；语料由宿主用 connectors 的
  `parseMbox`/`parseIcs` + notes 目录按 env 路径构建，只读不出进程。
- server：`/api/identity/persons|resolve|confirm|assert|unlink`（冲突 409）。
- desktop：`identity.*` IPC + preload + 右栏新增「身份」tab（种子解析 →
  候选勾选确认 → 图谱列表/解绑 + 手动登记表单）。
- 测试：identity 10 · CLI 54 · server 76 · desktop 104，类型检查全绿。

遗留（后续 RFC）：各平台 OAuth 授权抓取、Gravatar 之外的公开面（如 keybase proofs）。

## 落地记录（2026-11-06 第二批：回写 + 同步）

用户决策：主身份回写档案 + identity 纳入同步包（OAuth 与更多公开面不选）。

- `IdentityService` 新增 `onGraphChanged` 回调（confirm/assert/unlink/
  replacePersons 后触发）与纯函数 `derivePrimaryIdentity`（最高置信邮箱 +
  人类显示名；自动派生的名字不回写）。
- 三端宿主回写：`identity` 段 `email`/`name` 两键，`source:"import"` +
  provenance `identity-graph`；用户手写（source `user`）条目永不被覆盖，
  图谱侧条目随解绑自动清理。
- `@openintj/sync`：bundle 增加可选 `identity` 段（旧 bundle 兼容），按
  person id 并集、account id 并集合并（v1 语义：既有记录不被改写），
  报告新增 `identityPersonsAdded`/`identityAccountsAdded`；三端
  export/import 均已携带 identity（导入经 `replacePersons` 触发回写）。
- 测试：identity 11 · sync 7 · CLI 54 · server 77 · desktop 108，类型全绿。
