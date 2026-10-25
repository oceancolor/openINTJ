# Phase 7 首版验收报告 · 壳层与进程骨架

- 日期：2026-10-22
- 版本：openINTJ v0.1.0-alpha
- 验收范围：用户裁决的「首版重心 —— 壳层与进程骨架」
- 结论：**通过**。三进程链路、风险分级、执行闸门、事件回流四项均以实测输出为准，非「能起来就算过」
- **增补（2026-10-22 第二轮）**：用户选择「补安全底线 + 持久化」，D10 与 D1 已关闭，见第六节
- **增补（2026-10-22 第三轮）**：用户指派「立即跑 D4」，审批卡片与发起方已落地，见第七节

---

## 一、验收判据与实测结果

判据取自《系统设计》§3.1（进程模型）与 §3.2.M10（审批），不是自拟标准。

### 1. 三进程链路（MessagePort，不开本地端口）

```
[selfcheck] P-01 主进程 online
[selfcheck] P-03 agent-host agent-host online (pid=14680, runs=0)
[selfcheck] P-04 tool-host tool-host online (pid=14248)
[selfcheck] 三进程链路就绪（MessagePort，未开本地端口）
```

判据是**跨进程请求拿到返回值**，不是「进程被拉起来了」——子进程可能已 spawn 但端口未绑定。
`ping` 返回里带真实 `pid`，证明数据确实穿过了 `MessageChannelMain`。全应用未监听任何 TCP 端口。

### 2. 风险分级（R4-01 命中即 L3）

```
[probe] 分级 rm -rf → L3（R4-01）
```

### 3. 执行闸门（未获批准不产生进程）

```
[probe] 拦截生效：TOOL_HOST_ERROR: 拒绝执行：裁决结果为 PER_CALL_PROMPT（风险 L3），未获批准不产生进程
```

拒绝发生在 `spawn` 之前，不是「启动了再杀」。

### 4. 已批准执行（正例）

```
[probe] 已批准执行 node -v → exit=0 stdout=v22.22.2
```

`shell: false` + 最小环境变量下仍能正确解析 PATH，说明不依赖用户完整环境。

### 5. 界面实际渲染（读的是屏幕上真有的字）

```
openINTJ
Agent Console · v0.1.0
win32
保险箱 可用
隔离降级
Windows 侧采用「审批在环 + 命令白名单 + 工作区根目录约束」的组合替代 OS 级沙箱。隔离强度弱于 macOS，不等价。world-writable 目录下的写入无法被阻止。
P-01 主进程   在线
P-03 agent-host   在线   agent-host online (pid=15864, runs=0)
P-04 tool-host   在线   tool-host online (pid=18296)
审批档位（IC-08 · 按风险分级）  严格 / 标准 / 宽松
沙箱原语   windows-degraded
```

C1 必显告警条按原文展示，且无关闭入口；「宽松」档在 Windows 上不可选。

### 6. 完整闭环（渲染层 → 主进程 → agent-host → 事件回流）

```
[probe] run 已启动 → {"runId":"01M5FKZK7KVA732YVYVZP361VR","state":"CONTRACT_PENDING"}
[probe] 事件已回流渲染层 → GoalContractDrafted
```

`runId` 为 ULID；`state` 停在 `CONTRACT_PENDING` 而非直接 `RUNNING`，即 IC-07 硬闸门在状态机上成立。

---

## 二、代码落点

| 文件 | 职责 | 对应文档 |
| --- | --- | --- |
| `src/shared/events.ts` | 20 项 ActionType / 7 项 ObservationType 封闭词表、`EventEnvelope`（ULID + seq + traceId + tenantId + redaction + schemaVersion）、9 态任务机、4 类检查点作用域 | §3.2 |
| `src/shared/approval.ts` | 三档 × 六风险等级真值表、`RiskDecisionContext` 六要素、平台策略（Windows 禁宽松档） | §3.2.M10、IC-08 |
| `src/shared/ipc.ts` | `WireEnvelope` 四态、`BootInfo`、`AGENT_METHODS`/`TOOL_METHODS`、事件通道名 | §3.1 |
| `src/main/process-manager.ts` | `WorkerHost`：`utilityProcess.fork` + `MessageChannelMain`、requestId 关联、超时 reject、崩溃上报 | §3.1.1.1 |
| `src/main/index.ts` | 8 条 IPC 路由、`detectSandbox`（三平台）、档位切换收紧时撤销会话授权、保险箱探针 fail-stop、启动自检与冒烟 | §3.1、C1 |
| `src/agent-host/*` | 单调 ULID、内存事件存储（seq 单调 + 检查点位点）、run 状态机 | §4 |
| `src/tool-host/classifier.ts` | R4-01~R4-08 危险规则、R1-01~R1-05 只读白名单、R5 出网、R0 无法归类按最严 | §3.2.M10 |
| `src/tool-host/index.ts` | `spawn shell:false`、最小环境、512KB 截断、超时 SIGKILL、裁决校验 | US-08 AC-3 |
| `src/preload/index.ts` | `contextBridge` 窄接口，不暴露 `ipcRenderer` 本体 | §3.1 |
| `src/renderer/*` | `#030014` 深空底 + 三色极光 + 玻璃卡片，承接设计资产视觉系统 | 设计产出物 |
| `src/shared/llm.ts` | 三协议归一后的 `UnifiedMessage` / `ContentPart` / `ToolSpec` / `TokenUsage`，用量口径约定，协议判定的唯一入口 `parseProtocol()` | §3.2.M11 |
| `src/shared/contract.ts` | `GoalContract` 六字段、`extractContract`（容忍围栏，抠不出就如实失败） | IC-07 |
| `src/shared/approval-token.ts` | HMAC 票据 `issueTicket` / `verifyTicket`，绑定 runId + 命令哈希 + 风险等级 | D10 |
| `src/shared/schema.ts` | 四表 DDL + `applyMigrations`（`PRAGMA table_info` 判列后 `ALTER`，不删库） | D1 |
| `src/agent-host/llm/anthropic.ts` | `/v1/messages` 原生实现，丢弃 thinking 以保前缀缓存，400 且含 thinking 时去参重试 | §3.2.M11 |
| `src/agent-host/llm/openai.ts` | `/v1/chat/completions` 原生实现，`tool_result` 拆 `role:'tool'`，usage 减 `cached_tokens` | §3.2.M11 |
| `src/agent-host/tool-specs.ts` | `run_command` / `finish` 两个工具的声明与系统提示 | D3 |
| `src/agent-host/agent-loop.ts` | 模型驱动的多步循环：`MAX_STEPS=12`、`finish` 唯一出口、结果原样回灌 | D3 |
| `src/tool-host/audit-log.ts` | 六要素齐全才落库，缺任一抛 B100003 | D1、D10 |
| `src/main/credential-store.ts` | `safeStorage` 加密落盘、掩码、解密失败与文件缺失分开报、不可用即 fail-stop | IC-06 |
| `src/renderer/SettingsPage.tsx` | 设置与凭据中心页：只显示掩码、显示当前生效来源、测试连接、保存/清除 | IC-06 |

---

## 三、首版与设计文档的偏离（诚实清单）

以下不是「bug」，是首版主动收敛的范围；每一项都写明了补齐路径。

| # | 偏离 | 影响 | 补齐路径 |
| --- | --- | --- | --- |
| ~~D1~~ | ~~事件存储为内存态，进程重启即丢~~ | — | **已关闭，见第六节** |
| ~~D2~~ | ~~命令执行用 `child_process`，未上 `node-pty`~~ | — | **已关闭，见第十三节**（命令执行仍用 `child_process`，终端另起 PTY 通道） |
| ~~D3~~ | ~~无 LLM 适配层~~ | — | **已关闭，见第八节** |
| ~~D4~~ | ~~审批 UI 只有档位切换，无审批卡片~~ | — | **已关闭，见第七节** |
| D5 | 未内嵌 VSCodium | T2 方案 B 未落地 | 下载发行版 + Open VSX |
| ~~D6~~ | ~~未接 MCP~~ | — | **已关闭，见第十节** |
| D7 | 云端三项未做 | 无云端工作区 / 沙箱 / 消息通道 | 待 Q4-a（腾讯云账号）、Q4-b（实例规格） |
| ~~D8~~ | ~~事件只标注 `redaction` 字段，无实际脱敏器~~ | — | **已关闭，见第十节** |
| ~~D9~~ | ~~档位未落库，重启回落「标准」~~ | — | **已关闭，见第十节** |
| ~~D10~~ | ~~裁决结果由调用方声明~~ | — | **已关闭，见第六节** |

### 曾存在的结构性风险（D10）—— 已关闭

**（记录于此以保留判断依据；修复方案与实测见第六节。）**

首版 `ToolExecParams.risk.outcome` 由调用方（当时只有主进程）填入，tool-host 只校验它是否属于允许值。
当前调用链只有主进程，主进程可信，故当时不构成实际漏洞。

但 LLM 接线后 `agent-host` 会调用 tool-host，而 agent-host 运行的是模型产出的动作——
**任何「Agent 自己声明 outcome=AUTO_APPROVE」的实现都会把闸门变成摆设。**

修正方向：把裁决权收回 `tool-host` —— 调用方只传命令与 `cwd`，
tool-host 自己查真值表 + 会话授权得出 outcome 并落审计；跨进程不得传「已批准」这种结论。

---

## 四、首版修复记录

| 问题 | 现象 | 处置 |
| --- | --- | --- |
| preload 类型导入错位 | `ApprovalRequest` 从 `shared/ipc` 导入，但该类型定义在 `shared/approval` | 改导入来源 |
| tool-host 写-only 变量 | `outPort` 只赋值不读，`noUnusedLocals` 报错 | 移除 |
| 风险等级命名误导 | `L0-unknown` 里的 `L0` 意为「只读」，与「无法归类」语义冲突，审计日志易被误读为低危 | 统一改为 `R0-unknown` / `R5-network` |
| Electron 二进制下载超时 | 走 GitHub，9 分 36 秒未完成 | 加 `.npmrc` 指向 npmmirror，重取成功（33.4.11） |
| `ELECTRON_RUN_AS_NODE=1` 环境注入 | 直接 `electron .` 报 `Cannot read properties of undefined (reading 'whenReady')`，状似代码 bug | 新增 `scripts/run.mjs` 统一启动器剔除该变量；README 写明 |

---

## 五、下一步建议顺序

> 以下顺序是首版当时的判断，保留于此以留痕。截至第十节，1~4 项均已关闭（D3 见八、D4 见七、D8/D6/D9 见十），
> 剩下的是第 5 项，以及第十节 10.5 里那条「缺一次真实模型重跑」。

1. **D3 LLM 双协议适配层 + 目标契约生成**（原第 1 项 D10、第 2 项 D1、D4 已关闭）
2. **D8 脱敏管线**（事件已带 `redaction` 标注，还没真脱敏）
3. **D6 MCP 接线**（SDK 已在，成本低）
4. **D9 档位持久化**（小事，顺手）
5. **D5 VSCodium、D2 node-pty、D7 云端**（各自需要下载体积 / 原生编译 / 云账号）

### 仍需补齐的输入

| 编号 | 内容 | 阻塞项 |
| --- | --- | --- |
| 凭据 | Anthropic API Key；OpenAI 兼容端点 Base URL + Key + 模型名 | D3、契约生成、整个 Agent 主循环 |
| N-03 | 设备规格基线 | 上下文预算与并发上限的实测定标 |
| N-04b | 模型调用超时与重试阈值 | 适配层错误处理 |
| Q4-a | 腾讯云账号（AKSK 或 CAM 角色） | 云端三项 |
| Q4-b | 实例规格 | 云端成本模型与 SB-05 熔断阈值 |
| Q13 | Ardot 画布元素级规格 | 界面像素级还原（当前只做信息架构与视觉基调） |

其中凭据是唯一的硬阻塞：没有它，D3 之后的一切都无法验证「真的能跑」，只能继续写不可执行的骨架。

---

## 六、第二轮：安全底线与持久化（D10 + D1 关闭）

用户裁决「补安全底线 + 持久化」后完成。以下每条都有可复现的实测输出。

### 6.1 D10 裁决权收回 tool-host

**改法**：`ToolExecParams` 里**删掉 `risk` 字段**。调用方不再能说「这条命令是 L0」或「这条已经批准了」。

| 环节 | 首版 | 现在 |
| --- | --- | --- |
| 风险分级 | 调用方填 | tool-host 自己 `classify`（调用方无法指定等级） |
| 裁决 | 调用方填 `outcome` | tool-host 查 `TIER_TRUTH_TABLE`（`decideOutcome`） |
| 档位 | 调用方填 `tier` | 主进程经 `tool.tier.set` 推送的权威值 |
| 工作区根 | 调用方填 | 主进程经 `tool.init` 下发的固定值 |
| 「已批准」 | 调用方一句话 | 主进程签发的 HMAC 票据（`approval-token.ts`） |

**票据四条约束**（缺一条就形同虚设）：

1. 绑定 `runId` —— 防跨任务挪用
2. 绑定命令原文的 SHA-256 —— 防「批了 A 去跑 B」
3. 绑定 `riskClass` —— 防「按低危批、按高危跑」
4. 单次使用 + 5 分钟过期 —— 防重放

**密钥信任边界**：`TICKET_SECRET` 由主进程 `randomBytes(32)` 生成，
**只经 tool-host 专用端口下发**，不经 agent-host 中转。agent-host 无密钥，因此**无法伪造票据** ——
这条在下面的 ③ 里被实证。

### 6.2 实测（`npm run smoke`）

```
[init] tool-host 已就绪（workspaceRoot=…，tier=standard）
[probe] ① 分级 rm -rf → L3（R4-01）
[probe] ② 无票据被拒 → 拒绝执行：风险 L3 需逐次审批（风险 L3，档位 standard，规则 R4-07），未获批准不产生进程
[probe] ③ 伪造票据被拒 → 拒绝执行：签名校验失败（伪造或密钥不符）（风险 L3，档位 standard，规则 R4-07）
[probe] ④ 票据绑定命令生效 → 拒绝执行：票据与命令不匹配（防改口令重放）（风险 L2，档位 standard，规则 R3-01）
[probe] ⑤ 只读自动放行 → exit=0 stdout=v22.22.2（AUTO_APPROVE）
[probe] ⑥ 会话授权 → 首次 SESSION_GRANT，复用免票据（basis=user-grant）
```

③ 是这一轮最关键的断言：探针用**错误密钥**签了一张看起来完全合法的票据，
tool-host 在 `spawn` 之前就拒了 —— 意味着 agent-host（无密钥）即使被模型操纵也造不出有效批准。

### 6.3 D1 事件存储换 SQLite

`better-sqlite3@11.10`，为 Electron 33（ABI 130）单独取预编译二进制。

| 表 | 写入方 | 用途 |
| --- | --- | --- |
| `t_event` | P-03 | 事件流（单一事实来源），`seq` 单调，`UNIQUE(run_id, seq)` |
| `t_checkpoint` | P-03 | 检查点位点（`atSeq`），恢复重放起点 |
| `t_approval_audit` | **P-04** | 审批审计六要素 + `executed` 标记 |
| `t_schema_meta` | 双方 | 表结构版本 |

**审计为什么由 tool-host 写**：审计的意义是「即使 Agent 想赖账，记录也在」。
让 agent-host 写，它可以选择不写 —— 那就成了自我申报。写在执行方，调用方绕不过。
且**先留痕再执行**：被拦下的尝试同样入库（`executed=0`），事后能查「试过但被拒」。

**实测审计库内容**（另一个进程只读打开查询）：

```
schema version: { v: '2' }
audit rows: 6
   DENY  L3 standard PER_CALL_PROMPT risk-table  R4-07 | smoke
   DENY  L3 standard PER_CALL_PROMPT policy-deny  R4-07 | smoke
   DENY  L2 standard PER_CALL_PROMPT policy-deny  R3-01 | smoke
   EXEC  L0 standard AUTO_APPROVE    risk-table  R1-04 | smoke
   EXEC  L1 standard SESSION_GRANT   user-grant  R2-01 | smoke-session
   EXEC  L1 standard SESSION_GRANT   user-grant  R2-01 | smoke-session
```

第三次 `EXEC` 就是会话授权的复用：同一任务同一风险等级，第二次不再需要票据。

**跨进程持久化**：上一轮进程写入的 `01M5FN1X…` 在新进程里依然可查，
`checkpoints: 1（milestone-boundary / run-created, atSeq=1）`。

### 6.4 本轮踩到的坑

| 问题 | 现象 | 处置 |
| --- | --- | --- |
| 表结构漂移 | 旧库已有 `t_approval_audit`，`CREATE TABLE IF NOT EXISTS` 不补列，运行时报 `no column named executed` | 加 `applyMigrations`（`PRAGMA table_info` 判列 → `ALTER TABLE`），`SCHEMA_VERSION=2` |
| 探针选错命令 | 用 `echo SHOULD_NOT_RUN` 测「L3 应被拒」，结果 `echo` 在只读白名单里，自动放行了，看着像闸门失效 | 改用 `chmod 777`（R4-07）与 `cp … ../x`（R3-01）；教训写进探针注释 |
| `git` 写操作未归类 | `git init` / `git add` 落到「无法归类 → 每次审批」，吵且诱发审批疲劳 | 写操作提示扩到 git 的写子命令 |
| 原生模块 ABI | `better-sqlite3` 默认装的是 Node ABI 预编译，Electron 里加载会失败 | `prebuild-install --runtime=electron --target=33.4.11`，并用 `ELECTRON_RUN_AS_NODE` 直接验证加载 |

---

## 七、第三轮：D4 审批卡片与发起方（D4 关闭）

### 7.1 缺的不是界面，是发起方

首版审批通道 `evt:approval` 是通的，但**没有任何东西会发起审批** —— 卡片做出来也永远空着。
所以这轮先补链路，再做界面：

```
渲染层「发起调用」
  → P-01 api:invokeTool
  → P-03 agent.tool.invoke
      → 反向请求 P-01 → P-04 tool.precheck（拿风险等级与是否需要授权）
      → 落 ToolInvoked 事件
      → 需要授权：落 ApprovalRequested + 推 ApprovalRequest 到渲染层
          → 用户点「允许」→ P-01 api:approvalRespond → 签票 → 回 P-03
          → 用户点「拒绝」/ 超时 → 落 ApprovalRejected + 审计 executed=0 → 结束
      → P-03 带票据反向请求 P-01 → P-04 tool.exec
          → tool-host 自行分级、自行裁决、校验票据、落审计、再 spawn
      → 落 CommandOutput 事件，结果回渲染层
```

**P-03 与 P-04 不开直连通道**：P-03 的反向请求经 P-01 转发（`WorkerHost.handle`）。
这样「谁调用了什么」在主进程只有一个入口，便于审计与限流。

**手动入口不是旁路**：控制台的「发起调用」走的就是上面这条链，和将来 LLM 驱动的
工具调用是同一个入口（`agent.tool.invoke`）。区别只在「谁决定要跑这条命令」。

### 7.2 卡片长什么样

- 风险徽标：`L3 · 危险命令`，颜色随等级变，但**文字本身能独立读懂**（不只靠颜色）
- 命令原文：等宽字体、独立底色块，长命令横向滚动不换行、不截断（截断会让人批错东西）
- 元信息：`cwd` · 泳道 ID · 当前档位
- 三个按钮：`拒绝` / `允许一次` / `本任务内允许`（对应 `once` 与 `session` 两种票据范围）
- 顶栏常驻 `待审批 N` 徽标
- **超过 3 条聚合为批量审批卡**：「全部拒绝 / 全部允许一次」，同时保留逐条裁决入口

### 7.3 实测（`npm run smoke -- --smoke-approval`）

探针不是走 API 空转，而是在真实渲染进程里**读界面文字、点界面按钮**：

```
[probe] ⑦ run 就绪 → 01M5FPX54K6JVAQNVVZHFAV5ZB
[probe] ⑧ 审批卡已上屏（含风险徽标与三档按钮）
[probe] ⑨ 允许后执行 → {"ok":true,"risk":{"command":"dd if=/dev/zero …","riskClass":"L3","tier":"standard",…}}
[probe] ⑩ 拒绝后未执行 → {"ok":false,"reason":"用户拒绝执行",…}
[probe] ⑪ 4 条并发 → 聚合为批量审批卡
[probe] ⑫ 批量拒绝后队列清空
```

样本命令用 `dd if=/dev/zero of=/dev/null bs=1 count=0`：命中 R4-02 判 L3，
但执行起来是空操作 —— **验证闸门不该以「真的干了坏事」为代价**。

### 7.4 审计补齐：被拒的裁决也要留痕

第一版跑完发现 `t_approval_audit` 里只有「放行执行」的记录 ——
因为用户拒绝时根本不会调用 `exec`，而审计是 exec 路径里写的。
「用户拒绝过什么」是审计里最该留的一类，漏掉它等于否认发生过。

补 `tool.audit.deny`：只分级 + 落审计（`executed=0`），不执行。
仍由 tool-host 重新分级，调用方送来的等级不作数。

实测审计表（同一批 dd 命令）：

```
DENY PER_CALL_PROMPT user-deny   ×5   （单条拒绝 1 + 批量拒绝 4）
EXEC PER_CALL_PROMPT user-grant  ×1
```

事件流同步一致：`ApprovalRequested 12 / ApprovalDecided 12 / ApprovalRejected 10 / CommandOutput 2`。

### 7.5 尚未做的部分

| 项 | 说明 |
| --- | --- |
| 审批超时的用户提示 | 超时 120s 按拒绝处理（`basis:'timeout'`），界面目前只表现为卡片消失 + 结果行提示，没有倒计时 |
| 并发泳道隔离 | 同一 run 内目前串行等待；`parallelLaneId` 字段已带在请求里，泳道间并发尚未启用 |
| 审批历史的可视回看 | 审计表里有，界面没有查询入口（可随 D9 一起做） |

---

## 八、第四轮：D3 真实模型接入（D3 关闭）

### 8.1 接入方式：双协议各自原生实现，再归一

> 第七轮又加了第三套（`openai-responses`），见第十二节。本节写的原则仍然成立，
> 只是「双」变成了「三」：**判定收成一处，差异各自成文件**。

不做「用一个协议去模拟另一个」。模拟会丢掉三样东西，而这三样恰好都关乎成本与正确性：

| 丢掉的东西 | 后果 |
| --- | --- |
| `prompt caching` | 多轮循环里每轮都要重付全额输入费，长任务成本翻几倍 |
| `thinking block` 的结构语义 | 无法区分「模型想了什么」和「模型说了什么」，排障时看不出它为什么这么干 |
| `tool_result` 的独立消息形状 | OpenAI 要求 `role:'tool'` + `tool_call_id` 独立成条，塞进 user 文本模型会认不出这是工具返回 |

所以 `anthropic.ts` 说 Anthropic 的话，`openai.ts` 说 OpenAI 的话，各自解析完再归一到
`UnifiedMessage` / `ContentPart` / `UnifiedRequest` / `UnifiedResponse` / `TokenUsage`。

### 8.2 实测摸到五条网关脾气

腾讯 tokenhub 网关（`https://tokenhub.tencentmaas.com/v1`）同时提供 `/v1/messages` 与
`/v1/chat/completions`，119 个模型。接的过程中踩到五个坑，都已写进代码注释：

1. **思考块没有 signature**。Anthropic 侧的 thinking 块只有 `{type,thinking}`，服务端也不校验
   —— 但**回传它会打掉前缀缓存**（实测 `cache_read_input_tokens` 从 192 掉到 0）。
   所以 `toAnthropicContent` 主动丢弃 thinking：缓存比「让模型看见自己上次怎么想的」更值钱。
2. **`thinking:{type:'disabled'}` 并非所有模型都认**。`deepseek/deepseek-flash` 认（2.8s / 313 tok），
   `glm-5.3-flash` **直接 400**。代码对「400 且报文含 thinking」做一次去参重试，而不是整个请求判死。
3. **hy4-preview 的思考在 Anthropic 通道关不掉，在 OpenAI 通道能关**。这条最初只验了一半，
   差点写成一个错误结论：`/v1/messages` 上 system 里写「不要输出思考」被无视、`thinking:{type:'disabled'}`
   也不生效，约 28 tok/s，单次契约起草 15 秒以上。后来在 `/v1/chat/completions` 上用同一个参数实测：

   | 通道 | 参数 | 结果 |
   | --- | --- | --- |
   | `/v1/messages` | `thinking:{type:'disabled'}` | 无效，8.8s 级延迟 |
   | `/v1/chat/completions` | `thinking:{type:'disabled'}` | **有效**：同一句「你好」8.81s/231 tok（其中思考 221）→ 1.24s/7 tok（思考 0） |

   教训：**同一个参数在两套协议上的行为可以完全不同**，关思考必须两套各自实现、各自实测。
   （当时只验了 Anthropic 侧就断言「hy4 就是慢」，是拿一个通道的结论去推另一个通道。）
4. **两套协议的 usage 口径是反的**。Anthropic `input_tokens` 不含缓存命中（`cache_read` 另算）；
   OpenAI `prompt_tokens` 含缓存命中（`cached_tokens` 是子集）。不做减法直接映射，
   OpenAI 侧会把缓存命中按全额单价**重复计一次费**。现已统一到 Anthropic 口径。
5. **OpenAI 侧曾经根本没发 `thinking` 参数**。`cfg.disableThinking` 只在 `anthropic.ts` 里被读取，
   `openai.ts` 构造 body 时漏了它 —— 界面上「请求关闭思考」在 OpenAI 通道下**静默失效**，
   不报错、不提示，只是慢。这是本轮补上的（见第十一节）。

### 8.3 Agent 主循环：模型可自动化的是「想」，不是「批准」

`src/agent-host/agent-loop.ts`，每步的动作是：

```
模型输出 tool_use  →  invokeTool（与手动入口同一入口）
                   →  tool-host 自行分级 →（可能需要票据）→ 执行
                   →  tool_result 回灌模型  →  下一步
```

**循环本身没有任何执行能力**。它只会「请求执行」，也从不知道票据长什么样 ——
票据由主进程签发、直达 tool-host，agent-host 只是搬运工。
模型发出的 `run_command` 里既没有 `risk` 也没有 `approval` 字段，给了也没用，tool-host 不读。

三个刻意的约束：

| 约束 | 理由 |
| --- | --- |
| `MAX_STEPS = 12` | 模型会绕圈，绕圈烧的是真钱，必须有硬上限 |
| `finish` 是唯一正常出口 | 「说了几句就停下」判 `failed`，不算完成 —— 否则「跑了几步没报错」会被误当成成功，这正是 Agent 产品最常见的不诚实 |
| `finish` 必须给 `unmet` | 未满足的判据不许塞进 `satisfied` 凑数；完不成如实报比假装完成有用 |

### 8.4 实测（`npm run smoke -- --smoke-agent`）

**Anthropic 协议**（`deepseek/deepseek-flash` + 关思考）：

```
[probe] ⑱ 契约已起草 → 01M5FTH9P0NPA2F13102ABD807
[probe] ⑲ 契约已确认 → state=CONFIRMED，交给模型驱动
[probe] ⑳ 循环收尾 → outcome=delivered steps=3 输入 760 + 缓存命中 3712 / 输出 449
[probe] ㉑ 模型请求的命令确实过了裁决链（事件流 17 条，含 ModelMessage + ToolInvoked + riskClass）
[probe] ㉒ 执行结果卡片已上屏
[probe] ㉓ 收尾后无残留审批卡（僵尸卡片已收回）
```

模型自己决定跑 `node --version` → 再跑 `which -a node` 核对版本来源 → 用 `finish` 逐条给证据收尾。
探针目标刻意选得极小：**要证明的是链路通，不是让模型去改仓库，更不该在自检里烧掉几万 token**。

界面截图由 `--snapshot` 产出（Electron 自带 `capturePage`，不依赖系统截图工具）：

```bash
npm run smoke -- --smoke-agent --snapshot
# → .snapshots/openintj-<ts>.png（1584×935）
```

**OpenAI 协议**（同模型）：

```
[probe] ⑱ 契约已起草 → 01M5FT4MK5WW3QTMYXJ48GAQHG
[probe] ⑲ 契约已确认 → state=CONFIRMED，交给模型驱动
[probe] ⑳ 循环收尾 → outcome=delivered steps=2 输入 796 + 缓存命中 2560 / 输出 634
[probe] ㉑ 模型请求的命令确实过了裁决链（ModelMessage + ToolInvoked + riskClass 均在事件流中）
[probe] ㉒ 执行结果卡片已上屏
```

模型自己决定跑 `node --version` → 再跑 `which node` 核对版本来源 → 用 `finish` 逐条给证据收尾。
探针目标刻意选得极小：**要证明的是链路通，不是让模型去改仓库，更不该在自检里烧掉几万 token**。

### 8.5 本轮修掉的真实缺陷

| 缺陷 | 怎么发现的 | 修法 |
| --- | --- | --- |
| 执行结果只活在 React 局部 state | ㉒ 断言失败：探针直接调 API 绕过按钮，卡片没上屏 | 结果改从 `RunDelivered`/`RunFailed` 事件派生 —— 顺带修掉「界面重开结果就没了」 |
| OpenAI 侧输入 token 虚高 5 倍 | 两次跑同一目标，Anthropic 报 2461、OpenAI 报 12033 | 见 8.2 第 4 条，统一口径后报「796 + 缓存 2560」 |
| 界面空状态文案写死「LLM 凭据尚未配置」 | 模型已配置时还这么显示 | 按 `llm.configured` 分两段文案 |
| 「当前 Run 未启动」旁边写着「17 条事件」 | 截图看出来的：runId 只在按钮路径里设置 | runId 也从事件信封兜底（`e.runId`），从检查点续跑这类入口才不会显示错 |
| 循环结束后还挂着僵尸审批卡 | 截图右上角出现「待审批 1」，但循环早就收尾了 | 新增 `approval.cancel` 通道：run 收尾时收回该 run 的未决请求，卡片跟着消失 |
| **探针自己会假阴性** | `--smoke-run` 报「事件未回流」，但事件明明在屏幕上 | 见下 |

最后一条值得单说，因为它是**验证工具本身不可信**：

`--smoke-run` 的断言原来是「读整页 innerText 的前 800 字符，搜 `GoalContractDrafted`」。
这个 800 是给首屏打印排版用的截断。LLM 没接入时事件流只有两三条、排得很靠前，
断言一直是绿的；接上真实模型后界面多了契约卡片和一堆事件，那条事件被推到 800 字符以外，
断言立刻变红 —— 而功能完全正常。

修法不是「把 800 调大」，是**换断言方式**：

```
整页 innerText 做 substring   →   document.querySelectorAll('.stream-type') 逐条读类型列
整页搜「执行结果」            →   document.querySelector('.delivery') 是否存在
整页搜「成功判据」            →   .contract-section-title 的文本列表里是否真有这个分栏
```

教训：断言要用**结构化的定位**，不要用「页面上某处出现某段字」。
后者的通过与否取决于版面长度、元素顺序、甚至 CSS —— 这类断言绿的时候不能让你放心，
红的时候也说不清是产品坏了还是断言坏了。

### 8.6 仍未做的部分

| 项 | 说明 |
| --- | --- |
| 「设置与凭据中心」页（IC-06） | 凭据目前只走环境变量 —— 这是当前唯一不落明文盘的路径，比写进配置文件安全，但用户得会设环境变量 |
| 上下文压缩 | 长任务到 12 步前可能先撞模型上下文上限；`ContextCompressed` 词表已留，逻辑未写 |
| 检查点续跑 | `DELIVERED` 与 `contract-drafted` 都落了检查点，但没有「从检查点重放」的执行器 |
| 成本预算闸 | 用量已按步累计并进事件流，但没有「超过 N 元就停」的硬闸 |

---

## 九、第五轮：IC-06 设置与凭据中心（IC-06 关闭）

### 9.1 为什么这件事不能凑合

凭据只走环境变量有两个问题：用户得会设环境变量；换个终端就得重设一遍。
但把它写进配置文件更糟 —— **配置文件会落明文盘**。

所以这一轮的核心不是「加个设置页」，而是**找到一条既不麻烦用户、又不摊开密钥的路径**：
用 Electron 的 `safeStorage`（Windows DPAPI / macOS Keychain / Linux libsecret）加密后写盘。
密文可以随便存，没有当前用户的登录态解不开。

三条纪律：

| 纪律 | 落点 |
| --- | --- |
| 明文不落盘 | `safeStorage.encryptString` 后写 `credentials.enc`；冒烟探针**把文件字节读出来搜明文**，搜得到即判失败 |
| 保险箱不可用即拒绝保存 | fail-stop，绝不「降级成明文文件」—— 降级是最常见的安全事故起点：用户以为加密了，其实没有 |
| 明文不回渲染层 | 只给掩码 `sk-…tEbM`。明文只出现在两处：保存那一瞬、下发给 agent-host 的 MessagePort |

### 9.2 密钥怎么进到子进程

agent-host（P-03）要调模型就必须持有 key，这是无法回避的。但**谁来决定用哪一份**可以收拢：

```
agent-host 每次用之前  →  callMain('credential.get')  →  P-01
                                                        ├─ 保险箱优先
                                                        └─ 环境变量兜底
```

两个设计选择：

- **按需拉，不启动时推**。IPC 往返不到 1ms，换来的是「设置改完立刻生效」，
  而且不需要任何缓存失效逻辑 —— 缓存失效是这类代码最容易长 bug 的地方。
- **优先级只在 P-01 判断一次**。子进程不重复实现「保险箱还是环境变量」——
  两处实现迟早会不一致，而不一致的凭据行为极难排障。

### 9.3 实测（`npm run smoke -- --smoke-cred`）

```
[probe] ㉔ 已保存进保险箱 → sk-…tEbM（来源 vault）
[probe] ㉕ 回传只有掩码（sk-…tEbM），明文未出主进程
[probe] ㉖ 落盘为密文（247 字节，未包含明文密钥）
[probe] ㉗ 生效来源已切到 vault（保险箱优先于环境变量）
[probe] ㉘ 测试连接通过 → 1047ms · deepseek/deepseek-flash · 回复「ok」
[probe] ㉚ 设置与凭据中心已上屏（协议 / Base URL / 模型 / API Key / 思考开关）
[probe] ㉛ 清除后回落为 env 来源（环境变量仍然可用）
[probe]    已还原保险箱到测试前状态
```

㉖ 是这一轮最该盯的一条：它不是「代码里写了 safeStorage 所以应该加密了」，
而是把 247 字节的文件读出来，去里面找那把 40 位 key。找不到才算过。

探针结束按**原字节还原**保险箱，不在用户机器上留下测试写进去的东西。

### 9.4 一个必须分开的字段

`maskedKey` 有值 ≠ 保险箱里有凭据 —— 环境变量来源时 `maskedKey` 也有值（那是 env 里那把 key 的掩码）。
第一版界面用 `maskedKey` 判断「已保存」，结果是：

- 明明用的是环境变量，输入框却写「已保存 sk-…tEbM，留空则不修改」
- 「清除凭据」按钮可点，但点它清的是保险箱，清不掉环境变量

所以 `CredentialStatus` 里把 `vaultHasCredential` 单独列出来，与 `source` 一起判断。
**一个字段同时表达两件事，界面就一定会说错话。**

### 9.5 顺带修掉的另一条僵尸卡片路径

跑探针时 ㉓ 忽然红了：循环收尾后仍挂着 1 条审批卡。

查下来不是「卡没收」的逻辑问题，是**卡片消失的触发条件错了**：
原来只在 `respond()`（用户点按钮）里乐观移除卡片。
而审批也可能被别处应答 —— 另一个窗口、自动化测试替身、程序自己超时收卡 ——
这些路径都不经过按钮的 `onClick`，卡片就永远贴在屏幕上。

修法是把触发条件从「谁点的」改成「**这条审批已经有结论**」：

```
收到 ApprovalDecided 事件 → 按 requestId 移除对应卡片
```

这与本项目一贯的口径一致：**界面状态从事件流派生，不从某条调用路径的副作用派生**。
（上一轮的执行结果卡片也是同一个毛病，同样改法。）

### 9.6 仍未做的部分

| 项 | 说明 |
| --- | --- |
| 多套凭据档案 | 现在只有一份「当前生效」。多项目切换要来回覆盖，后续可做 profile 列表 |
| 密钥轮转提醒 | 没有「这把 key 存了 90 天了」之类提示 |
| 凭据使用审计 | 审计表记的是命令，不记「这次调用用的是哪份凭据」 |
| Linux 无密钥环的环境 | 目前直接 fail-stop，只提示改用环境变量；未做「本次会话内存凭据」的降级路径 |

---

## 十、第五轮：脱敏、MCP、档位持久化（D8 / D6 / D9 关闭）

这一轮做三件「不做就一直漏」的事：敏感内容怎么落盘、外部工具怎么进来、用户的设置怎么留住。
三件都不改变架构，但都会长期影响安全与信任。

### 10.1 D8 脱敏管线：底线不能由调用方决定

`shared/redaction.ts` 九类规则（`sk-*` / `gh*` / `AKIA*` / `xox*` / JWT / 私钥块 / URL 内嵌凭据 /
`Bearer` / `key=value` 形式），挂载点只有一个：`agent-host` 的 `emitEvent()`。

**为什么必须只有这一个口子**：如果让每个 `emitEvent` 的调用点自己脱敏，漏掉的那一处不会报错，
只会静静地漏。现在只有一条路径能写事件流，也就只有一处需要审。

三条刻意的设计：

1. **`redaction: 'none'` 也照样遮密钥**。底线不能依赖调用方标对级别 —— 标错是常态。
   命中后级别自动升为 `secret`，落库的是实际级别，不是请求的级别。
2. **刻意不遮「32 位以上 hex」**。git commit hash 是 40 位 hex，遮掉它排障就废了。
   安全措施如果让日常排障变不可能，它就会被关掉 —— 那是更差的结果。
3. **路径脱敏保留结构**：`~/projects/app/src/index.ts`。审计关心「动了哪个文件」，
   不关心用户名。

实测（`--smoke-redact`，㉜~㊲ 六条全绿）：

```
[probe] ㉜ 声明 none 仍遮住密钥（级别自动升为 secret）→ curl -H "Authorization: Bearer sk-***tEbM" https://example.c…
[probe] ㉝ 私钥 / JWT / 连接串密码 / kv 形式凭据均已遮蔽（命中 private-key,jwt,url-credential,kv-secret）
[probe] ㉞ commit hash 未被误伤（40 位 hex 原样保留）
[probe] ㉟ 路径脱敏为 ~/projects/app/src/index.ts（结构保留，排障仍可读）
[probe] ㊱ 嵌套结构与数字/普通文本保持原样，只有密钥值被替换
[probe] ㊲ ANSI 控制序列已清理
```

审计表里的 `command` 也过一遍 `redactText(..., 'secret')`：
`curl -H "Authorization: Bearer sk-..."` 这种命令太常见，原样入库是纯风险。

### 10.2 D6 MCP：外部工具不是旁路，是更长的一条同款路

**MCP 客户端放在 tool-host，不放在 agent-host。** 理由是 D10 的延续：
MCP 调用同样是「执行」，必须走同一条裁决与审计链。放到 agent-host 就等于给模型开了一条侧门 ——
那里没有裁决权，也没有审计库。

四个不那么显然的决定：

| 决定 | 原因 |
| --- | --- |
| 工具名 `mcp__<server>__<tool>`（双下划线） | OpenAI 的 function name 只允许 `[a-zA-Z0-9_-]`，冒号会被网关拒掉；工具名要在两套协议里通用 |
| 解析时从**左侧第一个** `__` 切 | 工具名本身可以含下划线；从右侧切会拆错 |
| 分级只看 server 的 `trust`，不复用 shell 分类器 | `classifier.ts` 是给命令字符串用的（正则匹配 `rm -rf`），而 MCP 参数是结构化 JSON —— 「它会不会写盘」静态判断不了。能力由 server 决定，不由参数决定 |
| 票据绑定 `mcp:server:tool {参数}` 的哈希 | MCP 没有命令行。不绑参数的话，「批准一次」就能被换成任意参数重放 —— 那等于免审批通道 |

还有一个「分开写」的细节：MCP 被拒走 `tool.mcp.deny`，而不是复用 shell 的 `recordDenial`。
后者会用 shell 分类器去判 `mcp:...` 这串字符串，算出来的等级是错的。
**审计里写一个错误的风险等级，比不写更糟** —— 它会让人做出错误的判断。

界面侧：设置页新增 MCP 面板，显示配置路径、每个 server 的连接状态与工具数、
以及**失败原因**。只显示「工具少了一个」会让用户以为产品坏了。

实测（`--smoke-mcp`，⓪ + ㊳~㊾ 十三条全绿）：

```
[probe] ⓪ init 已下发 MCP 配置路径 → C:/Users/Administrator/AppData/Roaming/openintj/mcp.json（当前 0 个 server）
[probe] ㊳ 两个 MCP server 已连接（工具数 probeRo:2, probeConfirm:2）
[probe] ㊴ 工具发现 4 个，命名与解析往返一致（mcp__probeRo__echo）
[probe] ㊵ 只读自动放行 → "pong:hello"（L0/AUTO_APPROVE，1ms）
[probe] ㊶ confirm 级无票据被拒 → 拒绝执行：风险 L2 需逐次审批（风险 L2，档位 standard，规则 R6-mcp-call）
[probe] ㊷ 被拒调用已入库 → executed=false L2/R6-mcp-call basis=user-deny
[probe] ㊸ 持票据执行成功 → "pong:with-ticket"（PER_CALL_PROMPT/user-grant）
[probe] ㊹ 票据绑定参数生效（换参数即失效）→ 拒绝执行：票据与命令不匹配（防改口令重放）
[probe] ㊺ isError 如实透传 → "boom: 这是预期内的失败结果"
[probe] ㊻ 设置页已列出 2 个 MCP server → ["probeRo","probeConfirm"]
[probe] ㊼ 工具目录 6 项（内置 run_command/finish + MCP 4）
[probe] ㊽ 坏配置如实上报 → mcp.json 读取失败：Expected property name or '}' ...
[probe] ㊾ MCP 命名规范校验通过（前缀识别 / 空名拒绝）
```

探针不 mock：它自带一个 stdio MCP server（`scripts/fixtures/mcp-probe-server.mjs`，行分隔 JSON-RPC，
只实现握手 / 列工具 / 调工具），用 `ELECTRON_RUN_AS_NODE=1` 拉起 —— 探针环境里没有独立 node。
跑完按**原字节还原**用户自己的 `mcp.json`，并重载一次让运行时状态与磁盘一致。

㊼ 是这一轮最该盯的一条：它验的不是「MCP 连上了」，而是
**「模型真的会看到这些工具」**。`runLoop` 与 `toolCatalog` 调同一个 `resolveToolSpecs()`，
所以「界面说有、模型看不到」这种偏差不存在。

### 10.3 D9 档位持久化：三处一起改，顺序不能反

落 `t_app_setting`（`SCHEMA_VERSION` 2→3）。新表不需要写迁移 —— 两个 store 都是
「先 `exec(SCHEMA_SQL)` 再 `applyMigrations`」，而 DDL 全是 `IF NOT EXISTS`，老库下次打开就自动有了。
只有「给已存在的表补列」才必须显式迁移。

两个容易写错的地方：

1. **恢复必须三处一起改**：主进程 `currentTier`、tool-host（真正的裁决方）、渲染层。
   只恢复一处的结果是「界面显示严格档，实际按标准档裁决」——
   那比不持久化更危险，因为用户以为自己已经收紧了。
2. **顺序是「先改到位，再落库」**。反过来的话，落库成功而推送失败，
   用户下次开机会看到严格档、实际按旧档跑。宁可「提示设置失败」，也不要「以为设了」。

渲染层新增 `onTierChanged` 通道：恢复发生在 `boot()` 里（`initToolHost` + `selfCheck` 之后），
可能比首屏慢半步。不推这条消息，用户会看到默认档，然后重新点一遍 —— 而它其实已经存住了。

实测（`--smoke-tier`，51~54 全绿）：

```
[probe] 51 切换即落库 → 库里 strict，裁决方 tier=strict
[probe] 52 「重启」后回到默认档（界面 standard，裁决方 standard）
[main]  档位已从库里恢复：standard → strict
[probe] 53 三处一起恢复 → 主进程 strict，裁决方 strict，界面 strict
[probe] 54 非法档位被忽略 → 仍为 standard（不做静默猜测）
```

「重启」是真的重启 —— 52 用的是 `webContents.reload()`，不是改 React state。
这样界面那一处也回到了默认档，53 才能证明推送通道真的生效。

### 10.4 本轮修掉的两个真实缺陷

| 问题 | 现象 | 处置 |
| --- | --- | --- |
| 探针自己假阴性 | ㊷ 第一次通过、第二次红：审计库跨次运行是累积的，`rows.length === 1` 的断言在第二轮必然失败 | 每次用一个全新的 `runId`。**断言依赖「库里只有这一条」，就把探针写成了只能用一次** |
| MCP 面板在截图里只剩半个卡片 | 面板在设置页底部，`capturePage` 不滚屏 | 截图前先 `scrollIntoView`。存证图拍不全等于没拍 |

另外把 `init` 里加载 MCP 改成**不 await**：一个 `npx` 现装包的 server 要 20s 才起来，
把 init 卡住就等于把「票据密钥下发」卡住，而那个失败的后果是全链路 fail-closed。
少几个工具是降级，跑不动是事故 —— 降级优先。

### 10.5 本轮未验证的部分（诚实清单）

| 项 | 原因 | 影响 |
| --- | --- | --- |
| D3 双协议 / Agent 主循环的**真实模型**重跑 | 本轮环境没有凭据（保险箱为空、`OPENINTJ_LLM_*` 未设），`--smoke-llm` / `--smoke-agent` 按设计「跳过而不伪造」 | 改动集中在 `agent-loop` 的 deps（新增 `tools` / `requestMcpTool`），已过双端 typecheck；但「模型真的去调 `mcp__` 工具」这一步只有代码与 ㊼ 的静态证据，缺一次真跑 |
| 用户真实 `mcp.json` 的兼容性 | 只验了自带探针 server 与坏配置两条路径 | 未试过 `npx @modelcontextprotocol/server-*` 这类真包（要联网装包） |
| HTTP / SSE 传输 | 只实现 stdio | 远端 MCP server 暂不可接，已列入「尚未接入」 |

---

## 十一、第六轮：OpenAI 通道的关思考是静默失效（补第十节 10.5 的缺口）

第十节留了一条没验完的事：「模型真的去调 `mcp__` 工具」缺一次真实模型跑。这一轮补验证时，
顺手发现了一个**只在 OpenAI 通道存在、不报错、只是慢**的缺陷。

### 11.1 缺陷：`disableThinking` 只实现了一半

`LlmConfig.disableThinking` 是一个跨协议的配置项，界面上是「请求关闭思考 / 保留思考」两个胶囊。
但代码里只有 `anthropic.ts` 读过它：

```
anthropic.ts  →  if (withThinkingOff) body['thinking'] = { type: 'disabled' }   ✅
openai.ts     →  （没有任何地方读 cfg.disableThinking）                          ❌
```

后果不是报错，是**慢**：用户在设置页选了「请求关闭思考」，走 OpenAI 协议时这个开关被完全忽略，
模型照常思考。而「慢」不会出现在任何日志里 —— 它只会表现为「这个产品怎么这么卡」。

这类缺陷比崩溃更难发现：崩溃会留下堆栈，静默失效只会留下用户的耐心。

### 11.2 修法：两套各自实现，各自降级

```ts
const buildBody = (withThinkingOff: boolean): Record<string, unknown> => {
  ...
  if (withThinkingOff) body['thinking'] = { type: 'disabled' }
  ...
}

let out = await post(buildBody(cfg.disableThinking))
if (out.status === 400 && cfg.disableThinking && /thinking/i.test(out.text)) {
  console.warn('[llm] 该模型/网关不接受 thinking 参数，去掉后重试一次')
  out = await post(buildBody(false))
}
```

与 Anthropic 侧同构，理由也同构：`thinking` 不是 OpenAI 官方规范，兼容网关认不认是运气。
认了省钱省时间，不认就 400 —— 401/403/429 不能重试，但「参数不认识」重试一次是合理的。

### 11.3 实测：同一参数，两个通道两个结果

| 通道 | 请求 | 耗时 | 输出 token（其中思考） |
| --- | --- | --- | --- |
| `/v1/chat/completions` | 原样（不带 thinking） | 8.81s | 231（思考 221） |
| `/v1/chat/completions` | `thinking:{type:'disabled'}` | **1.24s** | **7（思考 0）** |

同一句「你好」，**快 7 倍，token 少 33 倍**。

这直接推翻了第八节 8.2 的第 3 条结论。当时只验了 `/v1/messages` 就写下「hy4-preview 的思考关不掉」，
然后据此把 `deepseek-flash` 定为唯一推荐 —— 那是拿一个通道的实测去推另一个通道的行为。
现在两个通道、两个模型都能用，选型不再被这个误判绑住。

### 11.4 端到端：OpenAI 协议 + hy4-preview + 关思考

```
[probe] ⑮ 模型接入 → {"configured":true,"protocol":"openai","model":"hy4-preview","baseUrlHost":"tokenhub.tencentmaas.com","missing":[]}
[probe] ⑯ 契约已由真实模型起草 → 01M5GGC8HQ1230QMKAYWGV9K1F / state=CONTRACT_PENDING
[probe] ⑰ 契约卡片已上屏（5 个分栏：成功判据 / 明确不做 / 依赖假设 / 已知风险 / 待用户澄清）
[probe] ⑱ 契约已起草 → 01M5GGCHKZ4T70KFMN8Z9R7ER7
[probe] ⑲ 契约已确认 → state=CONFIRMED，交给模型驱动
[probe] ⑳ 循环收尾 → outcome=delivered steps=2 输入 1607 + 缓存命中 1472 / 输出 246
[probe] ㉑ 模型请求的命令确实过了裁决链（事件流 16 条，含 ModelMessage + ToolInvoked + riskClass）
[probe] ㉒ 执行结果卡片已上屏
[probe] ㉓ 收尾后无残留审批卡（僵尸卡片已收回）
```

⑳ 是这一轮最该看的数字：**2 步、输出 246 token、缓存命中 1472**。
如果关思考没生效，光第一步的思考就会吃掉 200+ token，2 步内交付几乎不可能。

⑮~㉓ 九条全绿，说明「OpenAI 协议 → 契约起草 → 人工确认 → 模型多步执行 → 事件回流界面」
整条链在 `hy4-preview` 上成立。第十节 10.5 的第一条缺口就此关闭。

### 11.5 仍未做

| 项 | 说明 |
| --- | --- |
| `reasoning_effort` 等原生参数 | 只实现了 `thinking:{type:'disabled'}` 这一种写法。官方 OpenAI 端点用 `reasoning_effort`，将来要接真 OpenAI 时需要按 `baseUrl` 分流或加配置项 |
| 关思考失败的可观测性 | 网关不认参数时会 400 并重试，但「重试后其实还在思考」无法从用量上告警 —— 目前只能靠 `reasoningTokens` 事后观察 |

### 11.6 附：Responses API 能力探测（`/v1/responses`，当时的探测记录）

同一网关还提供 `/v1/responses`（OpenAI 的下一代接口）。**写这一节时产品只支持 `anthropic` 与 `openai`
（Chat Completions）两套，所以它是能力探测记录；用户在下一轮明确选择「接成第三协议」，
落地过程与验证见第十二节。** 探测结论保留原样，因为它是第十二节所有实现的依据。

实测（`hy4-preview`，2026-10-22）：

| 能力 | 结论 |
| --- | --- |
| 基本调用 | 通。`{instructions, input:"你好"}` → 4.98s / 输出 131 tok（思考 117） |
| 关思考 | **`reasoning:{effort:"none"}` 生效** → 1.30s / 输出 7 tok（思考 0） |
| 工具声明 | `tools[]` 是**扁平**的：`{type:'function', name, description, parameters}`，没有 `function:{}` 包裹 |
| 工具调用返回 | `output[]` 里的 `{type:'function_call', call_id, name, arguments}`（`arguments` 同样是 JSON 字符串） |
| 多轮回传 | 通。input 数组追加 `{type:'function_call_output', call_id, output}` → 1.62s 正常续答 |
| usage 口径 | `input_tokens` 含缓存命中，`input_tokens_details.cached_tokens` 是子集 —— **与 Chat Completions 一致**，不是 Anthropic 口径 |

三个与现有实现的结构性差异：

1. **没有 `choices[]`，也没有 `finish_reason`**。返回是 `output[]` 数组（可能含 `reasoning` / `message` /
   `function_call` 三类 item），顶层只有 `status: 'completed'`。判断「是否要执行工具」得扫 output 里的
   `function_call` item，而不是看 `finish_reason === 'tool_calls'`。
2. **思考内容被降级成 summary**。`reasoning` item 里是 `summary[].text`（提炼过的摘要），
   不是 Chat Completions 的 `reasoning_content` 原文。对我们「思考不回放」的策略是加分项
   （更省 token），但也意味着拿不到完整推理过程。
3. **关思考的参数是标准的**。`reasoning:{effort:'none'}` 是 OpenAI 官方规范，
   而 Chat Completions 侧用的 `thinking:{type:'disabled'}` 是非标扩展。若将来要接真 OpenAI 官方端点，
   走 responses 比走 chat/completions 更不依赖运气。

接入需要改动：`LlmProtocol` 加成员、新建 `agent-host/llm/responses.ts`、归一化层适配
（`output[]` → `ContentPart[]`、`call_id` → `toolUseId`）、设置页协议胶囊加一项、探针与文档。
**当时未接入的原因**：现有两套协议已覆盖当前需求，第三套要常年维护三份。这个判断在下一轮被用户推翻
（需求是「网关支持哪套就用哪套」，而不是「够用就行」），落地见第十二节。

---

## 十二、第七轮：Responses 接成第三协议

用户裁决：**「现在接成第三协议」**。这一轮把它从「探测记录」变成「可用的通道」。

### 12.1 为什么不复用 `openai.ts`

做之前先比了一遍线格式，结论是复用会让两套都变成低质量的：

| | Chat Completions | Responses |
| --- | --- | --- |
| 系统提示 | `messages[0].role='system'` | 顶层 `instructions` |
| 输出结构 | `choices[0].message` | `output[]`（多个 item 并列） |
| 工具调用 | `message.tool_calls[]`（消息的一部分） | `output[]` 里独立的 `function_call` item |
| 工具结果 | `role:'tool'` + `tool_call_id` | `{type:'function_call_output', call_id, output}` |
| 工具定义 | `tools[].function{name,description,parameters}` | `tools[]` **扁平** |
| 结束原因 | `finish_reason` | **没有**，得从 output 反推 |
| 关思考 | `thinking:{type:'disabled'}`（非标） | `reasoning:{effort:'none'}`（官方规范） |

七行里六行不同，且「结束原因」这一行是**语义级别**的差异，不是字段名映射。
所以新建 `src/agent-host/llm/responses.ts`，`index.ts` 的 `complete()` 只加一个分派臂。

### 12.2 最容易写错的一处：`StopReason` 得自己推

Responses 没有 `finish_reason`。若照抄 Chat Completions 的 `mapFinishReason(choice.finish_reason)`，
拿到的永远是 `undefined → 'unknown'`，而 `agent-loop` 只在 `stopReason === 'tool_use'` 时才继续跑工具。

```ts
function deriveStopReason(data, hasFunctionCall): StopReason {
  if (hasFunctionCall) return 'tool_use'          // 关键：漏了这行，模型一调工具就被当成「说完了」
  if (data.status === 'incomplete') { ... }       // 截断
  return 'end_turn'
}
```

症状会是「Agent 只会聊天、从不干活」，而且**不报错**。这跟第十一节那个 `disableThinking` 静默失效
是同一类缺陷：不抛异常，只是行为不对。

### 12.3 顺手收掉一个隐患：协议判定原来有两份

加第三个成员时才发现，协议名在**两个地方**各解析了一遍，都是「不是 openai 就当 anthropic」：

```
src/shared/llm.ts:135        readLlmConfig：env → 协议
src/main/credential-store.ts:110  normalizeCredential：保险箱凭据 → 协议
```

只改一处的话，症状是「设置页能选 Responses、探针也说 openai-responses，但保险箱里存的还是 anthropic」——
换个来源就变行为，极难查。所以这轮把它收成唯一入口：

```ts
// shared/llm.ts —— 全工程唯一的判定点
export function parseProtocol(raw: string | undefined): LlmProtocol | null
```

三个调用方（环境变量 / 保险箱 / 设置页）全走它。同时**认不出的名字不猜**：
返回 `configured:false` 并把原因写进 `missing`，而不是默认 anthropic。

### 12.4 探针：先把「协议真的生效了」变成一条断言

加了一条 ⑮-b，比对「环境变量声明的协议」与「运行时实际生效的协议」：

```
[probe] ⑮ 模型接入 → {"configured":true,"protocol":"openai-responses","model":"hy4-preview",...}
[probe] ⑮-b 协议一致 → openai-responses
```

为什么要专门加：协议静默降级时，**原有九条探针会全绿**（它们只关心「模型能起草契约、能给命令」，
不关心走的是哪条通道）。绿得毫无意义比红更难发现。

负向验证（把名字写错一个字母，`openai-response`）：

```
[probe] ⑮ 模型接入 → {"configured":false,"protocol":null,...,"missing":
        ["OPENINTJ_LLM_PROTOCOL=openai-response（未知协议，可选 anthropic / openai / openai-responses）"]}
[probe] ⑮ 模型未配置，跳过契约生成（不伪造输出）
```

不降级、不猜、把可选项直接写在错误里 —— 这条负向验证比正向那条更有价值。

### 12.5 端到端：Responses 协议 + hy4-preview + 关思考

```
[probe] ⑮ 模型接入 → {"configured":true,"protocol":"openai-responses","model":"hy4-preview",...}
[probe] ⑮-b 协议一致 → openai-responses
[probe] ⑯ 契约已由真实模型起草 → 01M5GGWTB68XJ87JSZW44837KC / state=CONTRACT_PENDING
[probe] ⑰ 契约卡片已上屏（5 个分栏）
[probe] ⑱ 契约已起草 → 01M5GGX3PBCK3VZ0A3WRHJPWAR
[probe] ⑲ 契约已确认 → state=CONFIRMED，交给模型驱动
[probe] ⑳ 循环收尾 → outcome=delivered steps=2 输入 1556 + 缓存命中 1344 / 输出 281
[probe] ㉑ 模型请求的命令确实过了裁决链（事件流 16 条，含 ModelMessage + ToolInvoked + riskClass）
[probe] ㉒ 执行结果卡片已上屏
[probe] ㉓ 收尾后无残留审批卡（僵尸卡片已收回）
```

⑯~㉓ 与第十一节 OpenAI Chat 那次的结果同构，说明整条链在第三套协议上也成立。
其中 ⑳ 的 `steps=2` 与 ㉑ 的 `ToolInvoked` 是最关键的两条：

- `steps=2` 证明 `function_call` 被正确识别成 `tool_use`（12.2 那个坑没踩到）；
- ㉑ 证明 `call_id` → `toolUseId` → `function_call_output` 的回传是通的 ——
  否则第二步会因为「工具结果对不上号」而失败。

缓存命中 1344 也印证了 11.6 的探测结论：Responses 的 `input_tokens` 含缓存命中，
归一时已减去（`inputTokens = 1556` 是未命中部分），不会重复计费。

### 12.6 全量回归：54 条断言全绿（含一条探针自身的问题）

```
npm run smoke -- --smoke-run --smoke-approval --smoke-llm --smoke-agent \
                 --smoke-cred --smoke-redact --smoke-mcp --smoke-tier --snapshot
（OPENINTJ_LLM_PROTOCOL=openai-responses · hy4-preview）
```

第一次跑出**一条红**：㊻「设置页只列出 0 个 MCP server」。查下去发现不是产品缺陷，
而是**探针之间的状态耦合**：

- MCP 面板只在设置页**挂载时**拉一次状态（之后靠用户点「刷新」）；
- `--smoke-cred` 的 ㉚ 结束时把页面留在设置页；
- `--smoke-mcp` 的 ㊻ 再点「设置」是一次空操作（React 状态没变 → 不重渲染），
  面板仍是 `mcp.json` 为空时的旧数据 → 断言误红。

修法是让探针**先切回控制台再进设置页**，强制重新挂载。这也解释了它为什么单独跑是绿的、连着跑就红 ——
**「只有组合运行才红」的探针，先怀疑隔离，再怀疑产品**。

修完后全量通过：⓪ / ①~⑭ / ⑮ / ⑮-b / ⑯~㊾ / 51~54，共 54 条。

其中 ㉘「测试连接」`1008ms · hy4-preview · 回复「ok」` 值得单说：它走的是新协议，
而**设置页那个按钮一行代码都没改**。原因很简单 —— 它调的是 `complete()`，
协议分派在统一入口里做完了。这才是"归一化"该有的收益：新增一套协议，上层不用知道。

### 12.7 三套协议的分工（写进 README）

| 协议 | 端点 | 什么时候用 |
| --- | --- | --- |
| `anthropic` / `claude` / `messages` | `POST {baseUrl}/messages` | Claude 原生；要 prompt caching 与完整 thinking block |
| `openai` / `chat` | `POST {baseUrl}/chat/completions` | 兼容面最广 |
| `openai-responses` / `responses` | `POST {baseUrl}/responses` | 网关支持时优先：工具定义更简洁、关思考用官方规范 |

### 12.8 维护成本：这是真实的负债

三套协议意味着**任何 LLM 层的改动都要改三处**：新增能力（如 stream、图片输入、prompt caching 控制）、
用量口径调整、错误归一。加第三套的收益是「网关支持哪套用哪套」，代价是这个。
所以 README 的硬约束表里专门写了一条「协议判定只有一个入口」——
唯一能压低这个负债的，是把「判定」与「差异」分离：判定收成一处（`parseProtocol`），
差异各自成文件（`anthropic.ts` / `openai.ts` / `responses.ts`）。

---

## 十三、第八轮：内嵌终端（D2 / US-15 关闭）

用户说「继续」。剩下的偏离里，D5（VSCodium）与 D7（云端）都卡在外部资源，
只有 D2 能在这台机器上端到端做完，所以做它。

### 13.1 先解决一个前提：这台机器没有 C++ 工具链

`node-pty` 是原生模块，设计文档也写了「需按 Electron ABI 重建」。
但这台机器**没有装 Visual Studio / Build Tools**（两个 VS 目录都是空的），
`node-gyp rebuild` 必失败。所以先验证能不能绕开编译：

```
node-pty@1.1.0 的 install 脚本 = node scripts/prebuild.js || node-gyp rebuild
装上之后的产物在 node-pty/prebuilds/：
  win32-x64/{pty,conpty,conpty_console_list}.node
  darwin-{arm64,x64}/pty.node
```

它自带 **N-API 预编译**。N-API 的 ABI 稳定，不随 Node / Electron 版本变，
所以 `npm i node-pty` 一步到位、不需要 `electron-rebuild`。实测：

```
require('node-pty').spawn('cmd.exe', ['/c','echo pty-ok'])
→ exit=0，输出 "\x1b[2J\x1b[m\x1b[Hpty-ok\r\n\x1b]0;C:\Windows\SYSTEM32\cmd.exe\x07\x1b[?25h"
```

输出里带清屏序列和「设置窗口标题」序列 —— 这恰好证明它是**真 PTY**：
这些只有 ConPTY 会给，`child_process` 拿不到。

### 13.2 形态决策：为什么不做「裸 PTY 键盘直通」

US-15 的 AC-2 是硬约束：「终端不是绕过策略层的通道」。这条要求把形态选择逼到只剩一个答案：

| 形态 | 能不能执行前拦截 | 结论 |
| --- | --- | --- |
| 键盘直通（xterm 收键盘 → 原样写 pty） | **不能**。回车直接进 pty，shell 立刻执行，此时再分级只是事后追认 | ✗ |
| 按键全直通、只拦回车 | 不能。拦了回车命令就没提交，可用户已经在 vim 里按了回车 | ✗ |
| **整行提交**（输入行 → 分级 → 放行才写 pty） | **能**。命令边界掌握在自己手里 | ✓ |

所以首版终端是：**持久 PTY 会话 + 只读输出区（xterm.js）+ 底部命令输入行**。

不是「阉割版终端」，而是唯一能同时满足「真 PTY 语义」与「执行前裁决」的形态：

- **真 PTY 该有的一样不少**：`cd` / `export` 在会话内保留、颜色与进度条（TTY 检测）、
  按窗口宽度折行（resize 到 pty）；
- 失去的是交互式全屏程序（vim / top / ssh）—— 它们的「命令边界」不是换行。

这个代价必须写清楚，且**不是待修 bug**：想要 vim，就得放弃「每条命令先裁决」，
二者在同一个会话里不可兼得。

### 13.3 命令流：三层各管一件事

```
渲染层 TerminalPanel      提交「整行命令」，只负责显示
      ↓ api:terminalSubmit
主进程                    问 tool-host「要不要票」→ 要票就弹卡 → 签票 → 再提交
      ↓ tool.term.input
tool-host terminal.ts     分级 → 裁决 → 先留痕 → 放行才写 pty
```

三个约束：

1. **分级在 tool-host**（裁决权不外包）。主进程只负责「问人」与「签票」——
   它自己判断风险就等于把 D10 白做了。
2. **票据绑定命令原文**。终端没有命令行参数，票据绑的就是这一整行；换一行即失效（探针 61）。
3. **审批卡要出现在用户正在看的那个面板里**。终端页打开时控制台是 CSS 隐藏的，
   卡片弹在看不见的地方，用户只会觉得「什么也没发生」，然后重敲一遍。
   所以终端面板自己订阅一次审批事件、渲染一条内联审批条。

### 13.4 两个真 bug（都不是逻辑写错，是环境不按你想的来）

**① 面板提示写进 pty 区域，被 PowerShell 擦掉。**

最早把「✗ 已拒绝」「⏳ 等待裁决」直接 `term.write()` 进 xterm。现象：
代码明明执行了（现场记录里 `kind:'denied'`、`ref:true`、`rows:34`），终端里一个字都没有。

原因：那一刻 shell 正停在提示符上，**xterm 的光标就在提示符那一行的末尾**，
我们的字落在同一行；随后 PowerShell 重绘提示符（带「清行尾」序列）把它整段擦掉。
真终端里 prompt 那一行归 shell 所有，任何外部文字都会被下一次重绘清掉。

修法：**面板要说的话走自己的渲染通道**（一条 React 提示条），pty 区域只放 pty 输出。
判断依据很简单：谁是这块区域的所有者，谁才有权往里写字。

**② 探针按 `.tab[1]` 取设置页，加了「终端」标签后指错了地方。**

㉚ 的断言是「设置页字段都在」，红在下一行，看起来像页面渲染坏了。
实际是 `.tab[1]` 从「设置与凭据」变成了「终端」。修法：按文字找标签页。
**位置选择器在 UI 会长大的产品里不能用** —— 它不会报错，只会静默指错。

（还有一条小的：`auditRecent` 返回的 `executed` 是布尔，我按 `=== 0` 比较，断言永远匹配不到。
断言写错和功能坏掉的表现一模一样，都是红。）

### 13.5 实测（`--smoke-term`，55~64 全绿）

```
[probe] 55 终端会话已建立 → term-…-1 · powershell.exe · <工作区根> · 221x34
[probe] 56 只读命令自动放行 → 输出里出现 v22.22.2
[probe] 57 严格档下终端命令被拦在审批条前（终端不是旁路）
[probe] 58 批准前未执行（新增输出 0 字符，无命令结果泄漏）
[probe] 59 拒绝后明示原因并留痕（executed=0 / L0 / user-deny）
[probe] 60 允许后真的执行 → 终端里累计出现 2 次版本号
[probe] 61 票据绑定命令生效（换命令即失效）
[probe] 62 resize 生效 → 100x40
[probe] 63 中断已下发（Ctrl+C 不经裁决）
[probe] 64 关闭后会话已回收（再提交返回 no-session，列表为空）
```

其中两条是这轮的核心：

- **57 + 58**：严格档下同一条只读命令立刻要审批，而且**在批准之前命令没有执行**。
  只验「有没有弹卡」是不够的 —— 弹卡与执行可以同时发生；
- **61**：票据绑的是命令原文，拿「批准了 `node -v`」的票去提交 `whoami` 被拒。

全量回归 64 条断言（⓪ / ①~㉔ / ㉕~㉘ / ㉚~㊿ / 51~64）全绿，`npm run typecheck` 双端通过。

### 13.6 仍未做的部分（诚实清单）

| 项 | 说明 |
| --- | --- |
| 打包时的 asar unpack | 开发与冒烟直接读 `node_modules`，没问题；真打包必须把原生 `.node` 与 `node-pty/build/Release/**` unpack，否则在 asar 里加载不到。设计文档「部署」节已列，代码里还没落到 electron-builder 配置 |
| 终端输出进事件流 | US-15 的 AC-1 写了「输出同步出现在实时执行流」。终端会话目前不属于任何 run，写事件流需要「终端会话 = run」的建模；首版没做（也顺带避免了把终端流灌进事件表） |
| 多会话 / 分屏 | 后端已支持 3 个会话（`MAX_SESSIONS`），界面只有一个面板 |
| 全屏交互程序 | 见 13.2，与 AC-2 冲突，不做 |

---

## 十四、第九轮：内嵌 IDE（D5 / F27 关闭）

### 14.1 这轮最重要的一个决定：不走 `serve-web`，直启 reh-web

架构文档写的是「内嵌 VSCodium + Open VSX」，实现路径原本打算用官方的
`codium-tunnel serve-web`。实测把它否掉了：

```
$ ./bin/codium-tunnel.exe serve-web --host 127.0.0.1 --port 0 --connection-token X ...
Web UI available at http://127.0.0.1:55342?tkn=X

$ curl http://127.0.0.1:55342/?tkn=X
HTTP 202
The latest version of the VSCodium Server is downloading, please wait a moment...
```

**它在首次 HTTP 请求时联网下载服务端本体**：实测 94,038,270 字节、本机耗时 99 秒，
期间所有请求都返回 202 + 一页「正在下载」。三个后果：

- 首次打开是 100 秒白屏，不是 3 秒；
- 断网 / 内网环境直接不可用；
- 版本号由远端决定，同一份代码两次跑可能拿到两个服务端 —— 不可复现。

而它下载的那一坨（`vscodium-reh-web-*` 包）**本身就是一个可独立运行的 HTTP 服务端**：

```
vendor/vscodium-server/
├── node.exe              93.6 MB（自带运行时，不依赖系统 node 版本）
├── out/server-main.js    入口
├── product.json          version=1.140.06880, commit=807ba2d6…
├── bin/codium-server.cmd 只是 node.exe out/server-main.js 的一层包装
├── extensions/ resources/ node_modules/
```

于是改成：**自己把 reh-web 拉到 vendor/，运行时直启 `node.exe out/server-main.js`**。
参数与 `serve-web` 几乎一致（`--host` / `--port 0` / `--connection-token` /
`--server-data-dir` / `--accept-server-license-terms` / `--telemetry-level`）。

代价是版本要自己管；收益是**启动 291~346ms**（不是 99 秒）、离线可用、版本可复现，
体积也从「桌面版 229MB + 服务端 94MB」降到只要 94MB。

### 14.2 HTTP 握手（实测，别凭印象改）

| 请求 | 结果 |
| --- | --- |
| 无令牌 | **403 Forbidden**（纯文本 `Forbidden.`） |
| 错令牌 | **403 Forbidden** |
| `?tkn=<正确>` | **302** → `Location: /`，下发 cookie `vscode-tkn` |
| 仅带该 cookie 再取 `/` | **200** + workbench HTML |

令牌校验是真的生效的，不是摆设 —— 这正好是「外部产品边界」最该被验的一条。

### 14.3 边界怎么收紧的（四条）

| # | 措施 | 落点 |
| --- | --- | --- |
| 1 | 只绑 `127.0.0.1`，端口传 `0` 交给系统分配（不撞本机已有服务、不用默认 8000） | `src/main/vscodium.ts` 的 spawn 参数 |
| 2 | 每次启动现生成 `randomBytes(24)` 令牌，**只留在内存**，不过渲染层（`IdeStatus.url` 只带 origin） | `VscodiumHost.token` |
| 3 | 停止时 `taskkill /pid <pid> /T /F` 杀整棵树（它会派生扩展宿主，只杀父进程会留下占着端口的孤儿） | `killTree()` |
| 4 | 工作台用独立 session 分区（`persist:ide`）的 `WebContentsView` 承载，与主界面 cookie / storage 隔离；`before-quit` 里 `dispose()` | `ensureIdeView()` |

### 14.4 为什么不用 iframe

渲染层自己 iframe 一个第三方工作台要放松 CSP，而 CSP 是**整个渲染进程**一份的 ——
为一个面板把主界面的安全基线降掉不划算。改用主进程的 `WebContentsView` 盖在占位区上：

```
渲染层 IdePanel  ──ideLayout(rect)──▶  主进程 applyIdeLayout()
      │                                        │
      └─ 只留一块占位区（透明的洞）      WebContentsView.setBounds(rect)
                                        setVisible(active)
```

代价写清楚：工作台区域对渲染层是「一块透明的洞」，我们无法在上面叠自己的 DOM，
也无法截获它的键盘事件。所以面板状态条放在占位区**上方**而不是盖在工作台上。

### 14.5 探针结果（65~74，全绿）

```
[probe] 65 编辑器标签页可用 → 面板已挂载
[probe] 66 服务端就位 → 1.140.06880（phase=idle）
[probe] 67 IDE 已就绪 → http://127.0.0.1:61676/ · 启动耗时 291ms
[probe] 68 错令牌被拒 → HTTP 403（连接令牌真的在校验）
[probe] 69 正确令牌放行 → HTTP 302
[probe] 70 只绑回环 → 1 个非回环地址均不可达；端口 61676（非默认 8000）
[probe] 71 工作台已渲染 → http://127.0.0.1:61676/?folder=C%3A%5CUs… · .monaco-workbench 就位
[probe] 72 已停止 → phase=stopped
[probe] 73 端口已释放 → :61676 不再应答（进程树杀干净了）
[probe] 74 界面文案无禁用商标 → 仅出现 VSCodium
```

三条是这轮的核心：

- **68**：错令牌 403。只验「页面打开了」不算数 —— 令牌若是摆设，这个外部边界等于没设；
- **70**：真的去连了一次非回环地址（本机 1 个 LAN IP），连不上才说明没绑 `0.0.0.0`。
  只断言「URL 里写着 127.0.0.1」是循环论证；
- **73**：停止后端口不再应答。抓的是「只杀父进程、留下占端口的孤儿」这类最难归因的泄漏。

全量回归 **74 条断言**全绿（㉔ 因环境无 `OPENINTJ_LLM_*` 跳过，属预期），
`npm run typecheck` 双端通过。

### 14.6 这轮踩的坑

| 现象 | 根因 | 修法 |
| --- | --- | --- |
| 65 假红：`点击=no-tab 面板=yes` | 探针写成 `?.click() ?? 'no-tab'` —— **`click()` 返回 `undefined`，`??` 的兜底永远命中**，于是无论点没点到都显示 `no-tab` | 改成「先找 → 找不到早退 → 点完返回确定值」 |
| 起服务后满屏 `Unable to resolve nonexistent file .../extensions` | `--server-data-dir` 下没有 `extensions/` 与 `data/User/`，服务端启动时扫不到就报（非致命，但会淹没真正的失败原因） | 启动前 `mkdirSync` 预建 |
| 首次 `curl` 拿到 502 / 空响应 | 本机有 `HTTP_PROXY`，curl 走了代理；另外后台进程随 Bash 调用结束被回收，端口已死 | curl 加 `--noproxy '*'`；把「启动 → 探测 → 杀掉」放进同一次调用 |

### 14.7 仍未做的部分（诚实清单）

| 项 | 说明 |
| --- | --- |
| 打包分发 | `vendor/` 不入库。真打包要把 `vendor/vscodium-server` 作为 `extraResources` 打进 `resources/`，`layoutCandidates()` 里已经写了这条路，但 electron-builder 配置还没落 |
| 扩展预装 | Open VSX 是 VSCodium 的默认市场（`product.json` 里 `extensionsGallery.serviceUrl` 已指向 `open-vsx.org`），但首版没预装任何扩展，也没有做「推荐扩展」列表 |
| IDE 与工作区边界 | 目前用 `?folder=<工作区根>` 打开。服务端本身不限定它能读哪些路径 —— 它没有我们 tool-host 那套裁决链，这是「外部产品」这个选择的固有代价 |
| 多窗口 / 分屏 | 只有一个工作台实例，切走标签页只是隐藏（`setVisible(false)`），不会销毁 |
| 版本升级 | 固定版本（写在 `vendor/vscodium-server/SOURCE.json`），升级要靠重跑 `npm run vscodium:fetch` |

### 14.8 一条待办

`vendor/vscodium/`（桌面版，229MB）与 `.cache/VSCodium-win32-x64-1.140.06880.zip`
是早期按「桌面版 + serve-web」路径拉下来的，**现在都不再需要**，可以删掉省约 458MB。
重跑 `npm run vscodium:fetch` 只会拉 94MB 的 reh-web。
