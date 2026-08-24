# OpenINTJ Changelog

本文件追踪 OpenINTJ 的对外可观察变更。
版本号沿用 [SemVer](https://semver.org/lang/zh-CN/) 与
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格。

## [0.3.1] — 2026-08-24

应用图标改为字面产品名 openINTJ（去掉擅自生成的几何图形）。安装包版本 `0.3.1`。

### Changed

- `ts/apps/desktop/resources/icon.png` 由 `scripts/generate-icon.ps1` 绘制产品名；窗口标题栏显示 `v0.3.1`。

## [0.3.0] — 2026-08-24

桌面端首个正式安装包版本（tag `v0.3.0`，相对预发布 `v0.3.0-alpha.0`）。
含 RFC-005/006/007/008、桌面工作台，以及本轮运行时收口。
GitHub 尚未配置代码签名 secrets 时，CI 产出未签名 Win/mac/Linux 包，不再因空 `CSC_LINK` 失败。

### Added

- **技能工具硬隔离**：命中技能声明的 `tools` 后，ToolHub 按并集收窄 `list()` / `call()`；
  camelCase 别名归一到 `read_file` 等注册名。空声明不限制。并发 `tao.run` 用 ALS 隔离。
- **Desktop 配置热重装**：设置页保存 workspace / 模型 / 功能开关后原地重装 agent，无需退出进程。
  会话级字段不触发重装；Ollama URL 在失焦时写入。
- **真实搜索 gated harness**：`RUN_SEARCH_LIVE=1` 跑 Tavily/Brave 真请求（不进 normal CI）。
- **Linux CI 打包**：`release.yml` 矩阵增加 ubuntu AppImage。
- **应用图标**：`ts/apps/desktop/resources/icon.png`（1024×1024 像素；该版曾带擅自生成的几何图形，已在 0.3.1 换成产品名）。
- **未签名可发布的 Release CI**：缺 `WIN_CSC_*` / `MAC_CSC_*` / Apple 公证 secrets 时跳过签名与公证。

### Changed

- 技能 prompt 从「建议优先使用工具」改为「本轮仅可使用工具」，与硬隔离一致。
- Desktop `package.json` 版本 `0.3.0-alpha.0` → `0.3.0`；窗口标题栏显示 `v0.3.0`。

## [0.3.0] RFC-005/006/007/008 首期（2026-07-14，随本版一并发布）

> RFC-005 ModelRuntime + ADR-002、RFC-006 Product Behavior、RFC-007 TaskPool MVD。

### Added

- **RFC-008 输入结构化与消歧**：deterministic preflight 后、classifier 前增加自适应
  `structureUserInput`；保留 `originalInput`/`executionInput` 双语义；复杂或高歧义输入
  至多一次有界优化，关键缺失约束暂停并追问 1–3 问；Desktop 持久化并渲染「任务理解卡」。
  配置 `OPENINTJ_INPUT_STRUCTURING=off|adaptive|always`；OTel
  `openintj.input.structure`；Product Behavior control 组强制关闭以保证 A/B 可比。
- **Desktop 多模型任务工作台**：新增 Kimi K3、MiniMax M3、GLM 5.2
  OpenAI-compatible provider；模型 Profile/连接测试、`safeStorage` 加密凭据、对话级即时
  切换与有界历史；新增工作区 → 任务 → 对话 → 消息 SQLite 持久化、Inbox 首启迁移、
  任务树/恢复/归档，以及可关联 TaskPool run 的字段。
- **Desktop 受控重启**：标题栏和设置页提供重启入口；重启前停止接收新 chat、abort
  在途 LLM 请求、注销 IPC/监听并关闭 Agent 与工作台数据库。
- **ModelRuntime lifecycle / observability**：新增限频 single-flight `refreshHealth()`、
  `ModelRuntimeError`、structured provider attempts/lastError；typed
  `model.provider.*` / `model.embedding.fingerprint.*` hooks 对应 OTel spans/counters。
- **RFC-007 应用级重启恢复**：TaskGraph 快照新增原始 `goalInput`；server/desktop 在
  TaskPool + real data dir 下启动扫描 incomplete runs。默认把遗留运行安全标记 cancelled，
  只有 `OPENINTJ_TASK_POOL_RECOVERY=resume` / `taskPoolRecoveryPolicy: "resume"` 才续跑，
  防止崩溃窗口内重复外部副作用；旧版缺输入快照拒绝不可靠 resume。新增 TaskPool 单测、
  SQLite 集成测试及 server/desktop 应用级 E2E。
- **`@openintj/model-runtime`**：统一 `resolveLlmClient` / `resolveEmbedder`；`auto` 本地优先（Ollama → Hunyuan → 可见 mock）；显式 `ollama`/`hunyuan` strict fail-closed；embedding 指纹校验。
- **RFC-006**：`product-behavior.ts`、`trait-scenarios.ts`、`trait-eval.ts`；planning/clarification skills；`planning`/`analysis` 路由护栏；OTel `openintj.product.behavior.injected`。
- **RFC-006 配置/状态 parity**：CLI `--product-behavior treatment|control`，server 启动配置沿用
  `OPENINTJ_PRODUCT_BEHAVIOR`；CLI/server/desktop status 暴露行为版本与 treatment/control cohort。
- **RFC-006 trait 可观测**：确定性 `event.PRODUCT_TRAIT_SIGNAL` 与
  `openintj.product.trait.signal`，覆盖 planner 多步骤、clarification skill 命中、成功 search-before-answer；
  指标只代表 lifecycle/tool 事实，不推断模型意图。
- **RFC-006 确定性基线**：normal CI scripted runner 跑 8 traits / 9 cases 全通过；报告见
  `docs/architecture/rfc-006-deterministic-baseline.json`，明确不是 live-model score。
- **RFC-007 TaskPool 完整阶段**：`TaskRun` handle、节点状态机、稳定拓扑/合成、
  `SharedContext` partial results、DAG 验证、有界并发与失败/取消级联。
- **RFC-007 reliability**：`AbortSignal` 贯通 Tao/ReAct/tool 边界，timeout/cancel 分离、
  per-task watchdog、有界指数 backoff retry。
- **RFC-007 persistence / multi-agent**：`TaskStore` + `SqliteTaskStore` restart recovery；
  role-based `AgentInstancePool` 与 Zod `Channel` reducer（opt-in，不进入默认路径）。
- **TaskPool 验收 soak**：新增 gated `taskpool-soak.harness.spec.ts`；真 Ollama 在途 HTTP
  取消连续 3 轮、restart-shaped recovery 连续 25 轮通过。
- **RFC-007 三端与观测 parity**：CLI `--task-pool`、server status、desktop Settings toggle；
  TaskPool 对符合条件的复杂任务优先于 self-consistency；完整 lifecycle hooks 与 OTel
  run/task spans/counters/runId correlation。
- **桌面 Settings**：LLM `auto`、独立 embed provider、Ollama URL/model 配置项。
- **文档**：`RFC-006-product-behavior-contract.md`、`RFC-007-task-orchestration.md`（RFC-005 + ADR-002 已有）。

### Changed

- Hunyuan 默认模型从已退役的 `hy3-preview` / `hunyuan-turbos-latest` 迁移到正式
  `hy3`，默认端点改为 TokenHub；只映射已知旧默认值，自定义端点和模型保持不变。
- **Product Behavior v1.2**：版本号升至 `1.2.0`；T5 拆为不过度澄清与关键歧义必须澄清；
  与 RFC-008 输入结构化对齐，control 组关闭结构化。

- **Product Behavior v1.1 可执行契约**：确定性排序/转换/算术约束/关键澄清/越权破坏请求
  在 LLM 与工具前本地处理；结构化对比和分阶段计划支持一次有界答案修订；单句要求确定性收口。
  ReAct parser 兼容大小写协议标记、拒绝 FINAL 内部标记泄漏，并在 max-iteration 保留最佳 thought。
  T3 进一步区分带 URL 的真实搜索来源与 mock/失败/空结果，后者对时效性事实 fail-closed 为
  “无法可靠确认”，不再把一次 mock 工具调用当作事实依据。直接回忆题只从 `user_input`
  做 grounded preflight，避免错误 assistant 输出污染；中英文 tokenizer 修复无空格中文的
  keyword/BM25 召回。`qwen2.5:0.5b` longrun 两类场景连续两轮 recall/pass 均为 100%。
  三端行为一致。
- **TaskPool 激活与三端 parity**：TaskPool opt-in 自动启用必需的 classifier；CLI/server/
  desktop 状态统一返回 activation reason、prerequisite、persistence/recovery capability。
  server/desktop 另返回 recovery summary，Desktop StatusBar 展示激活原因。CLI embedding
  provider 参数补齐 `simple`/`xenova`，三端顶层 embed/classifier 投影一致。
- **LLM 在途取消传播**：`ChatOptions` 新增 `AbortSignal`，TaskPool worker cancellation 经
  Tao/ReAct（含 `runSingle`）传入 Ollama/Hunyuan fetch。调用方取消立即终止 HTTP 请求并
  保留原始 reason，不再等待 provider timeout；provider 自身超时仍报告 `TIMEOUT`。
- **真实 provider fail-closed**：从 Ollama/Hunyuan adapter 删除内部 mock 生成器与
  `isMockMode` 旁路。Ollama 网络/HTTP/非法响应、Hunyuan 缺 key/鉴权/HTTP/非法响应
  现在统一显式失败并更新 degraded/unauthorized 状态；显式 mock 仅由 ModelRuntime
  `MockLlmClient` 提供。真实 Ollama runtime E2E 4/4 与 Desktop mock smoke 5/5 通过。
- **Desktop workspace 打包边界**：electron-vite 将 `@openintj/*` 工作区包打进主进程 bundle，
  desktop 将这些包归为构建期依赖，避免 electron-builder 沿 pnpm symlink 走出 appDir；
  Windows 未签名 NSIS 已本机成功产出。release workflow 显式接入 Win/macOS 签名 secrets，
  macOS 开启 notarization。
- 三端 agent 改用 `@openintj/model-runtime`（移除重复 `pickLlm`/`buildLlm` 与 Ollama→Hunyuan mock 耦合）。
- 默认 `LLM_PROVIDER=auto`（server/desktop/cli）；`.env.example` 补充 `EMBED_PROVIDER` 与 Ollama 变量。
- trait evaluation runner 可选返回 `toolsUsed`/`trajectory` 结构化证据；T3 优先验证真实 search 工具，
  并强化 T4 单句简洁与 T7 约束确认 judge，同时兼容 finalAnswer-only runner。

### Benchmarks

- **Ollama embedding**：`nomic-embed-text` 在固定语料的产品路径与纯 cosine 路径均为
  **1.000 nDCG@4**（simple：0.773 / 0.396；xenova：1.000 / 0.944）。
- **RFC-006 live-model**：隔离 case memory、串行 cohort 后，`qwen2.5:0.5b` treatment
  连续两次 **9/9**，control **5/9** / **4/9**，baseline 关闭；longrun recall 仍为
  33.3%–50%。T3 当前使用 mock search，仅证明工具顺序，不代表事实答案正确。

## [Unreleased] —— #3 嵌入基准 xenova 真实数字回填 (2026-07-08)

> 收尾项：本机实跑 `RUN_EMBED_COMPARE=1`，把 `xenova`（`Xenova/all-MiniLM-L6-v2`，384 维）真实检索质量回填基准表。

### Benchmarks

- **xenova 实测**：纯 cosine（隔离语义）**0.944 nDCG@4**（simple 仅 0.396）、产品路径 **1.000**（simple 0.773）——
  神经句向量把隔离语义召回翻倍多，兑现了此前「预期收益空间」。见 `docs/architecture/retrieval-benchmark.md`。
- **默认选型结论**：CI/无依赖环境保持 `simple`（零依赖、可回归守护）；质量敏感且可接受本地模型的部署默认切 `xenova`
  （首跑下载 ~80MB 权重，之后离线可用，无需外部服务）。ollama 待本机 `ollama serve` 后回填（本次 `fetch failed` 跳过）。

### Chore

- `@openintj/plane-memory` devDeps 加 `@openintj/embed-xenova` / `@openintj/embed-ollama`（workspace，零外部重量），
  修复三方对比 harness 的可选 `import()` 在 vitest 下无法解析问题。`@xenova/transformers` 仍为 embed-xenova 的**可选 peer dep**
  （不进 root 硬依赖），按需 `pnpm -w add @xenova/transformers`。

## [Unreleased] —— #6 打包发布核实 + 发布运行手册 (2026-07-08)

> 核实 #6 已实现就绪（electron-builder + electron-updater + CI release + UpdateBanner，全链路接线且有测试），
> 补发布运行手册并纠正过时盘点（路线表 #6「待办」→ 实现就绪）。本次纯文档，无代码改动。

### Docs

- 新增 `docs/architecture/release-packaging.md`：本地打包命令、CI 切 release 流程（tag `v*` → `--publish always`）、
  updater 验证、以及**已知手动缺口**（品牌图标、代码签名、Linux CI、首个正式 release 未切）。
- 纠正 `next-session.md` 路线表 #6/#12 与「下一站」清单：#6/#10/#12/#11 均已完成，实质剩项仅 #3 的 xenova/ollama 真实数字回填。

## [Unreleased] —— #12 parity 扩展（ContextEngine/Hooks）+ 技能系统后续（权重衰减/工具绑定/蒸馏质量） (2026-07-08)

> parity 网从 5 slice 扩到 7（core 新增 context + taxonomy）；技能自学习补齐权重衰减、工具子集绑定、蒸馏校验。
> 详见 `docs/architecture/next-session.md` §12.8、`docs/architecture/phase3-6-parity-tests.md`。

### Added

- **parity: context slice**（`@openintj/core`）：锁定 ContextEngine 确定性内核 —— `ContextBudget` 算术
  （`availableTokens`/`usageRatio`/`memoryBudget`/`needsCompaction`）与 `ShaderConfig.get_shader_for_task`
  在 Python v2 ↔ TS `ContextBudgetTracker`/`getShaderForTask` 逐值等价（+12 测试）。
- **parity: taxonomy slice**（`@openintj/core`）：`EventType`/`CommandType` 与 Python 逐条相等；`ErrorCode`
  Python ⊆ TS（TS 多出 hook/react 专用码）。这是 HookBus 事件最接近的跨实现对齐面（+14 测试）。
- **技能权重衰减**：`SkillLearningRuntimeOpts.weightHalfLifeSec` + env `OPENINTJ_SKILL_WEIGHT_HALFLIFE_SEC`
  + `resolveSkillWeightHalfLifeSec`（三端接线）。读时指数半衰期：`weightFor` 随冷却回落，`reinforce` 累加前先衰减旧值。
- **技能工具子集绑定**：`Skill.tools`（frontmatter/sqlite/db 全链路）。文本协议下软绑定（`renderSkillPrompt` 追加
  「建议优先使用工具」行），`skillToolAllowlist` + `assembleSkillContext.onSelected` 暴露命中并集供装配方收窄。
  内建 seed 技能已声明 tools 示例。

### Changed

- **`createLlmSkillDistiller` 校验强化**：name/body 必填 + body 最小长度 + 字段截断 + triggers/tools 归一去重 +
  taskTypes 校验到合法 `TaskType`（过滤幻觉）+ 批内按 id/name 去重；prompt schema 增加 tools/taskTypes 约束。
- **parity 生成器**：`scripts/python-parity/generate_fixtures.py` 新增 `gen_context` / `gen_taxonomy`（只读 Python v2 冻结实现）。

### Tests

- core parity 55→81（+context 12 +taxonomy 14；含既有 governance 9）；skills 38→58；storage-sqlite skills +1
  （tools JSON 往返 + 旧行默认 []）。

## [Unreleased] —— 小缺口收口：enableReact:false 退化分支 + function-calling 决策文档化 (2026-07-08)

> 两件"小缺口"其实在更早会话已落地/已文档化（`runSingle` 退化路径 + ADR-001），盘点未同步。本次核实、
> 补测、纠盘点。详见 `docs/architecture/next-session.md` §12.7。

### Tests

- `@openintj/classifier` 新增 `routing.spec`：`decideRoute`（高置信简单类 → `single=true`/小 topK；fallback /
  低置信 / 非简单类 → `single=false`/默认 topK；policy 覆盖）6 例 + `outcomeSignal` 3 例——锁定「分类 →
  `enableReact:false` 退化路由」此前无专测的链路（退化分支本身早由 core `tao.spec` 守护）。

### Docs

- 纠正盘点 `next-session.md` §8.1（RFC-001 行）/§8.2 #5：`enableReact:false` 退化分支已实现（`runSingle`）且
  经 `decideRoute` 可达；function-calling vs 文本协议决策由 **ADR-001** 记录、RFC-001 §11 Q1 已关闭。

## [Unreleased] —— 方向一并发原语接真实 agent：self-consistency 并发上限 (2026-07-08)

> §9.3 已把 `forkJoin` 接进三端自一致性主路径，但仍是无界全并发。本次把 `@openintj/concurrency` 的
> `Semaphore` 经 `forkJoin.concurrency` 真正用在产品路径——给多采样设并发上限。详见
> `docs/architecture/next-session.md` §12.6。

### Added

- **`forkJoin` 并发上限**（`@openintj/concurrency`）：新增 `ForkJoinOpts.concurrency`——`1..<总数` 时用内部
  `Semaphore` 限流（拿到 permit 才执行，`finally` 释放）；不设 / `<=0` / `>=总数` → 全并发（行为不变，零开销）。
- **自一致性并发上限**：`SelfConsistencyConfig` 新增 `maxConcurrency`；`resolveSelfConsistency` 从
  `opts.maxConcurrency` > env `OPENINTJ_SELF_CONSISTENCY_CONCURRENCY` 解析。cli/server/desktop 三端
  `selfConsistency` 选项新增 `maxConcurrency`，`run()` 透传给 `forkJoin({ concurrency })`——避免多采样一次性打满
  LLM 配额。

### Changed

- `@openintj/concurrency` 集成状态注释更新：`forkJoin` + `Semaphore`（经 `forkJoin.concurrency`）标为**已接入
  产品路径**（自一致性）；Mutex/Channel/CV/Pool/Backpressure 仍标实验性。

### Tests

- concurrency +2（同时在跑子任务 peak ≤ 上限；上限 ≥ 总数则全并发 peak 达总数）；
  shared self-consistency +4（默认无 `maxConcurrency` / opts 透传 / env 读取 / opts 优先 env）。

## [Unreleased] —— RFC-004 workspace 能力面收官（设置 UI + config + utility worker） (2026-07-08)

> workspace 读写/watch + config 服务后端此前已在；本次补 renderer 消费面、config 字段完整性与启动接线、
> 以及把 CPU 密集的钝化挖掘下放 worker 线程。详见 `docs/architecture/next-session.md` §12.5。

### Added

- **桌面「设置」面板**（`SettingsPanel`）：消费既有 workspace/config IPC——显示沙箱信息、选择工作区目录、
  编辑 `AppConfig`（provider/检索模式 + 7 个开关）、实时展示 `fs.watch` 工作区变更（`onWorkspaceEvent`）。
- **`AppConfig` 字段补全**：新增 `enablePersona / enableSkills / enableSkillLearning / enableClassifier`，
  启动时透传给 `assembleDesktopAgent`（改动需重启生效，面板已标注）。
- **utility 挖掘 worker**（`@openintj/dormant`）：`runMineInWorker`（`worker_threads`）+
  `mineWithWorkerFallback`（失败回退内联）；`DormantRuntime` 新增 `mineRunner` 选项 + `lastMineUsedWorker`
  标记；desktop `dormantMineWorker` / env `OPENINTJ_DORMANT_WORKER=1` 启用（仅无 `llmExtract` 时下放）。

### Tests

- dormant +6（worker 透传 / 抛错回退与 `PatternMiner` 等价 / 空事件；`mineRunner` 走 worker / `llmExtract`
  不走 / 未配不走）。真实 worker 线程手动 e2e 通过。

## [Unreleased] —— 钝化记忆 persona 注入闭环（RFC-003 §3.6 收官） (2026-07-08)

> server/desktop 早已注入 persona，本次补齐其余四件让 §3.6 四条验收全绿：`getPersona()` 出口、
> CLI 注入 parity、A/B 杠杆、revoke 的 server/UI 收尾（脱敏此前已默认生效）。详见
> `docs/architecture/next-session.md` §12.4。

### Added

- **`DormantRuntime.getPersona()`**：读「已生效 PersonaConfig」的规范出口（语义等同 `snapshot()`）；
  desktop `DORMANT_PERSONA` IPC 与 server `GET /api/dormant/persona` 均切到它。
- **CLI 钝化记忆**（`@openintj/cli`）：新增 `enableDormant`/`dormantOpts`/`enablePersona`（env
  `OPENINTJ_DORMANT=1`）。内存态运行：每轮 `record` 用户输入，`contextProvider` 注入 `[用户画像]`
  （顺序统一为 persona → skills → `[记忆参考]`）。
- **persona 注入 A/B 杠杆**：`@openintj/shared` 新增 `resolvePersonaInjection`；三端新增
  `enablePersona` 选项 + env `OPENINTJ_PERSONA`（`0`/`false` 关，默认开）——关闭即得无 persona 基线组
  （§3.6 #3）。
- **server persona 撤销路由**：`POST /api/dormant/proposals/:id/revoke`（仅 `applied` 可撤，否则 404
  `not_found_or_not_applied`）；`GET /api/dormant/proposals` 接受 `status=revoked`。
- **DormantPanel 撤销 UI**：新增「已撤销」tab 与 applied 卡片上的「撤销」按钮（抄 `SkillPanel` 形态）。

### Tests

- shared +4（`resolvePersonaInjection` 优先级）、cli +4（注入命中 / A/B 不注入 / 全链路 / 未启用不注入，
  经 `react.beforeThought` 断言最终 system prompt）、server +2（revoke 删字段 + version++ + list；非 applied 404）。

## [Unreleased] —— 治理接进工具执行（RFC-004 §8：策略/配额门禁） (2026-07-08)

> `GovernancePlane` 三端早已构造却从不被调用——工具执行链路无策略/配额门禁。本次把治理接进
> `ToolHub.call` 与桌面 workspace IPC，补齐 RFC-004 §8「Governance → fs」边界。同时纠正过时盘点：
> fs/命令工具其实早是真实沙箱（`createWorkspaceTools`），`[mock]` 只剩 search 兜底。详见
> `docs/architecture/next-session.md` §12.3。

### Added

- **`ToolHub` 治理闸门**：`ToolHubOpts.gate?: ToolGate`（新导出类型），在 `tool.beforeCall` 之后、
  handler 之前执行；抛错 → `ToolCallResult.success=false`，**不触发熔断**（治理拒绝≠工具故障）、
  不发 `tool.onError`（避免被当可重试）。execution 不反向依赖 governance。
- **`GovernancePlane.checkToolCall(command)`**：镜像 `checkAndRecord`，走每分钟**工具配额**
  （`checkToolQuota`/`recordToolCall`）+ 策略黑名单 + 审计（不消耗 API 配额）。
- **`createToolCallGate(governance)`**：把 plane 包成 `TOOL_CALL` 命令的 gate；cli/server/desktop 三端
  `new ToolHub({ hooks, gate: createToolCallGate(governance) })`。
- **桌面 workspace IPC 门禁**：`WORKSPACE_READ/WRITE` 直连沙箱前也过同一 gate（RFC-004 §8）。

### Changed

- 默认策略不回归：白名单含 `read_file/search`；`write_file/execute_command` 非黑非白 → 放行；仅黑名单
  目标或超配额（默认 20 次/分钟）被拦。运行时 `policyEngine.block(target)` 可动态收紧。

### Tests

- execution +4（gate 拒绝不触发熔断 / 放行 / afterCall 可观测但不 onError）；governance +6
  （checkToolCall 放行·黑名单·动态 block·配额 + createToolCallGate 放行·拦截）；cli +1（端到端拉黑拦截）。

## [Unreleased] —— 检索性能/规模 B：LanceDB 原生 FTS (#10) + 嵌入基准 (#3) (2026-07-08)

> 大规模 fragment（N>10k）时把词法检索从「内存 BM25 全表扫描」下推到 LanceDB 原生 FTS；并给
> 嵌入器默认选型补上双路径量化基准。详见 `docs/architecture/retrieval-benchmark.md`
> 与 `docs/architecture/next-session.md` §十二。

### Added

- **存储层原生 FTS（#10）**：`VectorStore` 新增可选 `supportsFts` / `ensureFtsIndex()` /
  `searchText(query, opts)`。`LanceDBVectorStore` 用 `table.createIndex("content", {config: Index.fts()})`
  + `table.search(query, "fts")` 走 BM25 原生索引，旧版 / 不支持时 `supportsFts=false` 静默降级为纯向量；
  `InMemoryVectorStore` 实现 BM25-lite `searchText`（使融合逻辑在不装 LanceDB 时可单测）。
- **混合检索融合**：`fusion.ts` 新增 `rrfFuse`（RRF，只依赖名次、适配 cosine/BM25 异构分数）+
  `hybridVectorSearch(store, {query, queryEmbedding, topK, ...})`（向量榜 + FTS 榜 RRF 融合，
  `searchText` 缺失/空自动降级为纯向量）。
- **server 混合检索 opt-in**：`retrieveHybrid` 新增 `useLanceFts`（默认读 env `OPENINTJ_LANCE_FTS=1`）——
  开启后走 `hybridVectorSearch(persistentStore.vectorStore, …)`，结果映射回 `MemoryHybridHit`
  （RRF 分记入 `components.rrf`）；默认仍走内存 `MemoryHybridIndex`，行为零变化。
- **嵌入基准双路径（#3）**：`benchmarkEmbedderCosine`（纯 cosine，隔离 embedder 语义能力）与既有
  `benchmarkRetrieval`（产品路径）并列。simple 实测归档：产品路径 nDCG **0.773** / 纯 cosine **0.396**
  （维度无关）；新文档 `docs/architecture/retrieval-benchmark.md` 记方法/数字/复现命令。
  xenova/ollama 需装 `@xenova/transformers` / 起 ollama 后 `RUN_EMBED_COMPARE=1` 回填。
- **测试**：storage-lance +13（`fusion.spec.ts` 9 + in-memory FTS 4）；server hybrid-retrieve +3（FTS 路径）；
  plane-memory benchmark spec 加纯 cosine 维度不敏感断言。

## [Unreleased] —— 桌面「技能审批」UI 面板 (2026-07-08)

> 把技能系统 Phase 2 的自学习闭环接到桌面端用户手上——此前只有 HTTP/IPC 后端。抄 `DormantPanel`
> 落地一个「技能」tab：蒸馏 → 审批候选提案 → 查看生效技能与权重。详见
> `docs/architecture/next-session.md` §11.5。

### Added

- **`SkillPanel.tsx`**（desktop renderer）：右侧栏第 4 个 tab「技能」。顶部「蒸馏」按钮触发轨迹蒸馏；
  status filter（pending / approved / rejected / revoked / all）；pending 提案可✓批准/✗拒绝，
  approved 可撤销；底部「生效技能」折叠区显示学习技能 + 权重。未启用（`OPENINTJ_SKILLS_LEARN` 未开）
  时显示启用提示。`App.tsx` 用 `status.skills.pendingProposals` 给 tab 加待审批角标。
- **IPC 协议收窄**：`ipc-protocol` 新增技能响应 DTO（`SkillProposalDto`/`SkillListResponse`/
  `SkillDistillResponse`/`SkillDecisionResponse`/`SkillActiveDto`/`SkillActiveResponse`/
  `SkillLearningError`）+ `StatusResponse.skills`（`{enabled, pendingProposals, activeSkills}`）；
  desktop `agent.status()` 暴露 `skills`；preload 6 个 skill API 从 `Promise<unknown>` 收窄到精确联合类型。
- **测试**：`ipc-handlers.spec.ts` +4（未启用统一 `skills_learning_not_enabled`、注册全 skill channel、
  完整链路 distill→list→approve→active + `status.skills` schema 校验、approve 不存在 id → not_found）。

## [Unreleased] —— 技能系统 Phase 2：自学习闭环（outcome 加权 + 轨迹蒸馏 + 人审批 + DB 源） (2026-07-07)

> 把 Phase 1 的「静态作者能力包」升级为「越用越好 + 会长出新技能」：每次 `agent.run()` 的
> outcome 反馈给命中技能加权（现有技能越用越准），成功轨迹蒸馏成候选技能提案（pending），
> 人审批（HTTP/IPC）通过后写入 DB 源并**立即重载注册表**生效。复用飞轮既有模式——加权抄
> classifier `reinforce(outcomeSignal)`、蒸馏/审批抄 dormant `propose→approve→inject`、持久化抄
> storage-sqlite「接口在领域包、实现在 storage 包」。opt-in 分级 `OPENINTJ_SKILLS_LEARN=1`
> （隐含 `OPENINTJ_SKILLS`）默认关 → 默认行为零变化。详见 `docs/architecture/next-session.md` §十一
> 与 `docs/architecture/phase-skills-design.md`。

### Added

- **`@openintj/skills` 自学习核**：
  - `SkillStore` 接口 + `InMemorySkillStore`（`loadAll`/`upsertProposal`/`upsertApprovedSkill`/
    `removeApprovedSkill`/`saveWeight`/`clearAll`/`close`）；新类型 `SkillProposal`、`SkillWeight`。
  - `SkillLearningRuntime` 门面：`hydrate` / `noteSelected` / `recordOutcome`（对命中技能
    `reinforce(skillOutcomeSignal(status))`，有界 clamp、写穿 store）/ `weightFor` /
    `distill`（LLM 或启发式聚类产候选，跨次按 candidate id 去重）/ `listProposals`/`approve`/
    `reject`/`revoke`（approve/revoke 触发 `onSkillsChanged` 重载）/ `listApproved` / `close`。
    `skillOutcomeSignal` 与 classifier 同映射（本地实现，不反向依赖 classifier）。
  - `DbSkillSource`：把已审批技能供给 `SkillRegistry`，与 `FsSkillSource` 并列（后源同 id 覆盖）。
  - `createLlmSkillDistiller`：用 agent LLM 把成功轨迹蒸馏成 `SKILL.md` 草案（JSON 容错解析，
    失败自动回退启发式）。
- **`SqliteSkillStore` + `createSqliteSkillStore`**（`@openintj/storage-sqlite`）：`skill_approved` /
  `skill_proposals` / `skill_weights` / `skill_schema_version` 表，better-sqlite3 动态 import + WAL +
  版本化迁移 v1 + upsert 写穿 + zod 校验 `loadAll`；默认库 `<dataDir>/skills.sqlite`。
- **可观测**：`HookEventMap` 新增 `event.SKILL_PROPOSED`（`{ proposalId, skillId, evidenceCount }`）；
  `attachOtelToHooks` 新增 counter `openintj.skill.proposed`（每蒸馏一个候选 +1，attribute=skill）。
- **审批入口**：server HTTP `POST /api/skills/distill`、`GET /api/skills/proposals?status=`、
  `POST /api/skills/proposals/:id/{approve,reject,revoke}`、`GET /api/skills`（生效技能 + 权重）；
  desktop IPC 镜像（`SKILLS_DISTILL`/`LIST`/`APPROVE`/`REJECT`/`REVOKE`/`ACTIVE` + preload）。
  未启用统一返回 `skills_learning_not_enabled`（HTTP 503）。**桌面审批 UI 面板本期不做**（后续抄 `DormantPanel`）。
- **测试**：`skills/__tests__/{store,learning-runtime,db-source}.spec.ts` + 扩 `selector.spec.ts`
  （weight 偏置改排序 + 封顶）；`storage-sqlite/__tests__/skills.spec.ts`（`:memory:` 往返）；
  `telemetry-otel/__tests__/metrics.spec.ts` 加 skill.proposed 例；`server/__tests__/skills-learning-wiring.spec.ts`
  （默认关 503 / 蒸馏→审批→生效链路 / env 开关）。

### Changed

- **`SkillSelector`**：`SkillSelectorOpts` 新增可选 `weightFor` + `weightGain`（默认 0.05）+
  `weightBiasCap`（默认 0.3）；最终分加**有界权重偏置**（语义余弦仍主导，权重不压过相关度）。
- **`assembleSkillContext`**：新增 `extraSources`（接 `DbSkillSource`）/ `weightFor` / `onSelected`
  回调，`SkillContext` 新增 `reload()`（重载来源 + 重嵌入 + 清空命中缓存，供 approve/revoke 后立即生效）。
- **三端 agent 装配**（cli/server/desktop）：新增 `enableSkillLearning` opt（env `OPENINTJ_SKILLS_LEARN=1`，
  隐含开启 `enableSkills`）；real 模式挂 `SqliteSkillStore` 否则 `InMemorySkillStore`；构建
  `SkillLearningRuntime` → `hydrate` → 注册表加 `DbSkillSource` + 传 `weightFor`/`onSelected`/`onSkillsChanged`；
  `run()` 收尾（classifier.reinforce 旁）加 `skillLearning.recordOutcome`；`close()` 关 store。
  `@openintj/storage-sqlite` 新增对 `@openintj/skills` 的依赖与项目引用。

## [Unreleased] —— 技能系统 Phase 1：作者能力包（SKILL.md）按需注入 (2026-07-01)

> 把可复用的做法沉淀成**能力包**（`SKILL.md`）：每轮 query 经「目录 + 嵌入检索」两级预筛，
> 命中才把技能全文注入 system prompt（省 token），未命中零注入。可插拔 `SkillSource` 为
> Phase 2 自学习技能预留同一接口。opt-in 开关 `OPENINTJ_SKILLS=1` 默认关 → 默认行为零变化。
> 详见 `docs/architecture/next-session.md` §十一 与 `docs/architecture/phase-skills-design.md`。

### Added

- **新包 `@openintj/skills`**：`Skill` 类型 + 可插拔 `SkillSource` 接口 + `FsSkillSource`
  （递归发现并解析 `SKILL.md`：极简 frontmatter 解析不引 YAML 依赖，`id/name/description/triggers?/taskTypes?/priority?/version?` +
  body 正文；非法 `taskType` 过滤、缺 description/body 跳过、id 兜底目录名、同 id 后源覆盖）。
  `resolveSkillDirs`（内建 + `OPENINTJ_SKILLS_DIR`，分号/逗号分隔且不切 Windows 盘符）、
  `builtinSkillsDir()`（用包自身 `import.meta.url`，src/dist 都指向 `../skills`）。
- **`SkillRegistry` + `SkillSelector` + `renderSkillPrompt`**：注册表用注入 embedder 预计算
  「name+desc+triggers」向量 + 轻量目录；选择器 = embed 余弦 + trigger 关键词加成 + taskType 加成，
  过阈值（默认 0.35）取 top-k（默认 2），正文按 token 预算封顶（默认 700，至少留最高分一个）。
- **共享装配 helper `assembleSkillContext`**：三端共用，载入 + 选择器 + 按 (taskType,query) 记忆化
  （上限 128 清空）+ 命中发 `event.SKILL_SELECTED`；无可用技能返回 `undefined`（调用方零注入）。
- **种子技能**（`packages/skills/skills/`，随包发布）：`code-review` / `web-research` / `debugging`。
- **可观测**：`HookEventMap` 新增 `event.SKILL_SELECTED`（`{ skills:{id,score}[]; query }`）；
  `attachOtelToHooks` 新增 counter `openintj.skill.hit`（每次注入的每个技能各 +1，attribute=skill）。
- **测试**：`skills/__tests__/{fs-source,selector}.spec.ts`（13）、`telemetry-otel/__tests__/metrics.spec.ts` 加 skill.hit 例。

### Changed

- **三端 agent 装配 + `contextProvider`**（cli/server/desktop）：新增 `enableSkills` opt（env `OPENINTJ_SKILLS=1`）；
  命中技能块拼在 **persona 之后、`[记忆参考]` 之前**（CLI 无 persona 则接 base），复用 store embedder，按 query 记忆化避免多轮重复 embed。
  三端 `package.json` / `tsconfig.json` 加 `@openintj/skills` 依赖与引用；`pnpm-workspace.yaml` / 根 `tsconfig.json` 加新包。

## [Unreleased] —— Memory Flywheel: 增量检索 + 长跑验证 + 可强化分类器 (2026-06-30)

> 把「记忆」「检索」「分类」串成一个共享使用反馈的飞轮：每次 `agent.run()` 的
> (query → outcome) 信号同时喂给会话级增量检索索引与可强化分类器，让两者一起「越用越好」。
> 三个 opt-in 开关默认全关 → 默认行为零变化。详见 `docs/architecture/next-session.md` §十。

### Added

- **A1 记忆写入 change-feed**：`HookEventMap` 新增 `event.MEMORY_WRITTEN`
  （`{ fragment, op: "add" | "update" | "remove" }`）；`MemoryStore` 在 `add*` / `remove` /
  短期溢出晋升（`op:"update"`）/ 工作记忆溢出丢弃（`op:"remove"`）处发出该事件，
  `PersistentMemoryStore.reassignMemoryType` 同步补 `op:"update"`。hydrate 直推不发事件（用 `index()` 种子）。
- **A1 会话级增量混合索引 `MemoryHybridIndex`**（`@openintj/taskpool`）：订阅 `event.MEMORY_WRITTEN`
  做增量 `upsert`/`remove`，替代每次查询全量 `index()` 重建；支持 `memoryTypes`/`taskTags` 过滤。
  三端（cli/server/desktop）装配后 `seed()` + `subscribe()`，`close()` 退订。
- **A1 `ContextEngine.candidateRetrieve` 注入点**：opt-in `OPENINTJ_LOOP_HYBRID=1` 时主循环
  改走 hybrid 候选召回（`fragmentsToRanked` 把命中转回 `RankedMemory`，仍过 ShaderPipeline /
  taskType boost / accessCount）；默认仍走 `MemoryRetriever`。
- **A2 长跑验证 harness `@openintj/shared/longrun-eval.ts`**：`runLongRunSession` 逐轮记录命中/
  token/judge + 改进曲线（后半 vs 前半 recall）；`runLongRunAb` 多变体 A/B；
  `longrun-scenarios.ts` 提供有先后依赖的场景 fixtures；`formatLongRunRow/Turns/Ab` 控制台表。
  `apps/cli/__tests__/longrun.harness.spec.ts`（`RUN_LONGRUN=1` 门控）跑真实 agent + classifier-on/off A/B。
- **A2 飞轮可观测 counter**：`attachOtelToHooks` 新增 `openintj.retrieval.hit`（`event.MEMORY_LOADED`
  命中即 +1）与 `openintj.tokens.spent`（`event.LOOP_ITERATION` 累计 token）。
- **CLF 新包 `@openintj/classifier`**：`ReinforcingClassifier`（embed kNN/质心 classify + 软置信度 +
  低置信回退 `detectTaskType` 关键词启发式；`reinforce` 升/降权 exemplar + LRU 封顶）+ 种子 `DEFAULT_SEEDS`
  + 路由 `decideRoute`（高置信简单类 → `enableReact:false` 降 token）/ `outcomeSignal`（status → 反馈信号）。
- **CLF 分类器持久化**：`ClassifierStore` 接口 + `InMemoryClassifierStore`（默认）+
  `SqliteClassifierStore`（`@openintj/storage-sqlite`）；装配时 `hydrate()`，`reinforce`/`addSeeds` 后落盘。
- **外部联网搜索工具 `@openintj/plane-execution/web-search-tool.ts`**：`createWebSearchTool`
  （Tavily / Brave，provider 中立）+ `resolveWebSearchConfig`（按 `OPENINTJ_SEARCH_PROVIDER` /
  `OPENINTJ_SEARCH_API_KEY` / `TAVILY_API_KEY` / `BRAVE_API_KEY` 推断）。三端 `search` 工具优先级：
  外部 Web Search > 混元内建（仅旧平台有效）> 占位。失败不抛错（工具语义）；不配 key 零开销。
  起因：旧混元平台内建搜索随平台 2026-06-22 下线，TokenHub 改 Responses API 独立产品。
  测试 `web-search-tool.spec.ts`（10）。

### Changed

- **`TaoLoop.run()` 新增可选 `taskType` / `enableReact` / `topK` opts**：外部预分类时跳过内部分类、
  并按路由决定是否退化为单次 LLM；`topK` 透传给 `contextProvider`（`TaoContextInput.topK`）→
  `ContextEngine.build` → 检索，让「高置信简单类调小 topK 降 token」真正落到检索调用（此前 `RouteDecision.topK`
  仅计算未接入）。`TaoResult` / `ctx.metrics` 新增 `totalTokensSpent`（跨轮累计）。
  `detectTaskType` 提升为公开导出供分类器复用。
- **`MemoryPlane.recordUserInput/Output` 接受可选 `extraTags`**：把分类 label 写进 `taskTags`，
  与 retriever 的 taskType boost 叠加、随使用复利。
- **三端 agent 装配 + `run()`**（cli/server/desktop）：新增 `enableClassifier` opt（env `OPENINTJ_CLASSIFIER=1`）；
  `run()` 预分类 → 注入 taskType + 降 token 路由（`enableReact:false` 单次 LLM + `route.topK` 调小检索）→
  记忆带 label → 收尾 `reinforce(outcomeSignal(status))`。
  real 持久化模式自动挂 `SqliteClassifierStore`（`<dataDir>/classifier.sqlite`），`close()` 关闭。
- **`HybridRetriever.search` 支持 per-query `configOverride`**：会话级共享实例下仍可按查询覆盖融合参数；
  server `retrieveHybrid` / desktop `buildHybridRetrieve` 改用共享 `MemoryHybridIndex`，不再每查询重建。

### Fixed

- **#11 Dormant `dormant_events` 磁盘表无限增长**：自动清理此前只在 `mine()` 末尾跑，而 `mine()` 仅由用户
  显式触发（server `POST …/dormant/mine`、desktop `DORMANT_MINE`），长会话不 mine 时磁盘表照涨。
  `DormantRuntime` 现新增两处不依赖 mine 的兜底触发：`hydrate()` 启动末尾清一次（重启即收敛）、
  `record()` 每累计 `autoPruneEveryNEvents` 条清一次（配了 `eventRetentionMs`/`maxDiskEvents` 时默认 256，
  显式 `0` 关闭）。server/desktop 装配默认 `maxDiskEvents: 50_000`。`persistence.spec.ts` 补 3 例。

## [Unreleased] —— hotfix bundle #2 (2026-05-20 → 2026-05-21)

> 这是 alpha.8 之后的第二批 hotfix，主要解决 Windows 真盘启动链路上的三个独立坑。

### Added

- **`@openintj/shared` 新增 `loadOpenintjEnv()` + `summarizeLlmEnv()`**
  （`packages/shared/src/env.ts`）：
  - 走 Node 21.7+ 原生 `process.loadEnvFile`，不引入 dotenv 依赖
  - 从入口起点 **逐级向上** 找 `.env.local` / `.env`，直到 `.git` 根；
    支持本仓库的「外层 `F:\openINTJ\.env`+ 内层 `F:\openINTJ\ts\pnpm-workspace.yaml`」混合布局
  - 先加载的优先（已存在 `process.env` 永远最高优先级）
  - `summarizeLlmEnv()` 把 LLM 配置浓缩成单行日志，**绝不打印 API Key 本体**
  - 9 个 vitest spec 覆盖（多层目录 / `.env.local` 优先级 / shell env 不被覆盖 / key 不泄漏）
- **`vitest.global-setup.ts`** —— 跑测试前自动把 `better-sqlite3` 切回 Node ABI
- **`apps/desktop/scripts/ensure-electron-abi.cjs`** —— `predev` / `prepackage` 钩子，
  跑 Electron 前自动把 `better-sqlite3` 切到 Electron ABI

### Changed

- **CLI / server / desktop 三个入口启动时都自动 `loadOpenintjEnv()`** 并打印 LLM 摘要
  - `apps/cli/src/index.ts`、`apps/server/src/index.ts`、`apps/desktop/src/main/index.ts`
  - `.env.example` 文档承诺的"自动加载 .env"现在真生效；以前是没人写 loader
- **桌面端启动加 Chromium 命令行开关静音后台探测**
  （`disable-background-networking` / `disable-features=SafeBrowsing,NetworkTimeServiceQuerying,DialMediaRouteProvider,MediaRouter,OptimizationHints,Translate,InterestFeedContentSuggestions` / `disable-component-update` / `disable-domain-reliability`）
  - 干掉了 `ssl_client_socket_impl.cc handshake failed; net_error -107` 类噪音日志
  - opt-out：`OPENINTJ_DESKTOP_KEEP_BG_NET=1`

### Fixed

- **Desktop dev/prod Electron 启动崩在 better-sqlite3 NODE_MODULE_VERSION 不匹配**（继续修）：
  - 上一版改成 `postinstall: electron-builder install-app-deps`，结果发现两个隐藏问题：
    1. `electron-builder install-app-deps` 在 pnpm 布局里**报 finished 但实际不替换 .node 文件**；
       现在直接走 `prebuild-install --runtime=electron --target=33.x --force`
    2. 把 binding 切到 Electron ABI(130) 后，所有走 Node ABI(127) 的 vitest 都 dlopen 失败 →
       原 postinstall 彻底废，改成**双向自愈**：
       - **predev 钩子** 跑 `apps/desktop/scripts/ensure-electron-abi.cjs`，在 `pnpm desktop:dev` /
         `pnpm desktop:package` 前自动确保 binding = Electron ABI
       - **vitest globalSetup** 跑 `vitest.global-setup.ts`，在 `pnpm test` 前自动确保 binding = Node ABI
       - 两边都用 **子进程 probe** 来读 ABI 状态（关键：本进程不能 `require('better-sqlite3')`，
         否则 Windows 下 .node 句柄被锁住，prebuild-install EBUSY）
- **".env 没人加载" 静默坑** —— `.env.example` 写着会自动加载，但 cli/server/desktop 三个入口都没人 `dotenv.config()`，结果 `LLM_PROVIDER=hunyuan` 永远走不通；
  现在三处都接 `loadOpenintjEnv()` 自动 fix
- **`packages/shared` 此前只是一个 `__sharedPlaceholder` 占位**，本次扩展成真正的跨入口工具包

## [3.0.0-alpha.8] —— Phase 3.8 Hooks → OpenTelemetry (2026-05-20)

> 给 hooks 系统补一条官方观测出口：自动把 TAO / ReAct / Tool / Policy 事件
> 翻译成 OpenTelemetry span 树 + counter metric。业务零侵入；未启用零开销。
> 详见 [`docs/architecture/phase3-8-otel.md`](./docs/architecture/phase3-8-otel.md)。

### Added

- **新包 `@openintj/telemetry-otel`** —— Hook 事件 → OTel 适配
  - `attachOtelToHooks(bus, opts)` —— 订阅 hook 事件，per-traceId 维护
    iteration / action / tool span 帧栈；返回 `dispose()`
  - `bootstrapNodeOtel(opts)` —— 可选 SDK 引导（懒 import `sdk-trace-node` +
    `exporter-trace-otlp-http`；缺包才抛错，不影响 attach 零开销路径）
  - Span 树：`openintj.tao.iteration` → `openintj.react.action` → `openintj.tool.call`
  - Counter：`openintj.tao.iterations`、`openintj.react.actions`、`openintj.tool.calls`、
    `openintj.tool.errors`、`openintj.policy.blocked`、`openintj.memory.loaded`
  - SDK 全标 `peerDependenciesMeta.optional: true`，consumer 不调 bootstrap 就不用装
- **`__tests__/{noop,spans,metrics,dispose}.spec.ts`** —— 10 个新测试
  - 未注册 provider 时 0 错、0 span（零成本路径）
  - InMemorySpanExporter 断言 parent/child 关系 + ERROR 状态 + recordException
  - InMemoryMetricExporter 断言 6 个 counter 累计
  - `dispose()` 兜底 end 未结束 span + unregister 所有 handler
- **`apps/server/__tests__/otel-wiring.spec.ts`** —— 4 个 wiring 测试：
  代码 / env / 默认关 / 显式关 4 条路径都跑一遍真实 `agent.run()`
- **`docs/architecture/phase3-8-otel.md`** —— 阶段记录 + 选型 + 6 类陷阱

### Changed

- **`ts/apps/server/src/agent.ts`**：
  - `ServerAgentOpts.enableOtel?: boolean | AttachOtelOpts`
  - `resolveOtel(opts)`：bool / object / `OPENINTJ_OTEL=1` env 三通道
  - `ServerAgent.otel?: AttachedOtel`；`agent.close()` 调 `otel.dispose()`
- **`ts/apps/desktop/src/main/agent.ts`**：镜像 server 端装配
- **`ts/pnpm-workspace.yaml`**：加 `packages/telemetry/*`
- **`ts/tsconfig.json`**：refs 加 `packages/telemetry/otel`
- **`ts/apps/{server,desktop}/{package.json, tsconfig.json}`**：依赖 + ref
- **`ts/apps/server/package.json`**：devDep 加 `@opentelemetry/{api,sdk-trace-base}`
  （仅 wiring 测试用；运行时不需要）

### Testing

- 本地（Windows 11 / Node 22）：
  - `pnpm lint` exit 0（仍是 2 条 pre-existing useExhaustiveDependencies warn）
  - `pnpm exec turbo run typecheck --concurrency=1` → 35/35 successful
  - `pnpm exec turbo run test --concurrency=1` → 35/35 successful，
    **444 passed + 11 skipped**（净增 14：10 telemetry-otel + 4 server-wiring）

### Notes

- **零成本默认**：`enableOtel` 不真就根本不调 `attachOtelToHooks`；
  启用但未注册 TracerProvider 时 OTel API 返回 NoopTracer/NoopMeter，
  span / counter 都是空对象操作（setAttribute / add 是 noop）
- **HookBus traceId 是 UUID，OTel traceId 是 hex 128-bit**：不相同！
  本适配器把 HookBus traceId 写到 `trace_id` span 属性，方便反查
- **bootstrapNodeOtel idempotent**：用 ProxyTracerProvider 探针检测（traceId 全零）

## [3.0.0-alpha.7] —— Phase 3.7 Desktop E2E (Playwright + Electron) (2026-05-20)

> 给桌面端 renderer 补上最后一层端到端兜底——用 Playwright `_electron.launch`
> 启动真主进程 + 真 BrowserWindow，对真 DOM 做 7 个用例的断言。详见
> [`docs/architecture/phase3-7-desktop-e2e.md`](./docs/architecture/phase3-7-desktop-e2e.md)。

### Added

- **`ts/apps/desktop/e2e/`** —— Playwright 端到端套件
  - `playwright.config.ts` —— workers=1；`OPENINTJ_PLAYWRIGHT=1` 才执行，
    默认 `testIgnore: ["**/*"]` 保证不污染主 CI 路径
  - `fixtures.ts` —— `electronApp` + `page` fixture，默认 env：
    `LLM_PROVIDER=mock` + `OPENINTJ_DESKTOP_NO_PERSIST=1`
  - `tests/smoke.spec.ts` —— **5 tests**：app 启动 / header / status bar /
    chat 全链路（你好 → mock greet）/ trajectory 计数 / dormant tab 默认未启用
  - `tests/dormant.spec.ts` —— **2 tests**（`OPENINTJ_DORMANT=1`）：
    mine 按钮可见 + pending filter / 点 Mine 出现扫描摘要
  - `tsconfig.json` —— 独立 e2e 项目，不污染 `src/` 编译
- **`.github/workflows/ci.yml`** —— 新 job `e2e-desktop`（Ubuntu + xvfb），
  独立于 `e2e-persistence`，构建 desktop bundle → xvfb 包 Playwright →
  失败时 upload `playwright-report/`
- **`docs/architecture/phase3-7-desktop-e2e.md`** —— 阶段记录 + 选型 + 两个坑 + CI 集成

### Changed

- **`ts/apps/desktop/src/main/index.ts`**：preload 路径
  `../preload/index.js` → `../preload/index.mjs`
  - electron-vite 默认产物是 `.mjs`，路径不对会让 `window.openintj` 永远 undefined
  - 历史 vitest 走 mock electron 路径不触发该 bug，Playwright 真启动才暴露
- **`ts/apps/desktop/package.json`**：
  - devDep 加 `@playwright/test ^1.60.0`
  - `typecheck`：串第二段 `tsc --noEmit -p e2e/tsconfig.json`
  - 新 script `e2e`（build + run）/ `e2e:run`（只 run）
- **`ts/biome.json`**：`files.ignore` 加 `**/test-results/**` 与 `**/playwright-report/**`
  （Playwright 运行产物）

### Testing

- 本地 Windows（Node 22）`pnpm --filter @openintj/desktop run e2e`：
  - **7/7 passed**（34.8s）—— 5 smoke + 2 dormant
- 默认 CI 路径（不设 `OPENINTJ_PLAYWRIGHT`）：
  - `pnpm lint` exit 0（仍是 2 条 pre-existing useExhaustiveDependencies warn）
  - `pnpm exec turbo run typecheck --concurrency=1` → 33/33 successful
  - `pnpm exec turbo run test --concurrency=1` → 33/33 successful，
    **430 passed + 11 skipped**（与 alpha.6 持平，未引入新 unit）

### Notes

- 两个值得记的坑（详见 phase3-7 §四）：
  1. **electron-vite 输出 `.mjs` preload**：main 写死 `.js` 路径，silent fail，
     直到真 Electron 启动才暴露
  2. **Windows + Playwright `_electron.launch` 加 `--no-sandbox` 卡 30s 超时**：
     只在该具体组合下出现；Linux + xvfb 不需要 flag
- Playwright 跑包采用 `_electron` 模式，没有装 Chromium / Firefox / WebKit；
  CI 也跳过 `playwright install`，依赖大小可控
- 桌面端**渲染层第一次有机器化兜底**；之前只有 IPC contract 测试（21 个）

## [3.0.0-alpha.6] —— Phase 3.6 Python v2 ↔ TS 行为对齐测试 (2026-05-20)

> 给 TS 实现盖一层"行为级回归网"——把冻结的 Python v2.0 当语义参考，
> 在固定输入上断言 TS 输出等价。详见
> [`docs/architecture/phase3-6-parity-tests.md`](./docs/architecture/phase3-6-parity-tests.md)。

### Added

- **`scripts/python-parity/generate_fixtures.py`** —— Python 端取证脚本
  - 加载仓库根冻结的 `framework_core` / `memory_plane` / `control_plane` / `execution_plane`
  - 在预设输入上跑 → 把可观察输出固化为 4 份 JSON fixture（每个 TS 包一份）
  - **只读**：绝不修改 Python 代码；Python v2 已冻结
- **`scripts/python-parity/README.md`** —— 工具使用说明 + 已知偏差速查表
- **4 个 TS parity spec**（共 **64 个新 tests**）：
  - `ts/packages/core/__tests__/parity/python-v2.spec.ts` —— **23 tests**：
    SimpleEmbedder (SHA-256) / cosineSimilarity / decayImportance
  - `ts/packages/planes/control/__tests__/parity/python-v2.spec.ts` —— **21 tests**：
    GoalParser.parse 中英文意图 + 引号实体 + 优先级；Planner.createPlan 5 个公共 intent
  - `ts/packages/planes/execution/__tests__/parity/python-v2.spec.ts` —— **17 tests**：
    StepStateMachine 合法/非法转换表；Executor sequential / parallel 事件轨迹
  - `ts/packages/planes/memory/__tests__/parity/python-v2.spec.ts` —— **3 tests**：
    MemoryStore overflow；MemoryRetriever 评分组件 + 排序
- **4 份 fixture JSON**（`__tests__/parity/fixtures/python-v2.json`）：
  - 每份带 `schemaVersion` + `generatedFrom` + 关键设计 `notes`
  - 由 Python 端脚本统一生成，commit-in，CI 无需 Python
- **`docs/architecture/phase3-6-parity-tests.md`** —— 阶段记录 + 已知偏差矩阵 + 容差策略

### Changed

- **`ts/biome.json`**：`files.ignore` 加 `**/__tests__/parity/fixtures/**`
  （fixture 是 Python 产物，不参与 biome formatter）

### Testing

- CI 模式（`pnpm exec turbo run test --concurrency=1`）：
  - 33/33 packages successful
  - **430 passed + 11 skipped**（净增 64 个 parity 测试；previously 366 + 11）
- E2E 模式（`OPENINTJ_E2E=1`）：
  - 33/33 packages successful
  - **441 passed + 0 skipped**（previously 377 + 0）

### Notes

- 容差策略（详见 phase3-6 文档）：
  - SHA-256 向量 / cosineSimilarity：`1e-12`（bit-identical）
  - `decayImportance`：`1e-4`（Python 用 `0.693` 近似 `Math.LN2`；TS 更精确）
  - MemoryRetriever 评分：分量 `1e-12`（纯位运算）/ recency + 最终 score `1e-4`
- 5 类**已知偏差**已显式记录在 phase3-6 文档"已知偏差矩阵"：
  1. `decayImportance` 0.693 vs `Math.LN2` —— TS 精度更高
  2. MemoryRetriever 半衰期口径（Python 写死 `max_summary_length/10` 是 v2 bug；
     fixture 把 `max_summary_length=240` 让两边都跑 24h 半衰期，严格可比）
  3. Planner `delete`/`execute` 模板 —— TS 扩展；parity 只跑公共 5 个 intent
  4. Executor 死重试 bug —— TS 已修复；fixture 只跑全成功路径
  5. StepStateMachine 错误码命名 —— TS spec 接受两者之一
- fixture 一次生成、长期复用；只有 Python 端"延寿活动"（极少）或 `generate_fixtures.py`
  自身改动时才需要重跑

## [3.0.0-alpha.5] —— Phase 3.5 Dormant 审批 UI (2026-05-19)

> Phase 3.4 把 Dormant 持久化的模型/装配/IPC 都做完了，但桌面端 renderer 没接。
> 这一版把最后一公里补上：preload 暴露 5 个 dormant API + 桌面端审批面板 +
> StatusBar 暴露 dormant pending 角标。详见
> [`docs/architecture/phase3-5-dormant-approval-ui.md`](./docs/architecture/phase3-5-dormant-approval-ui.md)。

### Added

- **`apps/desktop/src/shared/ipc-protocol.ts`** —— 把 IPC 协议补成"所见即所得"
  - `StatusResponseSchema` 补 `persistence` / `retrievalMode` / `dormant` 三个 optional
    字段（与 main 进程 `agent.status()` 实际返回对齐；之前 renderer 端类型早就过期）
  - 新增响应 DTO：`DormantMineResponseSchema` / `DormantListResponseSchema` /
    `DormantDecisionResponseSchema` / `DormantPersonaResponseSchema` /
    `DormantProposalDtoSchema` / `DormantPatternDtoSchema`
  - 新增错误 schema：`DormantErrorSchema` / `DormantDecisionErrorSchema`
- **`apps/desktop/src/preload/index.ts`** —— 5 个新 API：
  - `dormantMine()` / `dormantList(req?)` / `dormantApprove({ proposalId })` /
    `dormantReject({ proposalId })` / `dormantPersona()`
  - 返回类型是联合类型（success | error），renderer 必须 narrow 才能用，无法把错误当数据
- **`apps/desktop/src/renderer/components/DormantPanel.tsx`** —— 新组件
  - 顶栏 [Mine] 按钮 + 状态 filter（pending/applied/rejected/all）
  - proposal 列表卡片：状态徽章 + 频次 + 置信度 + 描述 + `targetField ← value` +
    [✓ 应用] [✗ 拒绝] 按钮
  - 底部折叠区：当前 Persona JSON
  - 未启用时显示居中提示 + 启用方法
- **`apps/desktop/src/renderer/App.tsx`** —— 右侧栏从单 panel 改成 tab 布局
  - tab 标题：[推理轨迹] [Dormant + pending 数字角标]
  - tab 角标：`status.dormant.pendingProposals > 0` 时显示黄色徽章
- **`apps/desktop/src/renderer/components/StatusBar.tsx`**
  - 新增条目：检索模式 / 持久化模式 / Dormant 状态（passive 事件数 + 待审 proposal 数）
  - 类型从本地 `StatusSnapshot` interface 改为 protocol 中的 `StatusResponse` re-export

### Changed

- `TrajectoryPanel.tsx` 和 `DormantPanel.tsx` 去掉外层 border/bg/header 装饰
  （这些 chrome 现在由 App.tsx 的 tab 容器统一提供，避免双层 border）

### Testing

- `apps/desktop/__tests__/ipc-handlers.spec.ts`：12 → **18 tests passed**（+6）
  - STATUS 用 StatusResponseSchema 全字段校验（含 dormant + retrievalMode + persistence）
  - DORMANT_MINE 用 DormantMineResponseSchema 校验返回值结构
  - DORMANT_LIST 默认（无 status）返回所有 proposals + 字段校验
  - DORMANT_REJECT 返回 status=rejected + 不污染 persona
  - APPROVE/REJECT 不存在 proposalId 时返回 `not_found_or_already_decided`
  - DORMANT_PERSONA 未启用时返回 `dormant_not_enabled`

### Notes

- 当前 desktop 工作区只有 main-process 测试，**没有 renderer React 测试**
  （DormantPanel 的逻辑分支已被 IPC 层契约测试覆盖；UI 留给手动 / Playwright e2e #4）
- Mine 任务由用户主动触发；后台 mine 推送（`EVT_DORMANT`）留给未来需要时再做
- 用户唯一的 persona 写入路径仍然是审批 proposals，UI 不提供字段级直接编辑

---

## [3.0.0-alpha.4] —— Phase 3.4 Dormant 持久化 (2026-05-19)

> Phase 3.3 留下的最重的尾巴：PassiveStore / PersonaConfig 进程一断电就丢。
> 这一版给 Dormant 子系统接上 SQLite 真盘适配器，
> 把"用户审批过的偏好/习惯"留下来。详见
> [`docs/architecture/phase3-4-dormant-persistence.md`](./docs/architecture/phase3-4-dormant-persistence.md)。

### Added

- **`@openintj/dormant`**
  - 新增 `DormantPersistenceAdapter` 接口（`persistence.ts`）：`loadAll` / `recordEvent` /
    `upsertProposal` / `savePersona` / `clearAll` / `close`，热路径同步、不抛错
  - 新增 `InMemoryDormantStore`：参考实现 + 测试用
  - 新增 `DormantSnapshot` 类型
  - `DormantRuntime`：新增 `adapter` 槽 + `hydrate()` 方法；`record / mine / approve / reject /
    reset / close` 全部写穿 adapter
  - `PassiveStore`：新增 `recordBulk(events)` 批量回填
  - `InternalizationManager`：新增 `restoreState(proposals, persona?)` 不触发
    `lastUpdated` / `version` 自增
- **`@openintj/storage-sqlite`**
  - 新增 `SqliteDormantStore`：实现 `DormantPersistenceAdapter`，独立的 `dormant.sqlite`
    文件，schema v1（`dormant_events` / `dormant_proposals` / `dormant_persona`）+ WAL +
    prepared statements
  - 新增 `createSqliteDormantStore` 工厂；输入类型为 `SqliteDormantConfigInput`（`wal` 可选）
- **apps/server**
  - `ServerAgentOpts` 新增 `dormantPersistence: 'auto' | 'memory' | 'real'`（默认 `auto`）+
    `dormantDbPath`（覆盖默认 `${dataDir}/dormant.sqlite`）
  - `ServerAgent` 新增字段 `dormantPersistenceInfo: { adapter, dbPath? }`
  - `status().dormant.persistence` 暴露 adapter 名 / 路径
  - `assembleServerAgent`：在 `enableDormant + dataDir` 时自动挂 SqliteDormantStore，
    构造后 `await dormant.hydrate()`；`close()` 先 `await dormant.close()`
- **apps/desktop**
  - `DesktopAgent` 镜像同上：`dormantPersistence` / `dormantDbPath` / `dormantPersistenceInfo`
    / `status().dormant.persistence`
- **环境变量**
  - `OPENINTJ_DORMANT_DB_PATH` 可覆盖默认 SQLite 文件路径

### Changed

- `@openintj/storage-sqlite/index.ts` 修复重复 `export * from "./dormant.js"`
- biome formatter 一次性整理 4 个文件（`packages/storage/sqlite/tsconfig.json` 等）

### Testing

- **CI 模式**：
  - `packages/dormant/__tests__/persistence.spec.ts`：9 个（InMemoryDormantStore CRUD +
    hydrate + write-through）
  - `packages/storage/sqlite/__tests__/dormant.spec.ts`：11 个（`:memory:` 路径走真
    better-sqlite3）
  - `apps/server/__tests__/dormant-persistence-e2e.spec.ts`：2 个 memory 模式（4 个 e2e skip）
- **E2E 模式（`OPENINTJ_E2E=1`）**：上述 e2e 6 个全部跑通，含 `record → mine → approve →
  close → 重装配 → hydrate → 验证状态恢复` 的完整往返

### Notes

- 桌面端审批 UI 仍未接（#9.B 留给下一个 phase）
- `dormant_events` 表未做自动清理；当前 PassiveStore 仅内存层有 `maxPassiveEvents` 环形上限
- `dormant.sqlite` 是明文；用户敏感偏好不应通过 dormant 路径学习

---

## [3.0.0-alpha.3] —— Phase 3.3 RFC-003 装配进主 Agent (2026-05-11)

> RFC-003 的三个孤岛包（@openintj/concurrency / @openintj/dormant / @openintj/taskpool）
> 全部接进 apps/server 与 apps/desktop 主装配点，三条线均提供环境变量 / 代码 opt-in，
> 默认零开销，启用后能直接通过 HTTP / IPC 使用。

### Added

- **方向 1 — LLM 速率限制**（`@openintj/concurrency`）
  - 新增 `RateLimitedLlmClient`：TokenBucket 装饰 `LlmClient.chat / visionChat`
  - server / desktop opt-in：`opts.rateLimit = { qps, burst? }` 或 env `OPENINTJ_RATE_LIMIT_QPS` / `OPENINTJ_RATE_LIMIT_BURST`
- **方向 2 — HybridRetriever 混合检索**（`@openintj/taskpool`）
  - server: `retrieveHybrid()` 顶层函数 + 路由 `GET /api/memory?mode=hybrid[&rrf=true]`
  - desktop: `agent.retrieveHybrid()` + IPC `MEMORY_QUERY` 支持 `{ mode: 'hybrid', rrf }`
  - 默认检索模式 opt-in：`opts.retrievalMode = 'hybrid'` 或 env `OPENINTJ_RETRIEVAL_MODE=hybrid`
- **方向 3 — Dormant Memory Learning**（`@openintj/dormant`）
  - 新增 `DormantRuntime`：PassiveStore + PatternMiner + InternalizationManager 三件套门面
  - server 路由：`POST /api/dormant/mine` / `GET /api/dormant/proposals` / `POST /api/dormant/proposals/:id/approve|reject` / `GET /api/dormant/persona`
  - desktop IPC：`DORMANT_MINE / DORMANT_LIST / DORMANT_APPROVE / DORMANT_REJECT / DORMANT_PERSONA`
  - `agent.run()` 自动把用户输入和 final answer 喂进 PassiveStore（启用后才生效）
  - opt-in：`opts.enableDormant = true` 或 env `OPENINTJ_DORMANT=1`；未启用时所有 API 一律 503 / `dormant_not_enabled`

### Changed

- `apps/server/src/agent.ts` `ServerAgent` 新增字段：`retrievalMode` / 可选 `dormant` / `status().dormant` / `status().retrievalMode`
- `apps/desktop/src/main/agent.ts` `DesktopAgent` 镜像 server 端字段
- `apps/server/package.json` / `apps/desktop/package.json` 新增 workspace 依赖：`@openintj/concurrency` / `@openintj/dormant` / `@openintj/taskpool`
- IPC 协议 `apps/desktop/src/shared/ipc-protocol.ts` 扩展：
  - `MemoryQueryRequestSchema` 加 `mode` / `rrf`
  - 新增 `DormantListRequestSchema` / `DormantProposalDecisionSchema`
  - `IPC` 常量增加 5 个 Dormant channel

### Testing

- 新增测试（CI 模式）：
  - `@openintj/dormant`：`__tests__/dormant-runtime.spec.ts` 6 个
  - `@openintj/server`：`__tests__/dormant.spec.ts` 12 个 + `__tests__/hybrid-retrieve.spec.ts` 14 个 + `__tests__/rate-limited-llm.spec.ts` 9 个
  - `@openintj/desktop`：`__tests__/ipc-handlers.spec.ts` 扩展 5 个（hybrid + Dormant IPC）
- CI 跑分：默认 mode 312 passed / 7 skipped，E2E mode（`OPENINTJ_E2E=1`）全部跑通

### Design 备忘

- HybridRetriever 装配是"每次查询临时建索引"——适合中等规模（≤几千 fragments）；大规模建议改用 LanceDB FTS
- DormantRuntime 默认不持久化 PassiveStore 与 PersonaConfig；持久化层等下一个 phase 接入
- `RateLimitedLlmClient` 实现已经迁移到 `@openintj/concurrency` 包，`apps/server/src/rate-limited-llm.ts` 仅做兼容 re-export

---

## [3.0.0-alpha.2] —— Phase 3.2 GitHub Actions CI (2026-05-09)

> 把本地已经能跑通的 lint / typecheck / test (CI + E2E) 锁进 GitHub Actions，
> 给后续所有改动兜底。

### Added

- `.github/workflows/ci.yml`（仓库根，旧的错放在 `ts/.github/` 下从未触发，已删除）
  - **lint-and-typecheck**：matrix 跑 Node 20 + Node 22；先 biome lint，再 turbo typecheck
  - **test**：matrix 跑 ubuntu / windows / macos × Node 20；先 turbo build 再 turbo test（CI 模式）
  - **e2e-persistence**：仅 ubuntu，设 `OPENINTJ_E2E=1` 跑 LanceDB + SQLite 真盘端到端
  - 加 `concurrency.cancel-in-progress` 减少同分支重复跑
  - 全局 `NODE_OPTIONS=--max-old-space-size=6144` 防 tsc OOM
  - 全部 turbo 调用都带 `--concurrency=1`，统一跨 OS 的策略

### Changed

- `ts/turbo.json`：`test` 任务的 cache key 加入 `OPENINTJ_E2E` / `OPENINTJ_DATA_DIR` / `OPENINTJ_DESKTOP_NO_PERSIST` / `OPENINTJ_LANCE_DEBUG`
  - **关键修复**：之前 turbo 不感知这些 env 的变化，e2e job 会命中常规 test 的缓存、e2e 测试被默默跳过
- `ts/biome.json`：放宽与历史代码冲突的规则（`useLiteralKeys` / `noNonNullAssertion` / `noUnusedTemplateLiteral` / `noDelete` / `noArrayIndexKey` 等共 13 条）
  - 这些是**风格偏好**而非 bug；保留 `useImportType` / `noUnusedVariables` / `noUnusedImports` 等真正的正确性规则
  - 现状：`pnpm lint` exit 0，2 条 React `useExhaustiveDependencies` 警告（已知，不阻塞）
- biome formatter 一次性格式化 107 个 tsconfig.json / package.json（多行 references 数组改单行）

### Tooling

- 现在三条线都能本地一把跑通（也是 CI 跑的命令）：
  - `pnpm lint`
  - `pnpm exec turbo run typecheck --concurrency=1`
  - `pnpm exec turbo run test --concurrency=1`（默认 292，`OPENINTJ_E2E=1` 时 299）
- turbo cache key 修了之后：
  - 同 env 的二次运行：33/33 cache hit，full turbo ~500ms
  - 切换 `OPENINTJ_E2E` 取值：所有 test 任务 cache miss，重新执行

---

## [3.0.0-alpha.1] —— Phase 3.1 真实持久化 e2e (2026-05-09)

> Phase 3 第 1 步：把 `apps/server` / `apps/desktop` 从 in-memory 兜底切到真实磁盘
> （LanceDB + SQLite），并补端到端"写入 → 关闭 → 重启 → 读回"测试。
> CI 默认 292/292 绿（Phase 2 286 + 6 新增 in-mem）；`OPENINTJ_E2E=1` 全量 299/299 绿。

### Added

- **持久化工厂** `createPersistentMemoryStore`（`@openintj/plane-memory`）
  - 根据 `dataDir` / `mode` 自动选择 LanceDB+SQLite 真盘或 in-memory 兜底
  - 真盘模式自动建 `lancedb/` 子目录与 `metadata.db` 文件
  - 缺 `dataDir` 但 `mode='real'` 时显式抛错（fail-fast）
- **服务端入口** `assembleServerAgent({ dataDir?, persistenceMode? })`
  - 支持 env `OPENINTJ_DATA_DIR` 启用真盘
  - 新增 `agent.close()`（关 LanceDB / SQLite）与 `persistentInfo`
  - `/api/status` 暴露当前持久化模式与数据目录
- **桌面端入口** `assembleDesktopAgent({ dataDir?, persistenceMode? })`
  - Electron 主进程默认用 `app.getPath('userData')` 作 dataDir
  - `app.on('before-quit')` 钩 `agent.close()` 防止数据库句柄泄漏
  - env `OPENINTJ_DESKTOP_NO_PERSIST=1` 可强制走 in-memory（CI 友好）
- **e2e 测试**（`OPENINTJ_E2E=1` 启用）
  - `plane-memory/__tests__/persistence-factory.spec.ts`：工厂自身的真盘往返
  - `apps/server/__tests__/persistence-e2e.spec.ts`：装配 → 写 → close → 重装配 → hydrate → 检索 + 审计读回
  - `apps/desktop/__tests__/agent-persistence.spec.ts`：desktop agent 真盘往返与 NO_PERSIST 短路

### Changed

- `@openintj/storage-lance`：`apache-arrow` 从 `peerDependencies` 移到 `dependencies`（`init()` 必用，不是可选）
- `LanceDBVectorStore.init()`：从"靠 seed-row 推断 schema"改为用 `apache-arrow` 显式声明 `FixedSizeList<Float32, N>` + `List<Utf8>` schema；旧版 LanceDB 无 `createEmptyTable` 时回落到 seed-row + delete 路径
- `LanceDBVectorStore` 的 `delete` / `search` SQL：camelCase 列名一律双引号（LanceDB 大小写敏感，否则报 "No field named fragmentid"）
- `LanceDBVectorStore.search()`：新增 `normalizeEmbedding` / `normalizeStringArray`，把 LanceDB 返回的 TypedArray / Arrow Vector 规范化成 plain `number[]` / `string[]` 后再 `VectorRowSchema.parse`，修复"`count()` 返 N 但 `scanAll()` / `search()` 返空"的静默丢行 bug
- e2e suite 全部带 30 秒超时（`describe(..., { timeout: 30_000 }, ...)`），LanceDB 首次建表 + 重新打开偏慢

### Fixed

- 真实持久化模式下 vector search 返空数组（zod parse 因 TypedArray / Arrow Vector 静默失败）
- LanceDB SQL 过滤器对 camelCase 字段名报 "No field named fragmentid"
- `apache-arrow` 静态导入失败（peer 解析路径不一致）
- e2e 测试在重启第二个进程时因 5s 默认超时被误判为失败

### Tooling

- 调试用：设置 `OPENINTJ_LANCE_DEBUG=1` 时，`LanceDBVectorStore.search()` 会把 zod 解析失败的行打印到 stderr
- 本地真盘自检命令：`$env:OPENINTJ_E2E="1"; pnpm -r --workspace-concurrency=1 test`

---

## [3.0.0-alpha.0] —— Phase 2 完成 (2026-04-29)

> Phase 2 收尾：TS 端在 `v2.0-python-reference` 之上完成"装配 + 持久化 + 客户端 + RFC-003 三方向"四个纵深方向。
> typecheck 全绿；workspace 内 17 个测试包共 **286 个用例全部通过**（详见
> [`docs/architecture/phase2-complete.md`](./docs/architecture/phase2-complete.md)）。

### Added

- **Memory Shader Pipeline**（`@openintj/plane-memory`）
  - `vertexShader` / `geometryShader` / `fragmentShader` 三阶段对齐 Python `memory_plane.ShaderPipeline`
  - `ShaderPipeline` 主类 + `ContextEngine` 上下文构建器
  - 钩子事件：`event.SHADER_APPLIED`、`event.CONTEXT_COMPACTED`
- **EmbeddingProvider 抽象**（`@openintj/core`）
  - 统一 `EmbeddingProvider` 接口（同步 / 异步），保留 `SimpleEmbedder` 兜底
  - `MemoryStore` 与 `MemoryRetriever` 改造为可注入 provider
- **嵌入实现**
  - `@openintj/embed-ollama`：通过 Ollama `/api/embeddings` 端点
  - `@openintj/embed-xenova`：本地 `@xenova/transformers`（peer dependency）
- **持久化**
  - `@openintj/storage-lance`：`VectorStore` 接口 + `InMemoryVectorStore` + `LanceDBVectorStore`（peer 依赖 `@lancedb/lancedb`）
  - `@openintj/storage-sqlite`：`MetadataStore` 接口 + 内存兜底 + `SqliteMetadataStore`（peer 依赖 `better-sqlite3`），含 fragments_meta / audit / sessions 三张表与迁移
  - `PersistentMemoryStore`：包装内存层 + LanceDB + SQLite，启动 hydrate、写入 dual-write、`reassignMemoryType` 升级 short→long
- **`MemoryFragment.memoryType`**：显式区分 `short_term | working | long_term`
- **应用形态**
  - `apps/server`：Hono HTTP + SSE 流式 chat、`/api/status`、`/api/memory`、`/api/audit`，请求体由 zod 校验
  - `apps/desktop`：Electron 主进程 IPC（RFC-004 协议）+ preload `contextBridge` + Renderer（React 18 + Vite + Tailwind）三栏布局
- **RFC-003 三方向原型**
  - `@openintj/concurrency`：Mutex / Semaphore / Channel / ConditionVariable / AgentPool / ForkJoin / TokenBucket / BackpressureGate
  - `@openintj/taskpool`：SharedContext / HybridRetriever（vector + BM25 + RRF）/ TaskQueue（DAG 优先级）/ ObjectPool（hot/warm/cold + LRU）
  - `@openintj/dormant`：PassiveStore / PatternMiner（n-gram + 可注入 LLM 抽取，CJK 字符级分词）/ InternalizationManager（用户审批写入 PersonaConfig）
- **集成测试**：`apps/cli/__tests__/rfc3-integration.spec.ts` 覆盖三方向端到端流程
- **文档**：[`docs/architecture/phase2-complete.md`](./docs/architecture/phase2-complete.md) 收尾报告

### Changed

- `MemoryStore` / `MemoryRetriever` 现在以构造时注入的 `EmbeddingProvider` 为准；同步 API 在异步 provider 下会显式抛错
- `ContextEngine` 的预算追踪修正：`conversationTokens` 现在是累加而非覆盖，`CONTEXT_COMPACTED` 钩子触发条件更准确
- TAO/ReAct 与 4 平面默认在 `apps/server` / `apps/desktop` 通过 `assembleAgent`-pattern 统一装配

### Fixed

- `Executor` 重试路径：替换原 Python 端的"伪重试"，落地真正的指数退避 + 状态机合法转换
- `ShaderConfig` 拆出独立的 `recencyHalfLifeHours`，纠正 Python 端把"摘要最大长度"误用为"半衰期小时数"的 bug
- 多处 TypeScript `exactOptionalPropertyTypes: true` 严格模式下的类型问题（AgentPool 泛型 / BackpressureGate 定时器 / persistent-store 属性删除等）
- PatternMiner CJK 分词：从"按空白切词"改为"CJK 字符级 + Latin 词级"混合分词，能正确从中文流水中挖掘 n-gram

### Tooling

- 新增工作区目录：`packages/embed/*`、`packages/concurrency`、`packages/taskpool`、`packages/dormant`
- `pnpm-workspace.yaml` / `tsconfig.json` 引用同步更新
- 验证命令：`pnpm -r typecheck` 与 `pnpm -r --workspace-concurrency=1 test`（Windows 下并行 esbuild 偶发 "service was stopped" 时使用串行模式）

---

## [2.0.0-python-reference] —— Python 实现冻结 (2026-04-29)

- Python v2.0 在仓库根目录冻结为"语义参考实现"
- 不再接收新功能；仅修复严重安全 / 文档 / 行为对齐问题
- 详见 [`docs/architecture/python-reference.md`](./docs/architecture/python-reference.md)
