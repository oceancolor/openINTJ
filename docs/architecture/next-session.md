# 下一次工作交接备忘

> 本文件用于工作中断 / 多日离开后快速恢复上下文。
> 上次更新：2026-08-24（v0.3.1：应用图标改为产品名 openINTJ）。

---

## 🎯 当前工作队列（2026-08-24 更新）

07-19 六项生产接线与 RFC-008 / 桌面工作台已经落地。本轮把交接里仍能写代码的未完成项收了一轮：

1. ~~**技能工具硬隔离**~~：✅ 2026-08-24。ToolHub ALS allowlist + 三端 `runTao` 接线；
   seed 技能改为 `read_file` 等注册名，camelCase 仍兼容。
2. ~~**Desktop 配置热重装**~~：✅ 2026-08-24。运行时配置保存后 assemble 新 agent，失败则保留旧实例。
   全进程重启入口仍保留（Chromium 开关等）。
3. ~~**真实搜索验收入口**~~：✅ 2026-08-24。`RUN_SEARCH_LIVE=1` gated harness 已加；
   仍需本机配置 Tavily/Brave key 后实跑，才能把 T3 事实质量从 fail-closed 推进到真 provider 基线。
4. ~~**Linux CI 打包**~~：✅ 2026-08-24。`release.yml` 增加 ubuntu AppImage。
5. **签名正式发布**：未签名 `v0.3.0` / `v0.3.1` 已从本分支发出。`v0.3.1` 图标为产品名 openINTJ。
   **真正签名仍阻塞于仓库 secrets**。GitHub `origin/main` 与本仓库历史无交集（仅 Python 上传），
   因此 tag 打在 `rfc-005-007-implementation` 上，未能开 PR 合入 main。
6. **后续独立 RFC**（不混入本轮）：动态 LLM 拆图、默认多 Agent、streaming / 更完整 OpenAI-compatible
   provider、embedding migration。`qwen2.5:7b` 仍受本机内存限制。

2026-08-24 未做 / 需人工：

- 把 Win/mac 签名与 Apple 公证 secrets 配进 GitHub 后重打 tag / 再跑 Release
- 处理 `origin/main` 与实现分支无共同历史（目前是无关的 Python 上传）；需要你决定是否把实现分支设为默认分支或重建 main
- `RUN_SEARCH_LIVE=1` 真机跑批并把数字写回 RFC-006 T3
- 蒸馏候选语义去重；Mutex/Channel/CV/Pool 仍实验性

---

### 2026-07-19 生产接线（已完成）

RFC-005/006/007 的核心库、三端 opt-in 接线与确定性测试已经落地。当时补齐的六项：

1. ~~**TaskPool 重启恢复**~~：✅ 2026-07-19 完成。server/desktop 启动扫描
   `listIncompleteRuns()`；默认 cancel 防重复副作用，显式 `OPENINTJ_TASK_POOL_RECOVERY=resume`
   才续跑；快照补 `goalInput`，旧快照拒绝不可靠 resume；包级与两端真实 SQLite E2E 通过。
2. ~~**严格 provider 语义**~~：✅ 2026-07-19 完成。Ollama/Hunyuan adapter 不再包含
   mock 生成器；缺配置、鉴权、网络、HTTP 与非法响应均显式抛结构化错误，状态同步转为
   unauthorized/degraded。显式 mock 只由 ModelRuntime 构造；adapter/runtime 单测、
   Ollama 真机 4/4 与 Desktop mock smoke 5/5 通过。
3. ~~**LLM 取消传播**~~：✅ 2026-07-19 完成。`ChatOptions.signal` 从
   TaskPool → Tao → ReAct（含 micro-loop / `runSingle`）进入 Ollama/Hunyuan provider fetch；
   外部取消保留原始 reason，provider timeout 仍独立报告。core、TaskPool 与两 adapter
   在途请求取消回归测试通过。
4. ~~**ModelRuntime 生命周期与可观测性**~~：✅ 2026-07-19 完成。runtime 提供限频且
   single-flight 的 `refreshHealth()`、脱敏限长的 `ModelRuntimeError` / structured
   `lastError` / attempts；补齐 `model.provider.*` 与 fingerprint hooks、OTel spans/counters，
   server status、Desktop STATUS IPC、CLI status 均接入刷新。
5. ~~**TaskPool 激活契约**~~：✅ 2026-07-19 完成。三端 TaskPool opt-in 自动启用必需的
   classifier；状态明确返回 requested/active/reason/prerequisites 及 persistence/recovery。
   CLI 明示无 SQLite/restart recovery；server/desktop 返回启动 recovery summary；
   Desktop 设置页和 StatusBar 均显示依赖及激活原因。
6. ~~**三端配置 / 状态一致性**~~：✅ 2026-07-19 完成。CLI embedding provider 选项补齐
   simple/xenova，与 server/desktop 共用完整 provider 集；三端统一输出 ModelRuntime structured
   status、顶层 embed/classifier、健康刷新和 TaskPool activation capability。Desktop 运行时配置
   于 2026-08-24 改为保存后热重装；CLI flags 仅作用于当前进程，server env/opts 在启动装配时生效。

2026-07-19 真实环境验收已执行：

- Ollama runtime E2E 4/4；RFC-006 Product Behavior v1.1 后，`qwen2.5:0.5b`
  live trait A/B 连续两次 treatment **9/9**，control **5/9** / **4/9**，
  baseline 已关闭。T3 的 mock/失败/空搜索结果现在会 fail-closed 为无法可靠确认，不再作为
  事实结论证据；真实答案正确性仍待 Tavily/Brave provider 验收。
- `nomic-embed-text` embedding 基准两条路径均 nDCG@4 **1.000**。修复中文无空格分词、
  quick-response 路由并加入“只从 user_input 回答直接回忆题”的 grounded preflight 后，
  `qwen2.5:0.5b` live longrun 连续两轮 user-preferences / tech-decisions 均为
  recall **100%**、pass **100%**；回忆轮为 0 token，且不会被 assistant_output 污染。
- TaskPool 真 Ollama provider 取消 3 轮 + recovery 25 轮 soak 通过；新增 gated harness。
- Windows 未签名 NSIS 安装包已成功产出，并修复 electron-builder 跟随 pnpm workspace
  symlink 越出 appDir 的打包错误。签名 release 仍被品牌图标、证书 secrets、合入 main 与正式 tag 阻塞。

动态 LLM 拆图、默认多 Agent、streaming/OpenAI-compatible provider、embedding migration
属于后续独立 RFC，不混入本轮收口。

---

## 一、当前停在哪里

- **Phase 3.1（真实持久化 e2e）已收官**，仓库标签：`v3.0.0-alpha.1`
- **Phase 3.2（GitHub Actions CI）已收官**，仓库标签：`v3.0.0-alpha.2`
- **Phase 3.3（RFC-003 装配进主 Agent）已收官**，仓库标签：`v3.0.0-alpha.3`
- **Phase 3.4（Dormant 持久化 #9.A）已收官**，仓库标签：`v3.0.0-alpha.4`
- **Phase 3.5（Dormant 审批 UI #9.B）已收官**，仓库标签：`v3.0.0-alpha.5`
- **Phase 3.6（Python v2 ↔ TS 行为对齐测试 #1）已收官**，仓库标签：`v3.0.0-alpha.6`
- **Phase 3.7（Desktop E2E / Playwright + Electron #4）已收官**，仓库标签：`v3.0.0-alpha.7`
- **Phase 3.8（Hooks → OpenTelemetry #7）已收官**，仓库标签：`v3.0.0-alpha.8`
- **Windows 本地启动两批 hotfix 已落**（2026-05-20 → 05-21，无 tag，对应 CHANGELOG `[Unreleased]`）：
  - hotfix #1：Electron 33 vs Node 22 better-sqlite3 ABI 双向自愈 + `.env` 自动加载 + Chromium 后台 SSL 噪音静音
  - 现在 `pnpm desktop:dev` 可以在 Windows 上直接跑：`.env.local` 里写好 HUNYUAN_API_KEY 即可
- **2026-05-20 CI 快照**（历史数据，不代表当前 HEAD；当前状态以实际命令输出为准）：
  - `pnpm lint` exit 0（2 条 React useExhaustiveDependencies 警告，不阻塞）
  - `pnpm exec turbo run typecheck --concurrency=1` → 35/35 successful（新增 `@openintj/telemetry-otel`）
  - `pnpm exec turbo run test --concurrency=1`（CI 模式）→ 35/35 successful，**444 passed + 11 skipped**
  - `OPENINTJ_E2E=1 pnpm exec turbo run test --concurrency=1`（真盘模式）→ 35/35 successful，**455 passed，0 skipped**
  - **`OPENINTJ_PLAYWRIGHT=1 pnpm --filter @openintj/desktop run e2e`（Desktop E2E 模式）→ 7/7 passed（约 35s）**
  - turbo cache 已经把 `OPENINTJ_E2E` 等 env 算进 cache key，env 切换会强制 invalidate 测试任务
- **本轮主要产出（Phase 3.8 / #7 Hooks → OpenTelemetry）**：
  - **新包 `@openintj/telemetry-otel`**（packages/telemetry/otel/）：
    - `attachOtelToHooks(bus, opts)` —— 订阅 hook 事件、per-traceId 维护
      iteration / action / tool span 帧栈、产 6 个 counter；返回 `dispose()`
    - `bootstrapNodeOtel(opts)` —— 可选 SDK 引导（懒 import；缺包才抛错）
    - 10 个新 spec：noop 2 / spans 2 / metrics 3 / dispose 3
  - server / desktop agent 装配端：`enableOtel?: boolean | AttachOtelOpts` + `OPENINTJ_OTEL=1` env + `agent.otel`；`close()` 调 dispose
  - `apps/server/__tests__/otel-wiring.spec.ts`：4 个 wiring 验证（代码 / env / 默认关 / 显式关）
  - `pnpm-workspace.yaml`：加 `packages/telemetry/*`；根 `tsconfig.json` refs 加新包
  - `docs/architecture/phase3-8-otel.md`：阶段记录 + 选型 + 7 类陷阱
- **Phase 3.7 产出**（上一轮）：见 [`phase3-7-desktop-e2e.md`](./phase3-7-desktop-e2e.md)
- **Phase 3.5 产出**（上一轮）：见 [`phase3-5-dormant-approval-ui.md`](./phase3-5-dormant-approval-ui.md)
- **Phase 3.4 产出**：见 [`phase3-4-dormant-persistence.md`](./phase3-4-dormant-persistence.md)
- **工作区状态**：本节不记录易过期的 `git status`；每次接手直接运行 `git status --short`。

## 二、下次开机第一步：自检

```powershell
cd F:\openINTJ\ts
pnpm install                                             # 确认依赖未漂移
pnpm lint                                                # exit 0
pnpm exec turbo run typecheck --concurrency=1            # 以当前 workspace 实际项目数为准
pnpm exec turbo run test --concurrency=1                 # 以当前 HEAD 实际通过/跳过数为准

# 想跑真盘 e2e（需要 @lancedb/lancedb + better-sqlite3 已装）：
$env:OPENINTJ_E2E="1"
pnpm exec turbo run test --concurrency=1                 # 记录本次实际结果，不沿用历史计数
Remove-Item env:OPENINTJ_E2E

# 想跑 Desktop E2E（Playwright + 真 Electron + 真 BrowserWindow，约 35s）：
$env:OPENINTJ_PLAYWRIGHT="1"
pnpm --filter @openintj/desktop run e2e                  # 7/7 passed（含 build）
Remove-Item env:OPENINTJ_PLAYWRIGHT

# 验证 RFC-003 装配 opt-in：
$env:OPENINTJ_DORMANT="1"
$env:OPENINTJ_RETRIEVAL_MODE="hybrid"
$env:OPENINTJ_RATE_LIMIT_QPS="5"
pnpm --filter @openintj/server exec vitest run __tests__/dormant.spec.ts __tests__/hybrid-retrieve.spec.ts __tests__/rate-limited-llm.spec.ts
Remove-Item env:OPENINTJ_DORMANT, env:OPENINTJ_RETRIEVAL_MODE, env:OPENINTJ_RATE_LIMIT_QPS

# RFC-006 真实模型 trait A/B（不进 normal CI；需要本机 Ollama 已启动且模型可用）：
$env:RUN_TRAIT_EVAL="1"
$env:OPENINTJ_LLM_PROVIDER="ollama"
pnpm --filter @openintj/cli test -- trait.harness
Remove-Item env:RUN_TRAIT_EVAL, env:OPENINTJ_LLM_PROVIDER

# 重新生成 Python v2 ↔ TS 行为对齐 fixture（极少需要；Python v2 已冻结）：
cd F:\openINTJ
py scripts/python-parity/generate_fixtures.py            # 重写 4 份 fixture JSON
```

> Windows 下 turbo `--concurrency=1` 是统一策略：避免并行 tsc / esbuild 抢内存导致的 V8 OOM 和 esbuild "service was stopped"。
> RFC-006 live-model 基线已通过：`qwen2.5:0.5b` treatment 连续两次 9/9；
> `qwen2.5:7b` 在本机因 3.1GB CPU repack buffer 分配失败，待更大内存机器复测。
> normal CI 的 scripted 9/9 只证明评测机械与 judges，严禁记作真实模型质量分。
> e2e 测试需要 30s 超时（`describe(..., { timeout: 30_000 }, ...)`），LanceDB 首次建表 + 重新打开比较慢。
> 远端 CI：见 `.github/workflows/ci.yml`；触发分支与 `paths` 过滤以 workflow 当前内容为准。

## 三、Phase 3 路线归档（非当前待办）

来自 [`phase2-complete.md` §九](./phase2-complete.md#九未完成--后续路线)。本表保留阶段决策历史；
当前待办以文首“当前工作队列”为准。

| # | 任务 | 开工成本 | 收益 | 推荐度 | 状态 |
|---|---|---|---|:-:|:-:|
| 1 | ~~Python v2 ↔ TS 行为对齐测试~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-20 完成（Phase 3.6） |
| 2 | ~~真实持久化 e2e~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-09 完成 |
| 3 | ~~**嵌入基准**：simple vs xenova vs ollama 在固定语料上的 nDCG~~ | ~~低~~ | ~~中~~ | ⭐⭐ | ✅ 三方实测归档：simple 纯 cosine 0.396、xenova 0.944、Ollama `nomic-embed-text` 1.000；产品路径分别 0.773 / 1.000 / 1.000。见 `retrieval-benchmark.md` |
| 4 | ~~Desktop E2E（Playwright + Electron）~~ | ~~中~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-20 完成（Phase 3.7） |
| 5 | ~~RFC-003 装配进主 Agent~~ | ~~中~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-11 完成 |
| 6 | **打包发布**：electron-builder Win/macOS + electron-updater | 高 | 中 | ⭐ | 🟢 实现就绪：`electron-builder.yml`（Win nsis/mac dmg/linux AppImage，publish=github oceancolor/openINTJ）+ `updater.ts`（防御式、有测试）+ 主进程接线 + `UpdateBanner` UI + CI `release.yml`（tag v* → 构建 → `--publish always`）。剩运维手动项（图标/签名/首个真 release），见 `release-packaging.md` |
| 7 | ~~可观测性：Hooks → OpenTelemetry~~ | ~~低~~ | ~~低-中~~ | ⭐ | ✅ 2026-05-20 完成（Phase 3.8） |
| 8 | ~~GitHub Actions CI 工作流~~ | ~~低~~ | ~~中~~ | ⭐⭐ | ✅ 2026-05-09 完成 |
| 9.A | ~~Dormant 持久化（SqliteDormantStore + hydrate）~~ | ~~中~~ | ~~高~~ | ⭐⭐⭐ | ✅ 2026-05-19 完成（Phase 3.4） |
| 9.B | ~~Dormant 审批 UI（preload + DormantPanel + tab 布局）~~ | ~~中~~ | ~~中-高~~ | ⭐⭐⭐ | ✅ 2026-05-19 完成（Phase 3.5） |
| 10 | **HybridRetriever LanceDB FTS 路径**：大规模 fragments 时换 LanceDB 原生 FTS，避免每次重建索引 | 中 | 中 | ⭐⭐ | 🟢 存储层原生 FTS（`ensureFtsIndex`/`searchText`）+ `hybridVectorSearch` RRF 融合落地，server `retrieveHybrid` opt-in（`OPENINTJ_LANCE_FTS=1`）；见 §12.2 |
| 11 | ~~**Dormant 事件清理**：`pruneEvents(olderThanTs)` / LRU 防 `dormant_events` 无限增长~~ | ~~低~~ | ~~中~~ | ⭐⭐ | ✅ 2026-06-30 完成（接口/双适配器/runtime 自动清理 + hydrate/record 兜底触发；见 §10.7） |
| 12 | **Parity 扩展**：governance plane / Hooks / ContextEngine 接进 parity 网 | 中 | 中 | ⭐⭐ | 🟢 governance（9）+ context（ContextBudget/shaderForTask，12）+ taxonomy（EventType/CommandType/ErrorCode，14）已接；HookBus 本体无 Python 对手（TS-only），不设跨实现 parity，见 §12.8 |

**阶段收尾记录**：

- ~~#3 嵌入基准~~（✅ 2026-07-19 三方真实数字已回填；Ollama 纯 cosine / 产品路径均 1.000）
- ~~#11 dormant 事件清理~~（✅ 2026-06-30 完成，见 §10.7）
- ~~#12 parity 扩展~~（✅ 2026-07-08 governance+context+taxonomy，见 §12.8）
- ~~#6 打包发布~~（✅ 2026-07-08 实现就绪，剩运维手动项，见 `release-packaging.md`）
- ~~#10 HybridRetriever LanceDB FTS~~（✅ 见 §12.2）
- ~~**Phase 3.8.1 OTel 扩展**：Hono route + Electron IPC 自动 span 接到 agent span 树~~ ✅ 2026-06-30 完成（`withRootSpan`，见 §九）

## 四、上下文复盘清单

按这个顺序读，10 分钟即可回到工作状态：

1. 本文文首“当前工作队列”——当前唯一权威待办入口
2. [`CHANGELOG.md`](../../CHANGELOG.md) —— 当前 `[Unreleased]` 与最近落地记录
3. [`docs/rfcs/`](../rfcs/) —— RFC-001..007 及其边界
4. [`docs/architecture/phase2-complete.md`](./phase2-complete.md) —— 历史阶段快照，不再作为当前路线图
5. [`docs/architecture/python-reference.md`](./python-reference.md) —— Python v2 冻结说明 + 已知遗留清单
6. [`docs/agent-architecture-research_20260422.md`](../agent-architecture-research_20260422.md) —— 最初的架构论证底稿

## 五、当前已知陷阱 / 注意事项

- **CI / 本地统一用 turbo `--concurrency=1`**：跨 OS 一致策略，规避 tsc V8 OOM 与 esbuild 抢占
- **不要**直接编辑 `packages/*/dist/*` —— 改 `src` 后必须 `pnpm exec turbo run build` 才能让消费方测试看到
- **不要**碰 Python 仓库根目录代码（已冻结为 `v2.0-python-reference`，仅作语义参考）
- **LanceDB 关键陷阱**（这一轮踩的坑）：
  - 必须用 `apache-arrow` 显式声明 `FixedSizeList<Float32, N>` schema，**不能**靠 seed-row 推断（旧版 fallback 已保留）
  - SQL `where` 子句中 camelCase 字段必须 **双引号包裹**（`"fragmentId" = '...'`）
  - LanceDB 返回的 `embedding` 是 TypedArray / Arrow Vector，`taskTags` 是 Arrow Vector —— `lancedb.ts` 的 `normalizeEmbedding` / `normalizeStringArray` 会兜底转 `number[]` / `string[]`
  - `apache-arrow` 是 `@openintj/storage-lance` 的 **直接依赖**（不是 peer），因为 `init()` 必用
- **e2e 测试**：默认 skip。需要 `OPENINTJ_E2E=1`，且 `apps/server` / `apps/desktop` / `plane-memory` 工作区都已装 `@lancedb/lancedb` + `better-sqlite3`
- **Electron native binding ABI 不匹配**（2026-05-20 踩到、05-21 二次修复）：
  - vitest 在 Node 进程里跑 → better-sqlite3 ABI 127 ✅；Electron 33 用自家 Node fork（ABI 130）→ `desktop:dev` 真盘启动直接 `dlopen` 失败
  - **第一版 postinstall 修复废弃了** —— 把 binding 切到 Electron ABI 后所有 vitest 都炸；
    现在改成 **双向自愈**：
    - `apps/desktop/scripts/ensure-electron-abi.cjs`（`predev` / `prepackage` 钩子）：
      跑 `pnpm desktop:dev` / `desktop:package` 前自动切到 Electron ABI
    - `ts/vitest.global-setup.ts`：跑 `pnpm test` 前自动把 binding 切回 Node ABI
  - 两个脚本都用 **`spawnSync` 子进程 probe 读 ABI**（关键：本进程绝不能 `require('better-sqlite3')`，否则 Windows 下 .node 句柄被锁住，下一步 prebuild-install 报 EBUSY/EPERM）
  - rebuild 命令也得换 —— `electron-builder install-app-deps` 在 pnpm 布局里**会报 finished 但实际不替换 .node 文件**，改走 `prebuild-install --runtime=electron --target=33.x --force` 才真正落盘
  - 手动触发：`pnpm --filter @openintj/desktop run rebuild-native`（切到 Electron）/ vitest 跑一次（切回 Node）
- **`.env` 加载链路**（2026-05-21 修）：
  - `.env.example` 文档承诺"启动服务时自动加载"，但 cli/server/desktop 三个入口此前都没真 loader → `LLM_PROVIDER=hunyuan` 永远走 mock；
  - 新增 `@openintj/shared/env.ts` 的 `loadOpenintjEnv()`，三处入口都在最前面调用；
  - **目录布局陷阱**：本仓库是 `F:\openINTJ\.env`+ `F:\openINTJ\ts\pnpm-workspace.yaml` 混合双根；
    loader 用 **逐级向上扫描**，到 `.git` 根停。不能只看 `pnpm-workspace.yaml`（会把 `ts/` 错认成根）
  - 优先级（高 → 低）：shell env → 离 startDir 更近的 .env.local → 同级 .env → 上一级 .env.local → ...
  - `summarizeLlmEnv()` 启动时打印 LLM 摘要供 debug；**不打印 key 本体**
- **Electron Chromium 后台 SSL 噪音**（2026-05-20 看到的两条 `ssl_client_socket_impl.cc -107`）：
  - 那是 Safe Browsing / 强制门户检测 / NetworkTimeService 等组件被 GFW 打断，**与 Hunyuan 调用无关**
  - 已在 `apps/desktop/src/main/index.ts` 用 `app.commandLine.appendSwitch("disable-background-networking" / "disable-features=...")` 静音
  - 想恢复后台服务：`$env:OPENINTJ_DESKTOP_KEEP_BG_NET="1"`
- **turbo 缓存**：`OPENINTJ_E2E` / `OPENINTJ_DATA_DIR` / `OPENINTJ_DESKTOP_NO_PERSIST` / `OPENINTJ_LANCE_DEBUG` 已纳入 cache key（`turbo.json`），切换 env 会强制 invalidate test 任务
- **biome 已放宽**：`useLiteralKeys` / `noNonNullAssertion` / `noUnusedTemplateLiteral` 等 13 条与历史代码冲突的规则已关；不要再因为 lint 报错就批量改业务代码
- `@xenova/transformers` 仍是 peer dependency，按需 `pnpm add`

## 六、累计产出文件清单（Phase 3.1 → 3.6）

### Phase 3.1（持久化 e2e）

新增：

- `ts/packages/planes/memory/src/persistence-factory.ts`
- `ts/packages/planes/memory/__tests__/persistence-factory.spec.ts`
- `ts/apps/server/__tests__/persistence-e2e.spec.ts`
- `ts/apps/desktop/__tests__/agent-persistence.spec.ts`
- `docs/architecture/phase3-1-persistence.md`

改动：

- `ts/packages/storage/lance/src/lancedb.ts`：显式 schema + 双引号 SQL + 类型规范化
- `ts/packages/storage/lance/package.json`：`apache-arrow` 移到 `dependencies`
- `ts/packages/planes/memory/src/index.ts`：导出工厂
- `ts/apps/server/src/agent.ts`：用工厂 + dataDir 选项 + close
- `ts/apps/desktop/src/main/agent.ts`：用工厂 + dataDir 选项 + close
- `ts/apps/desktop/src/main/index.ts`：`app.getPath('userData')` + before-quit
- 各 `package.json`：装 peer deps（`@lancedb/lancedb` / `better-sqlite3`）

### Phase 3.2（GitHub Actions CI）

新增：

- `.github/workflows/ci.yml`（仓库根；旧的 `ts/.github/workflows/ci.yml` 已删除）

改动：

- `ts/turbo.json`：`test` 任务 cache key 加入 `OPENINTJ_*` env
- `ts/biome.json`：放宽与历史代码冲突的 13 条规则
- 全仓 107 个 `tsconfig.json` / `package.json` 被 biome formatter 自动收紧
- `docs/architecture/phase2-complete.md`：§九 #1 划掉
- `CHANGELOG.md`：新增 `3.0.0-alpha.1`、`3.0.0-alpha.2` 条目

### Phase 3.3（RFC-003 装配进主 Agent）

新增：

- `ts/packages/dormant/src/dormant-runtime.ts` + `__tests__/dormant-runtime.spec.ts`
- `ts/packages/concurrency/src/rate-limited-llm.ts`
- `ts/apps/server/src/hybrid-retrieve.ts`
- `ts/apps/server/__tests__/dormant.spec.ts` / `hybrid-retrieve.spec.ts` / `rate-limited-llm.spec.ts`
- `docs/architecture/phase3-3-rfc3-wiring.md`

改动：

- `ts/packages/dormant/src/index.ts`：导出 `DormantRuntime`
- `ts/packages/concurrency/src/index.ts`：导出 `RateLimitedLlmClient`
- `ts/apps/server/{src/agent.ts, src/routes.ts, src/rate-limited-llm.ts, package.json, tsconfig.json}`：装配三方向 opt-in + 路由 + 兼容 re-export
- `ts/apps/desktop/{src/main/agent.ts, src/main/ipc-handlers.ts, src/shared/ipc-protocol.ts, package.json, tsconfig.json}`：镜像 server 端装配 + 5 个 Dormant IPC channel + MEMORY_QUERY mode
- `ts/apps/desktop/__tests__/ipc-handlers.spec.ts`：扩展 5 个新测试
- `CHANGELOG.md`：新增 `3.0.0-alpha.3` 条目

### Phase 3.4（Dormant 持久化 / #9.A）

新增：

- `ts/packages/dormant/src/persistence.ts`（`DormantPersistenceAdapter` + `InMemoryDormantStore` + `DormantSnapshot`）
- `ts/packages/storage/sqlite/src/dormant.ts`（`SqliteDormantStore` + `createSqliteDormantStore`）
- `ts/packages/dormant/__tests__/persistence.spec.ts`（9 个）
- `ts/packages/storage/sqlite/__tests__/dormant.spec.ts`（11 个）
- `ts/apps/server/__tests__/dormant-persistence-e2e.spec.ts`（6 个，CI 2 PASS + 4 skip / E2E 6 PASS）
- `docs/architecture/phase3-4-dormant-persistence.md`

改动：

- `ts/packages/dormant/src/{dormant-runtime.ts, passive-store.ts, internalization-manager.ts, index.ts}`：adapter 槽 + hydrate + restoreState + recordBulk
- `ts/packages/storage/sqlite/src/index.ts`：修复重复 export bug
- `ts/apps/server/src/agent.ts` + `ts/apps/desktop/src/main/agent.ts`：新增 `dormantPersistence` / `dormantDbPath` opts + auto-wire + hydrate + `dormantPersistenceInfo`
- `CHANGELOG.md`：新增 `3.0.0-alpha.4` 条目

### Phase 3.5（Dormant 审批 UI / #9.B）

新增：

- `ts/apps/desktop/src/renderer/components/DormantPanel.tsx`（mine + filter + 卡片 + persona 折叠）
- `docs/architecture/phase3-5-dormant-approval-ui.md`

改动：

- `ts/apps/desktop/src/shared/ipc-protocol.ts`：StatusResponseSchema 补三字段 + 6 个 Dormant DTO/Response/Error schema
- `ts/apps/desktop/src/preload/index.ts`：暴露 5 个 dormant API（联合类型 success | error）
- `ts/apps/desktop/src/renderer/App.tsx`：右侧栏改 tab 布局 + Dormant pending 角标
- `ts/apps/desktop/src/renderer/components/StatusBar.tsx`：补 retrievalMode/persistence/dormant 三段；StatusSnapshot 切到 protocol re-export
- `ts/apps/desktop/src/renderer/components/TrajectoryPanel.tsx`：去外层 chrome（tab 容器统一提供）
- `ts/apps/desktop/__tests__/ipc-handlers.spec.ts`：扩展 6 个新契约测试
- `CHANGELOG.md`：新增 `3.0.0-alpha.5` 条目

### Phase 3.6（Python v2 ↔ TS 行为对齐测试 / #1）

新增：

- `scripts/python-parity/generate_fixtures.py`（Python 端只读取证脚本，覆盖 4 个 slice）
- `scripts/python-parity/README.md`（工具说明 + 已知偏差速查）
- `ts/packages/core/__tests__/parity/python-v2.spec.ts`（23 tests：SimpleEmbedder / cosine / decay）
- `ts/packages/core/__tests__/parity/fixtures/python-v2.json`
- `ts/packages/planes/control/__tests__/parity/python-v2.spec.ts`（21 tests：GoalParser / Planner）
- `ts/packages/planes/control/__tests__/parity/fixtures/python-v2.json`
- `ts/packages/planes/execution/__tests__/parity/python-v2.spec.ts`（17 tests：StateMachine / Executor）
- `ts/packages/planes/execution/__tests__/parity/fixtures/python-v2.json`
- `ts/packages/planes/memory/__tests__/parity/python-v2.spec.ts`（3 tests：Store overflow / Retriever scoring）
- `ts/packages/planes/memory/__tests__/parity/fixtures/python-v2.json`
- `docs/architecture/phase3-6-parity-tests.md`（阶段记录 + 已知偏差矩阵 + 容差策略）

改动：

- `ts/biome.json`：`files.ignore` 加 `**/__tests__/parity/fixtures/**`
- `CHANGELOG.md`：新增 `3.0.0-alpha.6` 条目

### Phase 3.8（Hooks → OpenTelemetry / #7）

新增：

- `ts/packages/telemetry/otel/`（新包 `@openintj/telemetry-otel`）：
  - `package.json`：deps=`@openintj/core` + `@opentelemetry/api`；6 个 SDK 包全标 `peerDependenciesMeta.optional`
  - `src/attach.ts`（~290 行）：`attachOtelToHooks(bus, opts)`，per-traceId span 帧栈 + 6 counter + dispose
  - `src/bootstrap.ts`（~100 行）：`bootstrapNodeOtel(opts)`，懒 import SDK + ProxyTracerProvider 探针 idempotent
  - `src/index.ts`：barrel
  - `__tests__/noop.spec.ts`（2 tests）：未注册 provider 不抛、不产 span
  - `__tests__/spans.spec.ts`（2 tests）：InMemorySpanExporter 断言 parent/child + ERROR 状态 + recordException
  - `__tests__/metrics.spec.ts`（3 tests）：InMemoryMetricExporter 断言 6 counter 累计
  - `__tests__/dispose.spec.ts`（3 tests）：dispose 兜底 end + unregister + 新 iteration 把旧 iter 标 unfinished
- `ts/apps/server/__tests__/otel-wiring.spec.ts`（4 tests）：enableOtel 三通道
- `docs/architecture/phase3-8-otel.md`：阶段记录 + 选型 + 7 类陷阱

改动：

- `ts/pnpm-workspace.yaml`：加 `packages/telemetry/*`
- `ts/tsconfig.json`：refs 加 `packages/telemetry/otel`
- `ts/apps/server/{src/agent.ts, package.json, tsconfig.json}`：
  - `enableOtel` opt + `resolveOtel(opts)` + `agent.otel` + close 调 dispose
  - devDep 加 `@opentelemetry/{api,sdk-trace-base}`（仅 wiring 测试用，运行时不需要）
- `ts/apps/desktop/{src/main/agent.ts, package.json, tsconfig.json}`：镜像 server
- `CHANGELOG.md`：新增 `3.0.0-alpha.8` 条目

### Phase 3.7（Desktop E2E / Playwright + Electron / #4）

新增：

- `ts/apps/desktop/e2e/playwright.config.ts`（workers=1，`OPENINTJ_PLAYWRIGHT=1` 才执行）
- `ts/apps/desktop/e2e/fixtures.ts`（`electronApp` + `page` fixture，默认 mock + no-persist）
- `ts/apps/desktop/e2e/tsconfig.json`（独立 e2e 项目，不污染 src）
- `ts/apps/desktop/e2e/tests/smoke.spec.ts`（5 tests：boot / status / chat / trajectory / dormant tab）
- `ts/apps/desktop/e2e/tests/dormant.spec.ts`（2 tests：mine 按钮 + 扫描摘要，需 `OPENINTJ_DORMANT=1`）
- `docs/architecture/phase3-7-desktop-e2e.md`（阶段记录 + 两个坑 + CI 集成）

改动：

- `ts/apps/desktop/src/main/index.ts`：preload 路径 `../preload/index.js` → `../preload/index.mjs`（修历史 silent fail）
- `ts/apps/desktop/package.json`：加 devDep `@playwright/test ^1.60`；
  `typecheck` 串第二段 `tsc --noEmit -p e2e/tsconfig.json`；
  新 script `e2e`（build + run）/ `e2e:run`（只 run）
- `ts/biome.json`：`files.ignore` 加 `**/test-results/**` 与 `**/playwright-report/**`
- `.github/workflows/ci.yml`：新增 `e2e-desktop` job（Ubuntu 24.04 + xvfb，需要 libnss3/libgtk-3-0 等运行时）
- `CHANGELOG.md`：新增 `3.0.0-alpha.7` 条目

---

## 七、Phase 3.3 / 3.4 / 3.5 / 3.6 / 3.7 / 3.8 关键陷阱（接续时回看）

1. **`DormantRuntime` 默认 `category: "other"` 时 proposals 为空**
   - PatternMiner 不配 `llmExtract` 会把每个 ngram 打成 "other"
   - `InternalizationManager.defaultMapToField` 忽略 "other"
   - **生产部署务必配 `dormantOpts.minerOpts.llmExtract`** 或自定义 `internalizationOpts.mapToField`
2. **HybridRetriever 每次查询重建索引** —— 中等规模够用，N>10k 切 LanceDB FTS
3. **rate-limit 装饰只覆盖 `chat / visionChat`** —— 未来加 stream/embeddings 接口需要同步扩展
4. **PassiveStore / PersonaConfig 不持久化** —— ✅ Phase 3.4 已解决（auto 模式：`dataDir + enableDormant=true` 自动挂 `SqliteDormantStore`）
5. **IPC 协议向后兼容**：新增字段都是 optional，旧 renderer 仍然能用；新 renderer 调旧 main 会拿到 `dormant_not_enabled` 而不是崩
6. **Phase 3.4 装配点**：装配顺序很重要 —— `await createSqliteDormantStore` → 传入 `DormantRuntime` 的 `adapter` 槽 → `await runtime.hydrate()`。close 时**先 dormant.close 再 persistentStore.close**
7. **Phase 3.4 `SqliteDormantConfigInput`**：`wal` 用 `z.boolean().default(true)`，input/output 类型不一致 —— 用 `z.input<>` 给装配点，`z.infer<>` 给内部
8. ~~**Phase 3.4 `dormant_events` 表无限增长**~~：✅ 2026-06-30（#11）完成。PassiveStore 内存有 `maxPassiveEvents` 环形上限；磁盘表由 `eventRetentionMs` / `maxDiskEvents` 保留策略 + 三处触发（`mine()` 末尾、`hydrate()` 启动、`record()` 每 `autoPruneEveryNEvents`≈256 条兜底）自动收敛。server/desktop 默认 `maxDiskEvents: 50_000`
9. ~~**桌面端审批 UI 仍未接**~~：✅ Phase 3.5 完成
10. **Phase 3.5 协议联合类型**：preload 5 个 dormant API 全部返回 `Success | Error` 联合类型 —— renderer 必须用 `'error' in r` narrow 才能拿数据。这是为了把"dormant 未启用"这类正常态从 try/catch 里剥离出来
11. **Phase 3.5 类型对齐**：`StatusBar.tsx` 不再本地定义 `StatusSnapshot`，而是 `type StatusSnapshot = StatusResponse`（来自 protocol）。新加字段时只改 ipc-protocol.ts 即可全栈传播
12. **Phase 3.5 renderer 0 测试**：desktop 工作区只有 main-process vitest，没有 jsdom/@testing-library/react。`DormantPanel` 的逻辑分支已被 IPC 契约测试覆盖；UI 渲染留给手动 / Playwright e2e（#4）
13. **Phase 3.6 parity fixture 是 commit-in 资产**：`packages/*/__tests__/parity/fixtures/python-v2.json` 由 `scripts/python-parity/generate_fixtures.py` 一次性生成。CI 不跑 Python；只有 Python v2 端被"延寿活动"修补、或 `generate_fixtures.py` 自身改动时才需要重跑。biome 已 ignore 该目录，**不要**对 fixture 跑格式化。
14. **Phase 3.6 已知偏差矩阵**：详见 `phase3-6-parity-tests.md` §三。不要随手"修齐" Planner delete/execute 模板回 general、或把 TS `Math.LN2` 改回 `0.693` —— 这些偏差**有意保留**（TS 修复或扩展 Python）。
15. **Phase 3.6 决意保留 Python 0.693 近似**：`decayImportance` parity 容差用 `1e-4` 而非 `1e-12`。换 embedder（如 xenova / nomic-embed）也别动这条；它只影响 `decay` 一项的纯数学精度。
16. **Phase 3.6 fixture `schemaVersion=1`**：TS spec 加载时会断言版本；将来如果想加新字段（如 `governance` slice），把 `schemaVersion` 升 2 强制旧 fixture 失效，避免误判。
17. **Phase 3.7 preload `.mjs` 才是正确路径**：electron-vite 默认产物是 `out/preload/index.mjs`。**不要**回退到 `.js`，否则 `window.openintj` 永远 undefined，但 vitest 走 mock electron 路径不会暴露 —— 只有真 Electron 启动才崩。Electron 28+ 原生支持 ESM preload。
18. **Phase 3.7 Windows + Playwright `_electron.launch` 别加 `--no-sandbox`**：该 flag 在 Windows + Electron 33 + Playwright 1.60 这个具体组合下会让 launch 卡死 30s。Linux + xvfb 不需要这个 flag。fixture 已经只传 `[MAIN_ENTRY]`；扩 e2e 用例时也别图省事重新加 flag。
19. **Phase 3.7 e2e 默认 opt-in**：`OPENINTJ_PLAYWRIGHT=1` 才会跑；不设 env 时 playwright.config.ts 顶部直接 `testIgnore: ["**/*"]`。pnpm test / turbo test 不会触发它。CI 走专用 `e2e-desktop` job（已加 `OPENINTJ_PLAYWRIGHT: "1"` env）。
20. **Phase 3.7 strict-mode locator**：`getByText(/mock 模式/)` 会撞 chat 气泡 + trajectory JSON dump。新加 e2e 断言前先用 tailwind 颜色 token 圈父定位（`div.bg-\\[\\#1e1e2e\\]` = 主聊天区，`div.bg-\\[\\#313244\\]` = assistant 气泡）。
21. **Phase 3.8 HookBus traceId ≠ OTel traceId**：前者是 UUID 字符串、后者是 128-bit hex 由 SDK 生成。本适配器把 HookBus traceId 写到 `trace_id` span 属性方便反查。不要让 caller 拿 `agent.otel` 当作 trace context 源。
22. **Phase 3.8 tool 事件必须带 traceId 才能挂对 parent**：`tool.beforeCall` / `tool.afterCall` / `tool.onError` emit 时必须传 `{ traceId }`（ToolHub 真实代码已这么做）。漏传的话 tool span 会挂在 'anon' trace 上，与 iteration / action 失联。写 hook 单测时尤其要注意。
23. **Phase 3.8 OTel SDK 是 optional peer**：`attachOtelToHooks` 只需 `@opentelemetry/api`（已是硬依赖）；`bootstrapNodeOtel` 懒 import 6 个 SDK 包，缺包就 throw。生产部署用 OTLP 时 consumer 自己 `pnpm add @opentelemetry/{sdk-trace-node,exporter-trace-otlp-http,resources,semantic-conventions}`。
24. **Phase 3.8 metric 默认 DELTA**：`InMemoryMetricExporter` 的 `AggregationTemporality.DELTA = 0`；构造时显式传 0，否则跨多次 emit 会丢中间增量。生产 exporter 一般是 CUMULATIVE，行为不同。
25. **Phase 3.8 tool.onError 不 end span**：故意设计，让 `tool.afterCall` 统一收尾（happy-path 一致）。如果业务异常分支不发 afterCall，`dispose()` 会兜底 end 并打 `disposed=true` 标记。

---

**回来工作时**：直接对我说 "继续 Phase 3 的 #X" 或 "先自检一遍" 都可以，我会顺着这份备忘接下去。

---

## 八、RFC 设计 vs 实现全面盘点（2026-05-30）

> 起因：用户反馈"重启后模型记不住上下文"。排查发现 `MemoryPlane` 只写不读——记忆持久化了但从未注入对话。
> 已修：`TaoLoop` 新增 `contextProvider`（每轮异步构造 system prompt），desktop/server/cli 三端接 `ContextEngine.build` 注入 `[记忆参考]`；`run()` 改为先跑再记录避免自命中。
> 借此对 RFC-001..004 做了一次完整的"设计 vs 实现"比对，结论与五项推进计划如下。

### 8.1 完成度矩阵

| RFC / 模块 | 设计意图 | 状态 | 缺口摘要 |
|---|---|:-:|---|
| RFC-001 TAO/ReAct | 双层循环 + 4 早停 + 多轮 | 🟢 主路径完成 | `enableReact:false` 与 ADR-001 已闭合；`tao-step-bench` 已守护框架开销。RFC-001 §11 Q2 的产品级“续轮”判据仍需单独决策 |
| RFC-002 Hooks | 强类型/优先级/短路/改写 | 🟢 完成 | `hook-bus-bench` 已守护 no-handler / 10-handler / register 的灾难性性能回退 |
| RFC-003 方向一 多线程 | Mutex/Channel/CV/Pool/ForkJoin 装配进 Agent | 🟢 有界接入 | `forkJoin` + `Semaphore`（经 `forkJoin.concurrency`）已进三端自一致性主路径（§9.3 并行 + §12.6 并发上限）；Mutex/Channel/CV/Pool/Backpressure 仍实验性 |
| RFC-003 / RFC-007 任务池 | 模板 DAG、有界并发、可靠状态机、持久化 | 🟢 主路径收口 | TaskPool 已三端 opt-in；SQLite 启动恢复、LLM HTTP 取消传播及真 Ollama cancel/recovery soak 均通过。默认 resume 仍保持显式 opt-in |
| RFC-003 方向三 钝化记忆 | 存→学→审批→**注入 systemPrompt** | 🟢 回路已闭合 | 2026-07-08 收官：`getPersona()` 出口 + 三端注入（含 CLI）+ A/B 杠杆（`OPENINTJ_PERSONA`）+ 脱敏（默认开）+ revoke（runtime/IPC/HTTP/UI 全通）。见 §12.4 |
| RFC-004 桌面 IPC | 安全模型 + 流式 + 系统能力 | 🟢 基本完成 | 流式✅ 自动更新✅ 工作区读写(治理门禁)✅ config 服务✅ fs.watch✅；2026-07-08 补齐**设置面板 UI**（消费 workspace/config IPC + 实时变更）+ config 字段/启动接线 + **utility 挖掘 worker**（opt-in）。见 §12.5 |
| 跨 RFC 执行工具 | 治理边界下真实 fs/命令 | 🟢 完成 | fs/命令工具早已是真实沙箱实现（`createWorkspaceTools`）；2026-07-08 又把治理接进 `ToolHub.call`（gate → `checkToolCall`）+ 桌面 IPC，`[mock]` 仅剩 search 兜底。见 §12.3 |
| #3 嵌入基准 | simple/xenova/ollama nDCG | 🟢 完成 | 三方双路径实测已归档；Ollama `nomic-embed-text` 两条路径 nDCG@4 均 1.000。见 `retrieval-benchmark.md` |

### 8.2 重点缺口（按严重度）

1. ~~**🔴 执行平面工具是 mock**~~。✅ **已解决（此条盘点已过时）**：fs/命令工具其实早已是真实沙箱实现（`createWorkspaceTools`：路径限定 workspace 根 + 读写大小上限 + 命令白名单/默认禁用），三端 agent 均已接；`[mock]` 只剩 search 兜底。2026-07-08 进一步把**治理接进工具执行**（`ToolHub` gate → `GovernancePlane.checkToolCall`：策略黑名单 + 每分钟工具配额 + 审计）+ 桌面 workspace IPC 同闸门，补齐 RFC-004 §8 的「Governance → fs」边界。见 §12.3。
2. ~~**🟠 钝化记忆 persona 未注入**~~ ✅ **已解决（2026-07-08，见 §12.4）**：`DormantRuntime.getPersona()` 出口就绪；server/desktop/**cli** 三端 `contextProvider` 均在 `[记忆参考]` 之前注入 `[用户画像]`（无需检索即生效，满足 §3.6 #2）；A/B 由 `resolvePersonaInjection`（`enablePersona` / `OPENINTJ_PERSONA=0`）控制（§3.6 #3）；脱敏 `record()` 前默认生效；revoke 打通 runtime + 桌面 IPC + `POST /api/dormant/proposals/:id/revoke` + DormantPanel「撤销」按钮（§3.6 #4）。
3. ~~**🟡 方向一/二并发原语只是原型**~~ ✅ **已缩小（2026-07-08，见 §12.6）**：`forkJoin` 早已进三端自一致性主路径（§9.3），本次把 `Semaphore` 经 `forkJoin.concurrency` 接上——`selfConsistency.maxConcurrency` / `OPENINTJ_SELF_CONSISTENCY_CONCURRENCY` 给多采样设并发上限（有界并发，避免打满 LLM 配额）。真实 `agent.run()` 现已接：`RateLimitedLlmClient`(rateLimit) + `HybridRetriever`(hybrid) + `forkJoin`/`Semaphore`(self-consistency)。剩 Mutex/Channel/CV/Pool/Backpressure 仍实验性（无主路径消费者）。
4. ~~**🟡 RFC-004 系统能力面缺失**~~ ✅ **已解决（2026-07-08，见 §12.5）**：workspace 读写/info/pickDir IPC + `fs.watch → EVT_WORKSPACE` + `ConfigService`(getConfig/updateConfig) 后端此前已在（治理门禁 + 契约测试）；本次补**renderer 消费面**（`SettingsPanel`：workspaceInfo/pickDir + config 增删改 + 实时变更流）、config schema 补 `enablePersona/enableSkills/enableSkillLearning/enableClassifier` 并接入启动装配、`mine()` 的 **utility worker 下放**（`OPENINTJ_DORMANT_WORKER=1`，失败回退内联）。注：§7 流式已等价实现（hook→IPC 实时推送）；配置热重载仍需重启（面板显式标注）。
5. ~~**🟢 小缺口**：`enableReact:false` 声明未实现；RFC-001 §11 Q1 未文档化~~ ✅ **均已闭合（见 §12.7）**：退化分支早已实现（`runSingle`）且经 `decideRoute`→`route.single` 在三端可达（core `tao.spec` + 新增 classifier `routing.spec` 守护）；function-calling vs 文本协议决策由 **ADR-001**（2026-06-30）记录（含代价与回退触发条件），RFC-001 §11 Q1 已关闭。

### 8.3 验证 & 可观测盘点

强：单测 444+ passed、Python parity、真盘持久化 e2e(`OPENINTJ_E2E`)、Desktop Playwright e2e(`OPENINTJ_PLAYWRIGHT`)、OTel trace/metric、desktop UI(StatusBar/Trajectory/Dormant/记忆Tab)、server `/api/{status,audit,memory,dormant/*}`。

仍弱 / 待补：
- 性能基准已经存在，但只守灾难性回退，不等于生产负载容量测试。
- 记忆召回已有 simple/xenova/Ollama 固定小语料指标；真实业务语料与大规模容量结果仍缺。
- live-model trait A/B 已连续两次 9/9；直接事实/约束 longrun recall 也连续两轮 100%。
  该结果只覆盖脚本化直接回忆，不代表开放式长期记忆质量；T3 已拒绝把 mock search 当成事实证据，
  但仍需真实搜索 provider、更强模型和更广语料建立事实质量与长期质量置信区间。
- fs/命令工具已是真实沙箱；联网 search 未配置 provider 时仍会走 mock 兜底，但 Product
  Behavior 会把该结果收口为“无法可靠确认”，不会把该路径当真实任务完成证据。
- OTel hooks/span/metric 与 ModelRuntime provider/fingerprint 事件已落地但默认 opt-in，尚无现成 dashboard/SLO。

### 8.4 五项推进计划（2026-05-30 全部落地）

| # | 任务 | 价值 | 状态 |
|---|---|---|:-:|
| 1 | **接真实工具 + 治理边界 fs** | 解锁产品核心价值 | ✅ |
| 2 | **闭合 persona 注入** | 补 RFC-003 §3.6 最后一公里 | ✅ |
| 3 | **检索 nDCG/recall 评测基准** | 让"记忆有效"从体感变数据 | ✅ |
| 4 | **性能基准**（hook-bus / TAO 单步） | 守护 RFC 性能承诺 | ✅ |
| 5 | **方向一/二：文档标注实验性** | 消除"库存能力"误读 | ✅ |

**各项落地细节：**

1. **真实工具**：新增 `@openintj/plane-execution` 的 `createWorkspaceTools`——`read_file`/`write_file`
   被沙箱限定在 workspace 根内（`resolveInRoot` 拒绝 `..`/绝对路径越界），`execute_command`
   **默认禁用**，需 `enableCommands` + 命令白名单（env `OPENINTJ_ENABLE_COMMANDS` /
   `OPENINTJ_ALLOWED_COMMANDS`）。cli/server/desktop 三端 agent 用它替换原 `noop`；
   desktop 默认工作区 = `Documents/OpenINTJ`。共享解析器 `resolveWorkspaceConfig`（@openintj/shared）。
   测试：`plane-execution/__tests__/workspace-tools.spec.ts`（12）+ cli agent 往返/越界/命令禁用断言。
2. **persona 注入**：`InternalizationManager.personaSystemPrompt()` 把已批准 PersonaConfig 渲染成
   `[用户画像]` 片段，`DormantRuntime.personaSystemPrompt()` 暴露；desktop/server 的 `contextProvider`
   把它拼到 baseSystemPrompt 前 → 内化偏好无需检索即生效。测试：cli rfc3-integration 补 2 断言。
3. **检索评测**：新增 `@openintj/plane-memory` 的 `src/eval/retrieval-metrics.ts`
   （nDCG/recall/precision/MRR + `evaluateRanker`），`retrieval-benchmark.spec.ts` 在固定主题语料上
   守护默认检索路径基线（当前 simple@dim64：nDCG@4≈0.77 / recall≈0.71 / MRR=1.0）。
   换 xenova/ollama embedder 复用同一 harness 对比即可。
4. **性能基准**：`core/__tests__/perf/` 下 `hook-bus-bench`（no-handler emit ≈0.77µs/op、10-handler
   ≈17µs/op、register ≈39µs/op）+ `tao-step-bench`（单轮 TAO 框架开销 ≈0.03ms/run）。
   宽松阈值守护灾难性回退，实际数字打印到 CI 日志。
5. **方向一/二定性**：`@openintj/concurrency` 与 `@openintj/taskpool` 的 index 顶部 + 新增 README
   明确标注「已接入产品：RateLimitedLlmClient / HybridRetriever；其余为实验性原语，未接入
   agent.run() 主路径」，并给出未来集成路线。package.json description 同步更新。

**追加（2026-05-30）：并发/多任务/多 Agent 可观测性**
- `HookEventMap` 新增 `pool.*` / `forkjoin.*` / `task.*` 事件（category=`concurrency`）。
- `AgentPool` / `forkJoin` / `TaskQueue` 支持注入 `HookBus`（可选，不传零开销），发出生命周期事件；
  TaskQueue 的 emit 在 mutex 临界区外，避免再入死锁。
- `attachOtelToHooks` 把它们翻译成独立 span（`openintj.pool.job` / `openintj.forkjoin` /
  `openintj.task.run`）+ counter（`openintj.pool.jobs` / `openintj.forkjoin.branches|rejected` /
  `openintj.task.enqueued|completed`），dispose 兜底结束未完成 span。
- 测试：concurrency observability（含成功/失败/向后兼容）、taskpool observability（DAG 依赖 + fail）、
  otel concurrency（span + metric + dispose 兜底）。这样即便方向一/二仍是实验库，
  其并发行为也已**可观测**——为日后接入主路径打底。

> 上述「仍未做」清单已在 2026-06-30 批量收尾，详见 [§九](#九已定位任务批量收尾2026-06-30)。

---

## 九、已定位任务批量收尾（2026-06-30）

> 起因：用户要求按 `3-1-5-2-4-6-7` 顺序推进 §八 末尾「仍未做」里那批**已定位、可直接做**的子项。
> 全部落地，未提交（无 tag，归入 CHANGELOG `[Unreleased]`）。下面按完成顺序记录。

### 9.1（#3）RFC-001 收尾：`enableReact:false` 退化分支 + ADR-001

- `ReactStateMachine.runSingle()`：单次 LLM 调用，不跑微循环、不下发工具描述、不解析 action；
  仍发 `react.beforeThought` / `react.afterThought` / `react.onStopCondition` 钩子保观测。
- `TaoLoop.run()`：`enableReact===false` 时走 `runSingle`，否则原 `run`。
- **ADR-001**（`docs/architecture/adr-001-react-tool-protocol.md`）：正式记录「ReAct 用文本协议
  （Thought/Action/FINAL）而非 OpenAI function-calling」的决策、理由（provider 中立 / 可观测 / Python v2 parity / 简单）、
  取舍与重评触发条件；RFC-001 §11 Q1 改为「已由 ADR-001 解决」。
- 测试：`core/__tests__/tao.spec.ts` 加 `enableReact:false` 委派断言（不调 toolRunner、原样返回 LLM 输出）。

### 9.2（#1）RFC-004 工作区 / 配置 IPC

- `ipc-protocol.ts`：新增 Workspace（read/write/info/pickDir + 变更事件）与 AppConfig（get/update）schema 与 channel；
  Dormant 状态枚举补 `revoked` + `DORMANT_REVOKE` channel。
- `config-store.ts`（新）：`ConfigService`（get/update + Zod 校验，JSON 落 userData，小体量用同步 IO）。
- `DesktopAgent` 暴露 `workspace.{config,tools}`；ipc-handlers 实现 WORKSPACE_*（委派 `workspace.tools`）/ CONFIG_*
  （委派 ConfigService，依赖注入便于测试）/ DORMANT_REVOKE；`fs.watch(root)` 推 `EVT_WORKSPACE`。
- `main/index.ts`：装配读 ConfigService 偏好（env 优先），Electron `dialog` 实现 `pickDirectory`。
- preload 暴露 workspaceInfo/Read/Write/PickDir/onWorkspaceEvent/getConfig/updateConfig/dormantRevoke。
- 测试：`ipc-handlers.spec.ts` 扩到 27（含越界 `..` 拒绝、配置落盘、approve→revoke 周期）。
- ⚠️ utility process 蒸馏 worker 仍跑在 main（复杂度 + 测试成本，留后续）。

### 9.3（#5）方向一/二有界接入产品路径：self-consistency

- `@openintj/shared` 新增 `self-consistency.ts`：`selectConsistentAnswer`（majority / longest / first + 平票兜底）
  + `resolveSelfConsistency`（opts/env 解析，samples 上限 8）。
- cli/server/desktop 三端 `run()`：开启时用 `forkJoin` 并行跑多份 `tao.run`，再选一份——
  **复用方向一的 `forkJoin` 观测**（`forkjoin.*` span/counter）把实验原语接成真实产品路径。
- 测试：shared self-consistency 单测 + cli agent 集成（3 samples → `forkjoin.afterJoin` 3 fulfilled）。

### 9.4（#2）钝化记忆：revoke + 脱敏 + abTest 脚手架

- **revoke**：`InternalizationProposal.status` 加 `revoked`；`InternalizationManager.revoke()`（`deleteNested`
  删 persona 字段 + 升 version）；`DormantRuntime.revoke()` 持久化提案与 persona 快照。
- **脱敏**：`redaction.ts`（`createRedactor` + `defaultRedactor`，规则覆盖 email/API key/信用卡/手机号/身份证；
  顺序上 idCard 先于 creditCard/phone 防贪婪误匹配）。`DormantRuntime.record()` 入库前脱敏。
- **abTest**：`ab-test.ts`（`runAbTest` 纯编排：多 variant × queries 打分聚合选 winner），为「越用越好」验证打底。
- 测试：redaction / ab-test 单测 + dormant-runtime 的 record 脱敏 / revoke 周期。

### 9.5（#4）检索：可插拔 embedder 三方对比 + 增量索引 + 任务完成度评测

- `retrieval-benchmark.ts`（新）：`benchmarkRetrieval(embedder)` 通用异步 harness（探维度 → 建库 → `evaluateRanker`）；
  spec 默认跑 `SimpleEmbedder` 守基线，`RUN_EMBED_COMPARE=1` 时三方对比 simple/xenova/ollama。
- `HybridRetriever` 增量索引：`upsert/upsertBatch/remove/clear`（增量维护 BM25 统计），替代每次全量重建；
  change-feed 已于 2026-06-30 接进 agent（见 §十 A1）。
- `@openintj/shared` 新增 `task-eval.ts`：端到端任务完成度 harness（`evaluateTasks` + `judgeContainsAll/judgeNonEmpty`）。
- 测试：taskpool 增量索引断言、task-eval 单测、retrieval-benchmark 基线。

### 9.6（#6）Parity 扩展 + OTel route/IPC 根 span

- **governance parity**：`generate_fixtures.py` 加 `gen_governance()`（白名单 / 阻断 / 审批 / 未知目标 × strictMode）；
  `plane-governance/__tests__/parity/{fixtures/python-v2.json, python-v2.spec.ts}`（9 tests）断言 TS `PolicyEngine.check`
  与 Python 同口径（allowed/result/riskLevel/`POLICY_BLOCKED`）。ContextEngine/taxonomy parity
  后于 2026-07-08 完成；HookBus 本体无 Python 对手，不设跨实现 parity（见 §12.8）。
- **OTel route/IPC 根 span**：`telemetry-otel` 新增 `withRootSpan(name, fn, {attributes})`——用 `startActiveSpan`
  把一次 HTTP/IPC 调用包成根 span，agent 内部 hook→span 因 `attach.ts` 用 `context.active()` 作父 → 自动挂到根下
  （需进程注册带 AsyncLocalStorage 的 ContextManager，`bootstrapNodeOtel` / `NodeTracerProvider.register()` 满足）。
  server `/api/chat`（stream/非 stream）与 desktop `IPC.CHAT` 已包裹。
  测试：`telemetry-otel/__tests__/root-span.spec.ts`（2 tests，NodeTracerProvider 真上下文，断言 `parentSpanId` + ERROR 标记）。

### 9.7（#7）回写路线表 / 文档债 + 未提交盘点

- 本节即文档回写；路线表 #3/#6(OTel)/#10/#12 状态已更新。
- **未提交盘点**（无 tag，归 `[Unreleased]`）：约 40 个文件改动 + 30 个新文件，集中在
  core(loop/hooks)、telemetry-otel、dormant、shared、taskpool、plane-execution/memory/governance、apps(cli/server/desktop)。
  仓库根 `.dockerignore`/`.env.example`/`deploy.sh`/`docker-compose.yml`/`nginx.conf` 仍是历史未跟踪（Python v2 部署相关，不属本阶段）。

### 9.8 历史留项状态（以当前工作队列为准）

- ~~RFC-004 utility process 蒸馏 worker（`mine()` 仍在 main 跑）~~。✅ 2026-07-08 完成 opt-in worker + 内联回退（§12.5）。
- ~~增量检索索引**接进 agent**（需 fragment change-feed 把 memory 写入广播给 HybridRetriever）~~。✅ 2026-06-30 完成（§十 A1）
- ~~HybridRetriever 换 LanceDB 原生 FTS（#10 余下部分）~~。✅ 2026-07-08 完成（存储层原生 FTS + RRF 融合 + server opt-in，见 §12.2）
- ~~Parity 扩展：Hooks / ContextEngine（governance 已接）~~。✅ ContextEngine/taxonomy 已完成；HookBus 无 Python 对手（§12.8）。
- ~~`pruneEvents(olderThanTs)`（#11 dormant 事件磁盘清理）~~。✅ 2026-06-30 完成（保留策略 + hydrate/record/mine 三处触发，见 §10.7）
- ~~#6 打包发布代码~~。✅ builder/updater/release workflow 已就绪；图标、签名、Linux CI 与首个真实 release 仍是运维验收项。
- ~~abTest / self-consistency 的长跑可观测验证（脚手架已就位，缺真实跑批数据）~~。✅ 2026-06-30 longrun harness 落地（§十 A2，真实跑批仍需 `RUN_LONGRUN=1` + LLM key 手动触发）

---

## 十、Memory Flywheel：增量检索 + 长跑验证 + 可强化分类器（2026-06-30）

> 起因：用户聚焦产品价值，要把「记忆」「检索」「分类」串成一个共享**使用反馈**的飞轮——
> 每次 `agent.run()` 的 (query → outcome) 信号同时喂给会话级增量检索索引与可强化分类器，
> 让两者一起「越用越好」。设计记录（含流程图/分阶段/风险/验证口径）见
> [`phase-flywheel-design.md`](./phase-flywheel-design.md)（由 Cursor Plan 归档）。
> 已提交（`79ed788` 主体 + `d5caa63` route.topK；归 CHANGELOG `[Unreleased]`）。**三个 opt-in 开关默认全关 → 默认行为零变化。**

### 10.1（A1）fragment change-feed + 会话级增量 HybridRetriever

- **change-feed**：`HookEventMap` 加 `event.MEMORY_WRITTEN`（`{ fragment, op }`）。`MemoryStore`
  在 `add*` / `remove` / 短期溢出晋升（`op:"update"`）/ 工作记忆溢出丢弃（`op:"remove"`）发事件；
  `PersistentMemoryStore.reassignMemoryType` 补 `op:"update"`。hydrate 直推**不发**事件（用 `index()` 种子）。
- **`MemoryHybridIndex`**（`@openintj/taskpool/src/memory-hybrid-index.ts`）：`seed()` 初始化 + `subscribe(hooks)`
  增量 `upsert`/`remove`，`search()` 支持 `memoryTypes`/`taskTags` 过滤（有过滤时超额取再裁，保证 topK）。
- **接主循环（opt-in）**：`ContextEngineOpts.candidateRetrieve` 注入点；三端 `OPENINTJ_LOOP_HYBRID=1`
  时走 hybrid 候选，`fragmentsToRanked`（`plane-memory/src/retriever.ts`）转回 `RankedMemory` 仍过
  ShaderPipeline / taskType boost / accessCount。`HybridRetriever.search` 加 per-query `configOverride`。
- **测试**：`plane-memory/__tests__/change-feed.spec.ts`、`taskpool/__tests__/memory-hybrid-index.spec.ts`、
  `plane-memory/__tests__/context-engine-hybrid.spec.ts`。

### 10.2（A2）长跑验证「越用越好」可观测

- **harness**：`@openintj/shared/src/longrun-eval.ts`——`runLongRunSession`（逐轮命中/token/judge + 改进曲线）、
  `runLongRunAb`（多变体打分对比）、`formatLongRunRow/Turns/Ab`；`longrun-scenarios.ts` 场景 fixtures。
- **token 指标**：`TaoLoop` 累计 `react.totalTokensSpent` → `TaoResult.totalTokensSpent` + `ctx.metrics`。
- **OTel counter**：`attachOtelToHooks` 加 `openintj.retrieval.hit`（`event.MEMORY_LOADED` 命中 +1）与
  `openintj.tokens.spent`（`event.LOOP_ITERATION` 累计）。
- **真实跑批**：`apps/cli/__tests__/longrun.harness.spec.ts`（`RUN_LONGRUN=1` 门控，不进 CI）跑真实 agent +
  classifier-on/off A/B（质量不退守护）。
- **测试**：`shared/__tests__/longrun-eval.spec.ts`（mock agent）、`telemetry-otel/__tests__/metrics.spec.ts`。

### 10.3（CLF）前端可强化分类器

- **新包 `@openintj/classifier`**：`ReinforcingClassifier`（embed kNN/质心 + 软置信度；低置信/无 exemplar
  回退 `detectTaskType` 关键词启发式——零 token；`reinforce` 升/降权 + 合并相似 + LRU 封顶；`toState`/`loadState`）。
  种子 `seeds.ts`（`DEFAULT_SEEDS`）；路由 `routing.ts`（`decideRoute` 高置信简单类→`enableReact:false` 降 token、
  `outcomeSignal` status→反馈信号）。
- **持久化**：`ClassifierStore` 接口 + `InMemoryClassifierStore`（默认）+ `SqliteClassifierStore`
  （`@openintj/storage-sqlite/src/classifier.ts`，仿 dormant）；`hydrate()`/`persist()` 接入。
- **接 `agent.run`（三端）**：`TaoLoop.run` 加可选 `taskType`/`enableReact`；`MemoryPlane.recordUserInput/Output`
  加 `extraTags`（带分类 label）。三端 `enableClassifier` opt（env `OPENINTJ_CLASSIFIER=1`）：预分类 → taskType +
  降 token 路由 → 记忆带 label → 收尾 `reinforce`。real 模式自动挂 `SqliteClassifierStore`（`<dataDir>/classifier.sqlite`），
  `close()` 关闭；CLI 在线程同步装配下用**首次 run 懒 hydrate/seed**。
- **测试**：`classifier/__tests__/{reinforcing-classifier,store}.spec.ts`、`storage/sqlite/__tests__/classifier.spec.ts`、
  `core/__tests__/tao.spec.ts`（taskType/enableReact 委派）。

### 10.4 验证 & 装配清单

- **自检**：`pnpm exec turbo run typecheck test --concurrency=1` → **58/58 task successful**（typecheck 全绿、各包 vitest 全过）。
- **新增包**：`@openintj/classifier`（已加 `pnpm-workspace.yaml` / 根 `tsconfig.json` refs）。
- **env 开关**（默认全关）：`OPENINTJ_LOOP_HYBRID=1`（主循环走 hybrid 候选）、`OPENINTJ_CLASSIFIER=1`（前端分类器）、
  `RUN_LONGRUN=1`（长跑 harness）。
- **桌面端预览验证**：TokenHub 迁移后（`hy3-preview`）实测对话链路通；旧混元平台 `hunyuan-turbos-latest`
  已于 2026-06-22 下线，`.env.local` 改走 `HUNYUAN_BASE_URL=https://tokenhub.tencentmaas.com/v1`。

### 10.5 TokenHub 迁移 + 联网搜索恢复（2026-06-30 续）

- **TokenHub 迁移**：旧混元平台 `hunyuan-turbos-latest` 于 2026-06-22 下线、整个旧平台 9-30 停服。
  `.env.local` 改走 TokenHub（OpenAI 兼容）：`HUNYUAN_BASE_URL=https://tokenhub.tencentmaas.com/v1`、
  `HUNYUAN_MODEL=hy3-preview`。客户端 `baseUrl` 自动追加 `/chat/completions`，**只填到 `/v1`**。实测对话通。
- **联网搜索恢复**（旧平台内建 search 随平台下线、TokenHub 改 Responses API 独立产品、参数未公开）：
  改走 **Function Calling + 外部搜索后端**（provider 中立）。新增 `@openintj/plane-execution/src/web-search-tool.ts`：
  `createWebSearchTool`（Tavily / Brave）+ `resolveWebSearchConfig`（env 推断 provider/key）。
  三端 `search` 工具优先级：外部 Web Search > 混元内建（仅旧平台）> 占位。
  env：`OPENINTJ_SEARCH_PROVIDER` + `OPENINTJ_SEARCH_API_KEY` 或 `TAVILY_API_KEY`/`BRAVE_API_KEY`（默认不配 → 零开销）。
  测试：`plane-execution/__tests__/web-search-tool.spec.ts`（10）。

### 10.6 本轮仍未做（飞轮衍生）

- TokenHub **Responses API** 原生联网搜索（若想用官方 search 而非第三方；参数需从模型详情页取）。
- 长跑 A/B 的真实跑批数据沉淀（harness 就绪，缺带 key 的批量结果）。
- 分类器路由策略调参（`RoutingPolicy` 阈值目前是保守默认）。

### 10.7 #11 Dormant 事件清理收尾（2026-06-30 续）

防 `dormant_events` 磁盘表无限增长。接口 / 双适配器 / runtime 清理逻辑此前已落，本轮补齐
**「不依赖 mine() 的兜底触发」**——此前自动清理只在 `mine()` 末尾跑，而 `mine()` 只由用户显式触发
（server `POST …/dormant/mine`、desktop `DORMANT_MINE` IPC），长会话不 mine 时磁盘表照样涨。

- **保留策略**（`DormantRuntimeOpts`）：`eventRetentionMs`（按时间）/ `maxDiskEvents`（LRU 条数），可叠加。
- **三处触发**：
  1. `mine()` 末尾 `maybeAutoPrune()`（原有）；
  2. `hydrate()` 启动末尾 `maybeAutoPrune()`——重启即收敛磁盘表；
  3. `record()` 每累计 `autoPruneEveryNEvents` 条触发一次（配了保留策略时默认 256；显式 `0` 关闭）。
- **装配**：server / desktop `DormantRuntime` 默认 `maxDiskEvents: 50_000`（`dormantOpts` 可覆盖）；CLI 不挂 dormant。
- **测试**：`dormant/__tests__/persistence.spec.ts` 新增 3 例（record 阈值触发 / hydrate 收敛 / `=0` 关闭），dormant 包 53 tests 全绿。

## 十一、技能系统（Phase 1 作者能力包，2026-07-01）

> 起因：让 agent「越用越会用」——把可复用的做法沉淀成**能力包**，按 query 命中才注入，
> 省 token 又不改默认行为。Phase 1 落地作者编写的 `SKILL.md`，Phase 2（仅铺垫）再加自学习蒸馏。
> 设计记录（流程图/分阶段/风险/验证口径 + Phase 2 铺垫）见
> [`phase-skills-design.md`](./phase-skills-design.md)（由 Cursor Plan 归档）。
> **opt-in 开关 `OPENINTJ_SKILLS=1` 默认关 → 默认行为零变化。**

### 11.1 新包 `@openintj/skills`

- **`Skill` 类型 + `SkillSource` 接口 + `FsSkillSource`**（`fs-source.ts`）：`SKILL.md` frontmatter
  `id/name/description/triggers?/taskTypes?/priority?/version?` + body 正文；极简 frontmatter 解析
  （`frontmatter.ts`，不引 YAML 依赖）；递归发现 `SKILL.md`，非法 `taskType` 过滤、缺 description/body 跳过、
  id 兜底目录名；同 id「后源覆盖」。`resolveSkillDirs`（内建 + `OPENINTJ_SKILLS_DIR` 分号/逗号分隔，不切盘符冒号）、
  `builtinSkillsDir()`（用包自身 `import.meta.url`，src/dist 都指向 `../skills`）。
- **`SkillRegistry`**（`registry.ts`）：多源载入 + 用注入 embedder 预计算「name+desc+triggers」匹配向量 + 轻量目录。
- **`SkillSelector` + `renderSkillPrompt`**（`selector.ts`）：embed 余弦 + trigger 关键词加成 + taskType 加成，
  过阈值（默认 0.35）取 top-k（默认 2），正文按 token 预算封顶（默认 700，至少留最高分一个）；渲染成 `[技能]` 块。
- **共享 helper `assembleSkillContext`**（`agent-helper.ts`）：三端共用，载入 + 选择器 + 按 (taskType,query) 记忆化
  （上限 128 清空防泄漏）+ 命中发 `event.SKILL_SELECTED`；无可用技能返回 `undefined`（调用方零注入）。
- **种子技能**（`packages/skills/skills/`）：`code-review` / `web-research` / `debugging`，随包 `files` 发布。

### 11.2 三端集成 + 可观测

- **注入点**：三端 `contextProvider`（`OPENINTJ_SKILLS=1` opt-in），复用 store embedder；技能块拼在
  **persona 之后、`[记忆参考]` 之前**（CLI 无 persona 则直接接 base）。CLI 工厂同步 → 持有 Promise 在异步 provider 里 await。
- **事件**：`HookEventMap` 加 `event.SKILL_SELECTED`（`{ skills:{id,score}[]; query }`）；
  `attachOtelToHooks` 加 counter `openintj.skill.hit`（每次注入的每个技能各 +1，attribute=skill）。

### 11.3 验证 & 装配清单

- **自检**：`turbo run typecheck --concurrency=1` → **39/39 全绿**；lint（biome）touched 文件全过；
  各 touched 包 vitest 单 fork 全过（skills 13、core 98、telemetry-otel 20、cli 18、server 48、desktop 33）。
  ⚠️ 本机内存吃紧，`turbo run test` 默认多线程 worker 会 OOM（`Zone Allocation failed`）——
  用 `vitest --pool=forks --poolOptions.forks.singleFork=true` + `NODE_OPTIONS=--max-old-space-size=4096` 逐包跑即通过。
- **新增包**：`@openintj/skills`（已加 `pnpm-workspace.yaml` / 根 `tsconfig.json` refs / 三端 `package.json` + `tsconfig.json` refs）。
- **env 开关**（默认关）：`OPENINTJ_SKILLS=1`（启用技能注入）、`OPENINTJ_SKILLS_DIR`（追加自定义技能目录）。

### 11.4 Phase 2 —— 自学习闭环（2026-07-07，已实现）

> opt-in 分级：`OPENINTJ_SKILLS_LEARN=1`（隐含 `OPENINTJ_SKILLS`）默认关 → 默认行为零变化。
> 加权抄 classifier `reinforce(outcomeSignal)`、蒸馏/审批抄 dormant `propose→approve→inject`、
> 持久化抄 storage-sqlite「接口在领域包、实现在 storage 包」。

- **加权核 `SkillLearningRuntime`**（`skills/src/learning-runtime.ts`）：`hydrate` 载 store；
  `noteSelected(query,taskType,ids)`（由 `assembleSkillContext.onSelected` 驱动，记本轮命中）；
  `recordOutcome(query,taskType,status,{finalAnswer,toolsUsed})` → 命中技能 `reinforce(skillOutcomeSignal(status))`
  （completed +1 / failed|timeout −0.5 / else +0.2，有界 clamp、写穿 store）+ 成功轨迹进 buffer；`weightFor`。
  `SkillSelector` 加**有界权重偏置**（`weightFor`×`weightGain=0.05`，`weightBiasCap=0.3` 封顶，语义仍主导）。
- **蒸馏/审批**：`distill()` 用户触发——`createLlmSkillDistiller`（接 agent LLM，JSON 容错，失败回退启发式）
  或启发式（按 taskType 聚类 + 高频 query 词 + 模板 body）产 `SkillProposal(pending)`，跨次按 candidate id 去重；
  `listProposals`/`approve`/`reject`/`revoke`——`approve` 写 `store.upsertApprovedSkill` + `onSkillsChanged`
  触发 `SkillContext.reload()`（重嵌入，新技能立即可选中）。
- **DB 源 + 持久化**：`DbSkillSource`（读 `runtime.listApproved()`，与 `FsSkillSource` 并列进注册表）；
  `SqliteSkillStore`/`createSqliteSkillStore`（`skill_approved`/`skill_proposals`/`skill_weights` + 迁移 v1，
  默认 `<dataDir>/skills.sqlite`；real 模式挂它，否则 `InMemorySkillStore`）。
- **可观测 + 审批入口**：`event.SKILL_PROPOSED` + counter `openintj.skill.proposed`；server HTTP
  `/api/skills/distill|proposals|proposals/:id/{approve,reject,revoke}|(GET)/api/skills`，desktop IPC 镜像
  （`SKILLS_*` + preload）；未启用统一 `skills_learning_not_enabled`(503)。**桌面审批 UI 面板暂缓**（后续抄 `DormantPanel`）。
- **验证**：`turbo run typecheck --concurrency=1` → **39/39 全绿**；lint touched 全过；单 fork vitest
  skills 32 / storage-sqlite 31 / telemetry-otel 21 / server 55(+8 skip) / cli 18 全过。
- **env 开关**（默认关）：`OPENINTJ_SKILLS_LEARN=1`。

### 11.5 桌面「技能审批」UI 面板（2026-07-08，已实现）

> 把 Phase 2 自学习闭环接到用户手上——此前只有 HTTP/IPC 后端，桌面端没界面。抄 `DormantPanel` 落地。

- **新组件 `SkillPanel.tsx`**（`renderer/components/`）：右侧栏第 4 个 tab「技能」。
  - 顶部「蒸馏」按钮 → `skillsDistill()`（成功轨迹提炼候选技能提案）。
  - status filter：pending / approved / rejected / revoked / all；每条提案显示技能名 / 描述 /
    证据（命中次数 + taskType + 示例 query）。
  - pending → ✓批准 / ✗拒绝；approved → 撤销。底部「生效技能」折叠区显示当前学习技能 + 权重。
  - `status.skills === undefined`（未启用 `OPENINTJ_SKILLS_LEARN`）→ 显示未启用提示 + 启用方法。
- **状态贯通**：`ipc-protocol` 新增 `SkillProposalDto`/`SkillListResponse`/`SkillDistillResponse`/
  `SkillDecisionResponse`/`SkillActiveDto`/`SkillActiveResponse`/`SkillLearningError` schema +
  `StatusResponse.skills`（`{enabled, pendingProposals, activeSkills}`）；desktop `agent.status()`
  暴露 `skills`；preload 6 个 skill API 从 `Promise<unknown>` 收窄到精确联合类型。
  `App.tsx` 用 `status.skills.pendingProposals` 给 tab 加待审批角标（同 Dormant）。
- **测试**：`ipc-handlers.spec.ts` 扩到 31（+4：未启用统一 `skills_learning_not_enabled`、注册全 channel、
  完整链路 distill→list→approve→active+status.skills schema 校验、approve ghost → not_found）。
  typecheck 全绿、biome touched 全过、desktop vitest 33+ 全过。

### 11.6 技能系统后续（2026-07-08 已推进，见 §12.8）

- ~~权重衰减 / LRU~~ ✅ 读时指数半衰期（`weightHalfLifeSec` / `OPENINTJ_SKILL_WEIGHT_HALFLIFE_SEC`）：`weightFor`
  按距 `lastUsed` 的时长衰减，`reinforce` 累加前先衰减旧值；不设即历史行为。
- ~~工具子集绑定~~ ✅ `Skill.tools` 硬隔离：技能块渲染「本轮仅可使用工具」，三端 ToolHub 按并集收窄 list/call。
- ~~蒸馏质量~~ ✅ `createLlmSkillDistiller` 校验强化：name/body 必填 + body 最小长度 + 各字段截断 + triggers/tools
  归一去重 + taskTypes 枚举校验 + 批内去重；新增 `llm-distill.spec`（12 例）。
- 仍未做：蒸馏候选**语义相似度**去重（当前按 id/name）；新工具类型仍需作者在 ToolHub 注册后才能被技能绑定。

---

## 十二、检索性能 / 规模：#10 LanceDB 原生 FTS + #3 嵌入基准（2026-07-08）

> 详见 `docs/architecture/retrieval-benchmark.md`（方法 / 实测数字 / 复现命令）。

### 12.2（#10）HybridRetriever 换 LanceDB 原生 FTS（已实现）

- **动机**：`MemoryHybridIndex` 每次 query 在内存对全部文档算 BM25 + cosine（O(N)/query），
  N>10k fragment 时全表扫描成本上升。LanceDB 自带 BM25 原生 FTS，可把词法检索下推到存储层。
- **存储层能力**（`packages/storage/lance/src/`）：`VectorStore` 新增可选 `supportsFts` /
  `ensureFtsIndex()` / `searchText(query, opts)`。
  - `LanceDBVectorStore`：`table.createIndex("content", {config: Index.fts()})` 建索引 +
    `table.search(query, "fts")` 查询；旧版 / 不支持 / 建索引失败时 `supportsFts=false` 静默降级。
  - `InMemoryVectorStore`：实现 BM25-lite `searchText`，让融合逻辑在**不装 LanceDB** 时也能单测。
- **融合**（`fusion.ts`）：`rrfFuse` + `hybridVectorSearch(store, {query, queryEmbedding, topK, ...})`——
  向量榜 + FTS 榜各出一份，RRF（只依赖名次，天然适配 cosine/BM25 异构分数）融合；`searchText`
  缺失 / 空时自动降级为纯向量。
- **接入**（`apps/server/src/hybrid-retrieve.ts`，opt-in）：默认仍走内存路径；`useLanceFts:true` 或
  env `OPENINTJ_LANCE_FTS=1` → `hybridVectorSearch(persistentStore.vectorStore, …)`，结果映射回
  `MemoryHybridHit`（RRF 分入 `components.rrf`）。
- **测试**：storage-lance 22（+9 fusion +4 in-memory FTS）、server hybrid-retrieve 17（+3 FTS 路径）全绿。

### 12.1（#3）嵌入基准：simple 实测 + 双路径 harness（simple 已跑，xenova/ollama 待回填）

- **双路径 harness**（`packages/planes/memory/src/eval/retrieval-benchmark.ts`）：
  - `benchmarkRetrieval`：`MemoryRetriever` 产品路径（cosine + 关键词重叠 + 时间衰减）。
  - `benchmarkEmbedderCosine`（新）：**纯 cosine**，隔离出 embedder 的语义能力（去掉关键词兜底）。
- **simple 实测**（12 篇 × 6 query，k=4）：产品路径 nDCG **0.773**；纯 cosine 仅 **0.396**——落差量化了
  SHA-256 词袋哈希「无真语义、维度无关」的短板，也是引入神经嵌入器的收益空间。
- **xenova/ollama**：本机未装 `@xenova/transformers` / ollama 服务未起，暂无数字；装好后
  `RUN_EMBED_COMPARE=1 pnpm --filter @openintj/plane-memory test retrieval-benchmark` 回填。
- **测试**：benchmark spec 加纯 cosine 维度不敏感断言；compare 分支并列打印两套评分表。

### 12.3 治理接进工具执行（RFC-004 §8，2026-07-08）

> 纠偏：旧盘点里「🔴 执行平面工具是 mock」已过时——fs/命令工具早是真实沙箱（`createWorkspaceTools`）。
> 真正的缺口是**治理平面从不被调用**：`GovernancePlane` 三端都 new 了，但 `checkAndRecord` 只在单测里跑，
> 工具执行链路无策略/配额门禁。本次补上。

- **`ToolHub` 通用 gate**（`packages/planes/execution/src/tool-hub.ts`）：`ToolHubOpts.gate?: ToolGate`，
  在 `tool.beforeCall` 之后、handler 之前执行；抛错 → `ToolCallResult.success=false`（**不触发熔断**——
  治理拒绝不是工具故障，也**不发 `tool.onError`** 避免被当可重试）。execution **不反向依赖** governance。
- **`GovernancePlane.checkToolCall(command)`**（governance）：镜像 `checkAndRecord`，但走**每分钟工具配额**
  （`checkToolQuota`/`recordToolCall`）+ 策略黑名单 + 审计；不消耗 API 配额。
- **`createToolCallGate(governance)`**（governance）：把 plane 包成 `TOOL_CALL` 命令的 gate，三端 agent
  `new ToolHub({ hooks, gate: createToolCallGate(governance) })`。
- **桌面 IPC**：`WORKSPACE_READ/WRITE` 直连沙箱的路径也过同一 gate（补 RFC-004 §8「Governance → fs」）。
- **默认不回归**：白名单含 `read_file/search`；`write_file/execute_command` 非黑非白 → 放行；仅黑名单目标
  （`shell-delete` 等）或超配额（默认 20/min）被拦。运行时 `policyEngine.block("write_file")` 可动态收紧。
- **测试**：execution +4（gate 拒绝不触发熔断 / 放行 / afterCall 可观测但不 onError）；governance +6
  （checkToolCall 放行/黑名单/动态 block/配额 + createToolCallGate 放行/拦截）；cli +1 端到端拉黑拦截。

### 12.4 钝化记忆 persona 注入闭环（RFC-003 §3.6 收官，2026-07-08）

> 纠偏：旧盘点 §8.2 说「persona 从不注入」已过时——server/desktop 早在飞轮阶段就把
> `dormant.personaSystemPrompt()` 接进了 `contextProvider`。本次补齐**其余四件**，让 §3.6 四条验收全绿。

- **`getPersona()` 出口**（`packages/dormant/src/dormant-runtime.ts`）：语义等同 `snapshot()`，作为
  装配层/UI 读「已生效人格」的规范入口名（§3.6 附录 A）。desktop `DORMANT_PERSONA` IPC 与 server
  `GET /api/dormant/persona` 均切到 `getPersona()`。
- **CLI 注入 parity**（`apps/cli/src/agent.ts`）：新增 `enableDormant`/`dormantOpts`/`enablePersona`。
  CLI 为**内存态**（不挂 adapter、无需 hydrate）：`run()` 每轮 `record` 用户输入；`contextProvider`
  在技能包/`[记忆参考]` 之前拼 `[用户画像]`。三端注入顺序统一：**persona → skills → 记忆参考**。
- **A/B 杠杆**（`packages/shared/src/persona-config.ts` `resolvePersonaInjection`）：
  `enablePersona`（opts）> `OPENINTJ_PERSONA`（env，`0`/`false` 关）> 默认开。三端 `contextProvider`
  用它 gate persona 行——关闭即得「无 persona 基线组」，满足 §3.6 #3 可观测 A/B。
- **脱敏**：`DormantRuntime.record()` 落库前默认过 `defaultRedactor`（邮箱/卡号/key/身份证/电话），
  `redactor:null` 显式关闭。此前已实现，本次仅纳入验收确认。
- **revoke 全链路**（§3.6 #4 可回退）：runtime/桌面 IPC 早已有；本次补 server
  `POST /api/dormant/proposals/:id/revoke`（仅 `applied` 可撤，否则 404 `not_found_or_not_applied`）+
  list 接受 `status=revoked` + `DormantPanel` 加「已撤销」tab 与 applied 卡片上的「撤销」按钮
  （抄 `SkillPanel` 形态）。
- **测试**：shared +4（`resolvePersonaInjection` 优先级）；cli +4（initialPersona 注入命中 `[用户画像]`、
  `enablePersona:false` 不注入、record→mine→approve 全链路、未启用则 `agent.dormant` undefined 且不注入，
  经 `react.beforeThought` 断言最终 system prompt）；server +2（revoke 删字段/version++/可 list、非 applied 404）。
  typecheck + biome 全绿。

### 12.5 RFC-004 workspace 能力面收官（2026-07-08）

> 纠偏：旧盘点 §8.2 #4 说「workspace 读写 / config 面 / utility worker 全缺」——**后端其实早已在**
> （WORKSPACE_READ/WRITE/INFO/PICK_DIR + `fs.watch → EVT_WORKSPACE` + `ConfigService`，均有契约测试、
> 读写过治理 gate）。真正缺的是 **renderer 消费面**、**config 字段完整性/启动接线**、**utility 挖掘 worker**。

- **设置面板**（`apps/desktop/src/renderer/components/SettingsPanel.tsx`，接进右栏「设置」tab）：
  - 工作区段：`workspaceInfo()` 显示沙箱根/命令开关/白名单；「选择目录…」→ `workspacePickDir()`
    （持久化到 config，下次启动生效）。
  - 配置段：`getConfig()`/`updateConfig()` 编辑 llmProvider / retrievalMode + 7 个开关；运行时项保存后热重装 agent。
  - 变更流段：`onWorkspaceEvent()` 实时列出 `fs.watch` 的 rename/change（最近 20 条）。
- **config 字段补全**（`ipc-protocol.ts` `AppConfigSchema`）：新增 `enablePersona / enableSkills /
  enableSkillLearning / enableClassifier`，并在 `main/index.ts` 启动装配时透传给 `assembleDesktopAgent`。
- **utility 挖掘 worker**（`@openintj/dormant`，RFC-004 §2）：
  - `mine-worker.ts`：`worker_threads` 入口，跑 `PatternMiner.mine`（仅可序列化 opts，无 `llmExtract`）。
  - `worker-miner.ts`：`runMineInWorker`（真实线程）+ `mineWithWorkerFallback`（先 worker、任何失败回退内联，
    `runner` 可注入便于测试）。
  - `DormantRuntime` 加 `mineRunner` 选项 + `lastMineUsedWorker` 标记：配了且无 `llmExtract` 才下放；
    带 `llmExtract`（LLM 在主线程）恒内联。desktop `OPENINTJ_DORMANT_WORKER=1` / `dormantMineWorker`
    启用。dormant 包被 externalize，`dist/mine-worker.js` 随 node_modules 分发，`new URL(...import.meta.url)`
    运行时可解析（已用真实线程 e2e 验证）。
- **测试**：dormant +6（fallback 编排：worker 透传 / 抛错回退等价 PatternMiner / 空事件；DormantRuntime
  mineRunner 接线：走 worker / llmExtract 不走 / 未配不走）；desktop config schema 扩展经现有 ipc-handlers
  契约测试守护。typecheck + biome 全绿；真实 worker 线程手动 e2e 通过（3 patterns off-thread）。
- **未做（明确留量）**：renderer 无 jsdom 单测 → 面板交互留 Playwright。config 热重载已于 2026-08-24 落地。
  `skillLearning` 已由保存配置透传到 desktop agent 装配，不再是 env-only。

### 12.6 方向一并发原语接真实 agent：self-consistency 并发上限（2026-07-08）

> 纠偏 §11「方向一/二并发原语只是原型」：§9.3 已把 `forkJoin` 接进三端自一致性主路径，但仍是**无界**
> 全并发（N 个采样一次性打满 LLM）。本次补的是**有界并发**——把 `@openintj/concurrency` 的 `Semaphore`
> 经 `forkJoin.concurrency` 真正用在产品路径上。

- `forkJoin` 加 `concurrency` 选项（`ForkJoinOpts`）：`1..<items.length` 时用内部 `Semaphore` 限流——
  拿到 permit 才调 `fn`，`finally` 释放；不传 / `<=0` / `>=items.length` → 全并发（历史行为不变，零开销）。
- `SelfConsistencyConfig` 加 `maxConcurrency`；`resolveSelfConsistency` 从 `opts.maxConcurrency` >
  `OPENINTJ_SELF_CONSISTENCY_CONCURRENCY` 解析（不设即全并发）。
- cli/server/desktop 三端 `AgentOptions.selfConsistency` 加 `maxConcurrency`，`run()` 里透传给
  `forkJoin({ concurrency })`——给昂贵的多采样设并发上限，避免一次性打满下游配额。
- `@openintj/concurrency` index 头集成状态更新：`forkJoin` + `Semaphore`（经 `forkJoin.concurrency`）
  标为**已接入产品路径**（自一致性），Mutex/Channel/CV/Pool/Backpressure 仍标实验性。
- **测试**：concurrency +2（peak ≤ 上限；上限 ≥ 总数则全并发 peak 达总数）；shared self-consistency +4
  （默认无 maxConcurrency / opts 透传 / env 读取 / opts 优先 env）。typecheck + biome 全绿。

### 12.7 小缺口收口：enableReact:false 退化分支 + function-calling 决策文档化（2026-07-08）

> 纠偏 §8.2 #5：两件"小缺口"其实在更早的会话已落地/已文档化，只是盘点未同步。本次核实、补测、纠盘点。

- **`enableReact:false` 退化分支**：`ReactStateMachine.runSingle`（`react.ts`）——跳过 ReAct 微循环与工具
  下发，做单次 LLM 调用直接作答，仍发 `react.beforeThought/afterThought/onStopCondition` 保持观测一致。
  `TaoLoop.run` 按 `opts.enableReact ?? config.enableReact` 选 `run`/`runSingle`；三端 agent 经分类器
  `decideRoute(cls).single`（高置信 + 简单类且非兜底）设 `enableReact:false`——纯对话/快速响应省 token/时延。
- **可达性守护**：core `tao.spec` 已有「enableReact=false → 单次调用、不解析 Action、不调工具」用例；
  本次补 `@openintj/classifier` `routing.spec`（`decideRoute` 的 single/topK 判定 6 例 + `outcomeSignal` 3 例），
  锁定「分类 → 退化路由」这一段此前无专测的链路。
- **function-calling 决策**：**ADR-001**（`docs/architecture/adr-001-react-tool-protocol.md`，2026-06-30 已采纳）
  记录「ReAct 统一走文本协议（Thought/Action/FINAL）而非 OpenAI function-calling」的理由（本地优先/多 provider
  中立、可观测、Python v2 parity）、代价（鲁棒性、token、无并行工具）、**重新评估触发条件**与迁移路径。
  RFC-001 §11 Q1 已引用该 ADR 并标记关闭。

### 12.8 #12 parity 扩展（ContextEngine/Hooks）+ 技能系统后续（2026-07-08）

**A. parity 扩展（`scripts/python-parity/generate_fixtures.py` +2 slice，core 55→81 parity 测试）**

> 两端 `build_context` 整体架构不同（Python `ConversationMessage`+`token//4` vs TS `ShaderPipeline`+`estimateTokens`），
> 全量 ContextEngine parity 代价大且脆。改为锁定其**确定性内核**与**共享枚举契约**——真正跨实现必须一致的部分。

- **context slice**（`ts/packages/core/__tests__/parity/{context.spec.ts,fixtures/context.json}`，+12）：
  `ContextBudget` 算术（`availableTokens`/`usageRatio`/`memoryBudget`/`needsCompaction@[0.5,0.8,0.9]`，6 组预算）
  与 `ShaderConfig.get_shader_for_task`（6 类 → shader mode）在 Python ↔ TS `ContextBudgetTracker`/`getShaderForTask`
  逐值等价（usageRatio 12 位小数）。
- **taxonomy slice**（`taxonomy.spec.ts`/`taxonomy.json`，+14）：这是 Python v2「Hooks/事件」最接近的对齐面——
  HookBus 发的框架事件用同一套 `EventType`。断言 `EventType`/`CommandType` 逐条相等；`ErrorCode` **Python ⊆ TS**
  （TS 多出 `HOOK_ERROR`/`STATE_TRANSITION_INVALID`/`LOOP_LIMIT_REACHED`/`REACT_DUPLICATE_LOOP` 等 hook/react 专用码）。
- **HookBus 本体无 Python 对手**：Python v2 用局部 `events: List[Event]`，无 HookBus 抽象，故无「HookBus 行为」跨实现
  parity 目标；其行为由 core 单测 + `hook-bus-bench` + concurrency observability 测试守护。parity 只锁其**事件/错误码分类**。
- 生成器只读 Python v2（冻结）；`context.json`/`taxonomy.json` 为新增确定性 fixture。memory/governance fixture 含
  `time.time()` 与浮点末位，重跑会漂动 → 本次未随之改动（保持既有）。

**B. 技能系统后续（`@openintj/skills`）**

- **权重衰减**：`SkillLearningRuntimeOpts.weightHalfLifeSec`（或 env `OPENINTJ_SKILL_WEIGHT_HALFLIFE_SEC`，
  `resolveSkillWeightHalfLifeSec` 三端接线）。读时指数衰减 `w*0.5^(age/halfLife)`：`weightFor` 供选择器偏置随冷却
  自然回落，`reinforce` 累加前先把旧值衰减到当下（陈旧高权重不永久霸榜）。不设 → 历史行为。
- **工具子集绑定**：`Skill.tools`（frontmatter `tools:` 解析 / sqlite `SkillSchema` default [] 向后兼容 / db 源透传 /
  蒸馏草案携带）。命中后技能块写「本轮仅可使用工具」；三端 ToolHub 按并集硬收窄 list/call。
  `skillToolAllowlist` 求命中并集，经 `assembleSkillContext.onSelected(query,taskType,ids,tools)` 暴露供装配方收窄。
  内建 seed（code-review/debugging/web-research）已声明 tools 示例。
- **蒸馏质量**：`createLlmSkillDistiller` 校验强化 —— name/body 必填 + body 最小长度（默认 16）+ name/desc/body 截断 +
  triggers/tools 归一去重 + taskTypes 校验到合法 `TaskType`（过滤幻觉）+ 批内按 id/name 去重；prompt schema 加 tools/taskTypes 约束。
- **测试**：skills 38→58（decay 4 + resolver 5 + tools（fs/selector/allowlist）+ llm-distill 12）；storage-sqlite skills +1
  （tools JSON 往返 + 旧行默认 []）。typecheck + biome 全绿。
