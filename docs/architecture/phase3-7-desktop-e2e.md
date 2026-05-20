# Phase 3.7 —— Desktop E2E（Playwright + Electron / #4）

> 状态：**已收官**（2026-05-20）  
> 仓库标签：`v3.0.0-alpha.7`  
> 覆盖路线图：[`phase2-complete.md` §九](./phase2-complete.md#九未完成--后续路线) #4

---

## 一、目标

Phase 3.5 把 Dormant 审批面板装进 renderer 后，桌面端共 4 个 React 组件
（`App` / `ChatPanel` / `TrajectoryPanel` / `DormantPanel` / `StatusBar`），
**渲染层 0 测试** ——
contract 已被 `__tests__/ipc-handlers.spec.ts` 全覆盖（21 个），
但 React 组件本身、preload bridge 在真 Electron 下能否跑起来、
chat → IPC → ReAct → 回流到 React state 这条全链路是否真的工作
没有任何机器化兜底。

本阶段补这条链路上唯一的关键缺口：
**用 Playwright `_electron.launch` 启动真主进程 + 真 BrowserWindow，
对真 DOM 做断言**。不引入 jsdom / @testing-library/react —— 这两条
线下次有需要再上（专门测 React state reducer 才有意义）。

## 二、设计

### 1. 选型：Playwright `_electron` vs Spectron vs WebdriverIO

| 候选 | 维护状态 | Electron 33 兼容 | 安装包大小 | 体感 |
|---|---|:-:|---|---|
| **`@playwright/test._electron`** | ✅ Microsoft 官方 | ✅ 1.60+ | ~50MB（不下浏览器） | **选** |
| Spectron | ❌ 2022 已弃 | ❌ | — | — |
| WebdriverIO + chromedriver | 维护中 | 部分 | ~100MB | 体感重 |

Playwright 直接 spawn Electron 二进制 + Chrome DevTools Protocol，
不需要装 Chromium，对 Linux CI 友好；API 与 web 测试一致，学习成本零。

### 2. 启动模型

```
+------------------+   _electron.launch  +-----------------+
| Playwright proc  | ──────────────────▶ | Electron main   |
| (single worker)  |                     |  - whenReady    |
+------------------+                     |  - assembleAgent|
        │                                |  - createWindow |
        │ firstWindow()                  +-------┬─────────+
        ▼                                        │
+------------------+    file://renderer/...      │
| React renderer   | ◀──────────────────────────┘
| (real DOM)       |
+------------------+
```

- **一个 worker**：Electron 主进程是单例资源，并发 launch 会撞窗口和
  IPC handler 注册（`ipcMain.handle` 同 channel 重注册抛错）
- **mock LLM**：env `LLM_PROVIDER=mock` → `HunyuanClient({ apiKey: "" })`
  → `mock-responses.ts` 返回固定文本，回答稳定可断言
- **关持久化**：`OPENINTJ_DESKTOP_NO_PERSIST=1` → memory 模式，
  不挂 LanceDB / better-sqlite3，启动时间从 ~10s 降到 ~7s
- **关 dormant**（默认）：跳过 `DormantRuntime` 装配；想测 dormant
  的 spec 用 `test.use({ extraEnv: { OPENINTJ_DORMANT: "1" } })` 自己开

### 3. 文件布局

```
apps/desktop/
  e2e/
    playwright.config.ts    # workers=1, testIgnore by env switch
    fixtures.ts             # electronApp + page fixture
    tsconfig.json           # 独立 e2e tsconfig（不污染 src 编译）
    tests/
      smoke.spec.ts         # 5 tests
      dormant.spec.ts       # 2 tests
```

`e2e/` 不被 `apps/desktop/tsconfig.json` 的 `include: ["src/**/*"]` 命中，
所以主 `tsc` 完全无视它；新加的 `pnpm typecheck` 增量跑
`tsc --noEmit -p e2e/tsconfig.json` 单独验它。

### 4. 默认 opt-in，不污染主 CI 路径

Playwright 在 Linux 上需要 GTK/X 一堆系统包，本地 Windows 又依赖装好
的 Electron 二进制，把它放进默认 `pnpm test` 会拖慢所有人。
策略：

- `playwright.config.ts` 顶部读 `OPENINTJ_PLAYWRIGHT=1`，
  没设就把 `testIgnore: ["**/*"]` 全开 → runner 直接 noop
- `pnpm --filter @openintj/desktop run test`（vitest）**不**触发 e2e
- 跑法只有一个：
  ```powershell
  $env:OPENINTJ_PLAYWRIGHT="1"
  pnpm --filter @openintj/desktop run e2e   # build + playwright
  Remove-Item env:OPENINTJ_PLAYWRIGHT
  ```
- CI 走专用 job `e2e-desktop`，跟 `e2e-persistence` 同级，Ubuntu + xvfb

## 三、测试覆盖（7 个）

### smoke.spec.ts（默认装配 = mock + no-persist + 无 dormant）

| # | 用例 | 断言点 |
|---|---|---|
| 1 | boots and renders header | `<header>` 含 `OpenINTJ` / `v3.0 Local Desktop` |
| 2 | status bar populates within 8s | `LLM:` 行可见 + `工具:` 列表非空（10s 内） |
| 3 | chat round-trip: 你好 → mock greet | 在 `bg-[#1e1e2e]` 主聊天区找到 `你好` 气泡 + `OpenINTJ Agent（mock 模式）` 回答 |
| 4 | trajectory tab counts after chat | 聊天主区出现 assistant `bg-[#313244]` 气泡；右栏 `推理轨迹` tab 计数 ≥ 1 |
| 5 | dormant tab shows '未启用' | 点 `Dormant` tab 出现 `OPENINTJ_DORMANT=1` 提示；切回 `推理轨迹` 提示消失 |

### dormant.spec.ts（`OPENINTJ_DORMANT=1` + mock + no-persist）

| # | 用例 | 断言点 |
|---|---|---|
| 6 | dormant tab not '未启用' | 不再有 `Dormant 子系统未启用` 文本；`Mine` 按钮可见；默认 filter `待审批` 显示 `暂无待审批 proposal` |
| 7 | clicking Mine produces summary | 点 Mine → 顶部出现 `扫描 N 条事件` 行（N=0 也算成功） |

### 关键定位技巧

聊天文本与 trajectory 面板的 `<pre>` JSON dump 都含 mock 回答原文，
直接 `getByText(/mock 模式/)` 会撞 strict-mode 多匹配。
最稳的办法是先用 tailwind 颜色 token 圈出主聊天区：

```ts
const chat = page.locator("div.bg-\\[\\#1e1e2e\\]");
const assistantBubble = chat.locator("div.bg-\\[\\#313244\\]");
```

文本断言全部链在 `chat` 这个父定位上，避免污染。

## 四、踩到的两个坑

### 坑 1：electron-vite 输出 preload `.mjs`，但 main 加载 `.js`

`out/preload/index.mjs` 实际生成，但 `src/main/index.ts` 写
`preload: path.join(__dirname, "../preload/index.js")` —— 路径直接错。

历史上没有暴露是因为：preload 加载失败时 Electron 只在 stderr 打一行警告
然后 renderer 继续起，只是 `window.openintj` 是 undefined；
而 vitest 单测全程不启动真 Electron，根本不走这条路径。

**修复**：`src/main/index.ts` 改成 `../preload/index.mjs`。
Electron 28+ 原生支持 ESM preload via `.mjs`，dev / prod 路径一致。

### 坑 2：Windows + Playwright `_electron.launch` 加 `--no-sandbox` 卡 30s 超时

最初 fixture 用：

```ts
args: [MAIN_ENTRY, "--no-sandbox"],
cwd: DESKTOP_ROOT,
```

Playwright launch 永远不返回，30s 超时。手工直接 spawn `electron.exe MAIN_ENTRY --no-sandbox`
完全 OK；用 `_electron.launch({ args: [MAIN_ENTRY] })`（无 sandbox flag、无 cwd）
也完全 OK。

**结论**：在 Windows + Electron 33 + Playwright 1.60 这一具体组合下，
`--no-sandbox` 触发 Playwright 内部 inspector-loader 卡死。
Linux 上不需要该 flag（sandbox 已是默认 / xvfb 不冲突）。

**修复**：去掉 `--no-sandbox` 和 `cwd`，只传 `[MAIN_ENTRY]`。
Linux CI 也照常工作（xvfb 让 sandbox 不抗议）。

## 五、CI 集成

`.github/workflows/ci.yml` 加 `e2e-desktop` job，结构对齐 `e2e-persistence`：

- needs: `lint-and-typecheck`
- Ubuntu-latest + Node 20
- 装 Electron 33 在 Ubuntu 24.04 需要的运行时：
  ```
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0
  libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0
  libcairo2 libasound2t64 libgtk-3-0 xvfb
  ```
  （`libasound2` 在 Ubuntu 24.04 改名 `libasound2t64`；如果以后底镜像换回 22.04 要回滚）
- 先 `pnpm --filter @openintj/desktop run build` 让 `out/` 存在
- `xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'`
  包一层 `pnpm --filter @openintj/desktop run e2e:run`
- 失败时 upload `playwright-report/` + `test-results/` 7 天

## 六、Schema / 文件清单

### 新增

- `ts/apps/desktop/e2e/playwright.config.ts`
- `ts/apps/desktop/e2e/fixtures.ts`
- `ts/apps/desktop/e2e/tsconfig.json`
- `ts/apps/desktop/e2e/tests/smoke.spec.ts`（5 tests）
- `ts/apps/desktop/e2e/tests/dormant.spec.ts`（2 tests）
- `docs/architecture/phase3-7-desktop-e2e.md`（本文）

### 改动

- `ts/apps/desktop/package.json`：
  - 加 devDep `@playwright/test ^1.60`
  - `typecheck` 串第二段 `tsc --noEmit -p e2e/tsconfig.json`
  - 新 script `e2e`（build + run）/ `e2e:run`（只 run）
- `ts/apps/desktop/src/main/index.ts`：preload 路径 `index.js` → `index.mjs`
- `ts/biome.json`：`files.ignore` 加 `**/test-results/**` 与 `**/playwright-report/**`
- `.github/workflows/ci.yml`：新增 `e2e-desktop` job
- `CHANGELOG.md`：新增 `3.0.0-alpha.7` 条目
- `docs/architecture/next-session.md`：标 Phase 3.7 完成、补踩坑清单

## 七、验证结果（本地，Windows 11 / Node 22.22）

```
$env:OPENINTJ_PLAYWRIGHT="1"
pnpm --filter @openintj/desktop run e2e

Running 7 tests using 1 worker
  ok 1 dormant.spec.ts:19  desktop dormant ... dormant tab shows mine button + pending filter
  ok 2 dormant.spec.ts:34  desktop dormant ... clicking Mine produces a summary line
  ok 3 smoke.spec.ts:14    desktop smoke ... boots and renders header
  ok 4 smoke.spec.ts:19    desktop smoke ... status bar populates within 8s
  ok 5 smoke.spec.ts:26    desktop smoke ... chat round-trip: 你好 → mock greet answer
  ok 6 smoke.spec.ts:39    desktop smoke ... trajectory tab counts after chat
  ok 7 smoke.spec.ts:55    desktop smoke ... dormant tab shows '未启用' when not enabled
  7 passed (34.8s)
```

回归（默认 CI 路径，没设 `OPENINTJ_PLAYWRIGHT`）：

```
pnpm lint                                       # exit 0（仍是 2 条 useExhaustiveDependencies warn）
pnpm exec turbo run typecheck --concurrency=1   # 33/33 successful
pnpm exec turbo run test --concurrency=1        # 33/33 successful（430 pass + 11 skip）
```

## 八、留给下一阶段的小尾巴

1. **`StatusBar` 文本切割 bug**：probe 时看到状态栏底部输出 `检索` 被截成 `检�?`，
   是 Windows codepage 把多字节字符压扁了，**不是真 bug**（只在终端体感，DOM 里正常）。
2. **renderer 单元测试**（jsdom + @testing-library/react）仍未引入。当前选择是
   "Playwright 覆盖端到端 + 组件 props 由 IPC contract 锁死"；如果将来想做
   细粒度 React 状态机断言，再单开。
3. **多窗口**：当前 desktop main 在 macOS `activate` 事件时会重建窗口，
   Playwright e2e 只覆盖单窗口冷启动。如果以后做多窗口工作流，e2e 要扩。
4. **Playwright 跑包**：Playwright 自带 chromium / firefox / webkit 二进制；
   `_electron` 模式完全用不到 → 安装时跳过浏览器下载可以再瘦身（`PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`）。
   本阶段没做，因为 pnpm install 没显式拉浏览器（pnpm 不自动跑 `playwright install`），
   实际安装包大小已经只是 playwright-core；如果以后有人手贱跑了
   `playwright install` 再考虑。
