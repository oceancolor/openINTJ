# 打包发布 & 自动更新（#6）

> 状态：**未签名 Windows NSIS 已于 2026-07-19 本机产出**（electron-builder 配置 +
> electron-updater + CI 发布工作流 + renderer 更新条均已接线）。
> 签名正式发布仍有运维阻塞（品牌图标、证书 secrets、合入 main、正式 tag）。

桌面端（`ts/apps/desktop`）用 **electron-builder** 出安装包、**electron-updater** 从 GitHub Release 拉更新。

## 一、构成

| 件 | 路径 | 作用 |
|---|---|---|
| 打包配置 | `ts/apps/desktop/electron-builder.yml` | Win `nsis` / macOS `dmg` / Linux `AppImage`；`publish: github (oceancolor/openINTJ)`；原生模块 `asarUnpack` |
| 自动更新 | `ts/apps/desktop/src/main/updater.ts` | `initAutoUpdater`：仅打包后启用（`app.isPackaged` / `OPENINTJ_FORCE_UPDATER=1`），防御式，异常只推 renderer 不崩主进程 |
| 主进程接线 | `ts/apps/desktop/src/main/index.ts` | 启动装 updater，退出 `dispose` |
| IPC 契约 | `ts/apps/desktop/src/shared/ipc-protocol.ts` | `UPDATE_CHECK` / `UPDATE_INSTALL` 调用；`EVT_UPDATE` 流式状态；`UpdateEventSchema` |
| 更新条 UI | `ts/apps/desktop/src/renderer/components/UpdateBanner.tsx` | 有可用更新/下载中/已就绪/出错时显示；已下载给「重启安装」（挂在 `App.tsx`） |
| CI 发布 | `.github/workflows/release.yml` | 打 `v*` tag → win/mac 构建 → `electron-builder --publish always` → GitHub Release |
| 原生 ABI 对齐 | `ts/apps/desktop/scripts/ensure-electron-abi.cjs` | 把 better-sqlite3 / lancedb 的 `.node` 对齐到 Electron ABI（打包/开发前置） |

更新事件状态机（`EVT_UPDATE.status`）：`checking → available → downloading → downloaded`（失败 `error`；未打包 `disabled`；无更新 `not-available`）。

## 二、本地打包

```bash
cd ts
pnpm exec turbo run build --concurrency=1        # 先构建各包 + electron-vite bundle
cd apps/desktop
node scripts/ensure-electron-abi.cjs             # 对齐原生模块 ABI（关键，否则运行时加载 .node 失败）
pnpm exec electron-builder --config electron-builder.yml --win   # 或 --mac / --linux
# 产物在 ts/apps/desktop/release/
```

不发布（只出本地安装包）时**不要**带 `--publish always`，也无需 `GH_TOKEN`。
Windows 若未开启开发者模式 / 创建符号链接权限，electron-builder 解压 winCodeSign 会失败；仅验证
未签名包时可临时追加 `--config.win.signAndEditExecutable=false`，正式签名构建不可使用该开关。

pnpm workspace 包由 electron-vite 打进主进程 bundle，并在 desktop 中列为 devDependencies；
不要把 `@openintj/*` 改回 production dependencies，否则 electron-builder 会沿 workspace symlink
走出 `apps/desktop`，在 asar 阶段报 `must be under ... apps/desktop`。

## 三、切一个正式 release（CI）

1. 确认要发布的 commit 已到 `main`（CI 绿）。
2. 版本号：`ts/apps/desktop/package.json` 的 `version` 决定安装包/`latest.yml` 版本号，需与 tag 对齐。
3. 打 tag 并推：
   ```bash
   git tag v3.0.0
   git push origin v3.0.0
   ```
4. `release.yml` 触发：windows-latest + macos-latest 各自 `electron-builder --publish always`，用内置 `GITHUB_TOKEN`
   上传安装包 + `latest.yml` / `latest-mac.yml` 到对应 GitHub Release；签名环境由下列 repository
   secrets 显式映射：
   - Windows：`WIN_CSC_LINK`、`WIN_CSC_KEY_PASSWORD`
   - macOS：`MAC_CSC_LINK`、`MAC_CSC_KEY_PASSWORD`、`APPLE_ID`、
     `APPLE_APP_SPECIFIC_PASSWORD`、`APPLE_TEAM_ID`
5. 已安装的客户端下次启动（延迟 4s）自动检查该 Release 并后台下载；用户可在更新条点「重启安装」。

`workflow_dispatch` 也能手动触发（用于验证，不打 tag 时 version 取 package.json）。

## 四、验证 updater（不发真包）

- 单测：`ts/apps/desktop/__tests__/updater.spec.ts`（未打包禁用路径 + force 模式下事件转发/进度取整）。
- 本机联调：设 `OPENINTJ_FORCE_UPDATER=1` 启动 dev，可走真实 `checkForUpdates`（需可达的 Release 源）。

## 五、已知手动缺口（发布前需人工处理）

- **品牌图标**：`ts/apps/desktop/resources/` 目前只有 `.gitkeep`。放 `icon.png`（≥512²，建议 1024²）后 electron-builder
  会自动派生 Win `.ico` / mac `.icns`；不放则用 Electron 默认图标（能出包，仅不带品牌）。
- **代码签名**：2026-07-19 `gh secret list` 为空；当前未签名 → Windows SmartScreen /
  macOS Gatekeeper 会告警。
  - Win：配置 `WIN_CSC_LINK` / `WIN_CSC_KEY_PASSWORD`（或另行接入 Azure Trusted Signing）。
  - mac：`hardenedRuntime` 与 `notarize` 已开，仍需上节列出的 Apple 证书及 notarization secrets。
- **Linux CI**：`electron-builder.yml` 声明了 Linux AppImage，但 `release.yml` 矩阵只跑 win/mac；Linux 目前只支持本地 `--linux` 打包。
- **首个正式 release 尚未切**：GitHub 当前无 Release；当前实现分支仍需提交、合入 `main`，
  将 desktop `version` 与 tag 对齐后才能进行签名端到端验证。Windows 未签名安装包
  `OpenINTJ-3.0.0-alpha.0-x64.exe` 已本机产出，macOS 跨平台打包仍需由 CI 验证。
