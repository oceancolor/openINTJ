# 打包发布 & 自动更新（#6）

> 状态：**实现已就绪**（electron-builder 配置 + electron-updater + CI 发布工作流 + renderer 更新条，全部有测试/接线）。
> 剩余为**运维手动项**（品牌图标、代码签名、首个正式 release），见文末「已知手动缺口」。

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

## 三、切一个正式 release（CI）

1. 确认要发布的 commit 已到 `main`（CI 绿）。
2. 版本号：`ts/apps/desktop/package.json` 的 `version` 决定安装包/`latest.yml` 版本号，需与 tag 对齐。
3. 打 tag 并推：
   ```bash
   git tag v3.0.0
   git push origin v3.0.0
   ```
4. `release.yml` 触发：windows-latest + macos-latest 各自 `electron-builder --publish always`，用内置 `GITHUB_TOKEN`
   上传安装包 + `latest.yml` / `latest-mac.yml` 到对应 GitHub Release。
5. 已安装的客户端下次启动（延迟 4s）自动检查该 Release 并后台下载；用户可在更新条点「重启安装」。

`workflow_dispatch` 也能手动触发（用于验证，不打 tag 时 version 取 package.json）。

## 四、验证 updater（不发真包）

- 单测：`ts/apps/desktop/__tests__/updater.spec.ts`（未打包禁用路径 + force 模式下事件转发/进度取整）。
- 本机联调：设 `OPENINTJ_FORCE_UPDATER=1` 启动 dev，可走真实 `checkForUpdates`（需可达的 Release 源）。

## 五、已知手动缺口（发布前需人工处理）

- **品牌图标**：`ts/apps/desktop/resources/` 目前只有 `.gitkeep`。放 `icon.png`（≥512²，建议 1024²）后 electron-builder
  会自动派生 Win `.ico` / mac `.icns`；不放则用 Electron 默认图标（能出包，仅不带品牌）。
- **代码签名**：当前未签名 → Windows SmartScreen / macOS Gatekeeper 会告警。
  - Win：配 `CSC_LINK` / `CSC_KEY_PASSWORD`（或 Azure Trusted Signing）。
  - mac：`hardenedRuntime: true` 已开，但仍需 Apple 证书 + `notarize`（配 `APPLE_ID` / `APPLE_APP_SPECIFIC_PASSWORD` / `APPLE_TEAM_ID`）。
- **Linux CI**：`electron-builder.yml` 声明了 Linux AppImage，但 `release.yml` 矩阵只跑 win/mac；Linux 目前只支持本地 `--linux` 打包。
- **首个正式 release 尚未切**：以上流程已就绪但未在真实 tag 上跑通端到端（原生模块跨平台打包可能需按 §一「ABI 对齐」微调）。
