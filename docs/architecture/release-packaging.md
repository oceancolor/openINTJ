# 打包发布 & 自动更新（#6）

> 状态：**应用图标为产品名 openINTJ**（`resources/icon.png`，1024×1024 像素，供 electron-builder 派生 ico/icns）；Release CI 在缺证书时产出未签名 Win/mac/Linux 包
> （不再把空 `CSC_LINK` 当成文件路径）。**真正代码签名 / Apple 公证仍需仓库 secrets。**
> 桌面版本 `0.3.1`；未签名 GitHub Release `v0.3.1` 使用产品名 openINTJ 作为应用图标。

桌面端（`ts/apps/desktop`）用 **electron-builder** 出安装包、**electron-updater** 从 GitHub Release 拉更新。

## 一、构成

| 件 | 路径 | 作用 |
|---|---|---|
| 打包配置 | `ts/apps/desktop/electron-builder.yml` | Win `nsis` / macOS `dmg` / Linux `AppImage`；`publish: github (oceancolor/openINTJ)`；原生模块 `asarUnpack` |
| 自动更新 | `ts/apps/desktop/src/main/updater.ts` | `initAutoUpdater`：仅打包后启用（`app.isPackaged` / `OPENINTJ_FORCE_UPDATER=1`），防御式，异常只推 renderer 不崩主进程 |
| 主进程接线 | `ts/apps/desktop/src/main/index.ts` | 启动装 updater，退出 `dispose` |
| IPC 契约 | `ts/apps/desktop/src/shared/ipc-protocol.ts` | `UPDATE_CHECK` / `UPDATE_INSTALL` 调用；`EVT_UPDATE` 流式状态；`UpdateEventSchema` |
| 更新条 UI | `ts/apps/desktop/src/renderer/components/UpdateBanner.tsx` | 有可用更新/下载中/已就绪/出错时显示；已下载给「重启安装」（挂在 `App.tsx`） |
| CI 发布 | `.github/workflows/release.yml` | 打 `v*` tag → win/mac/linux 构建 → `electron-builder --publish always` → GitHub Release |
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
   git tag v0.3.1
   git push origin v0.3.1
   ```
4. `release.yml` 触发：windows / macos / ubuntu 各自 `electron-builder --publish always`，用内置
   `GITHUB_TOKEN` 上传安装包 + `latest.yml` / `latest-mac.yml` / `latest-linux.yml`。
   证书与公证 secrets 见第五节；未配置则发未签名包。
5. 已安装的客户端下次启动（延迟 4s）自动检查该 Release 并后台下载；用户可在更新条点「重启安装」。

`workflow_dispatch` 也能手动触发（用于验证，不打 tag 时 version 取 package.json）。

## 四、验证 updater（不发真包）

- 单测：`ts/apps/desktop/__tests__/updater.spec.ts`（未打包禁用路径 + force 模式下事件转发/进度取整）。
- 本机联调：设 `OPENINTJ_FORCE_UPDATER=1` 启动 dev，可走真实 `checkForUpdates`（需可达的 Release 源）。

## 五、代码签名 secrets（可选；缺省则未签名发布）

仓库目前没有签名证书。未配置时 CI **仍应成功**，只是安装包未签名：Windows SmartScreen /
macOS Gatekeeper 会告警。要做签名/公证，在 GitHub repo Settings → Secrets 添加：

| Secret | 用途 |
|---|---|
| `WIN_CSC_LINK` | Windows 代码签名证书（`.p12`/`.pfx` 的 HTTPS URL 或 electron-builder 可读取的路径） |
| `WIN_CSC_KEY_PASSWORD` | 上述证书密码 |
| `MAC_CSC_LINK` | Developer ID Application 证书（`.p12`） |
| `MAC_CSC_KEY_PASSWORD` | 上述证书密码 |
| `APPLE_ID` | 用于公证的 Apple ID |
| `APPLE_APP_SPECIFIC_PASSWORD` | Apple 应用专用密码 |
| `APPLE_TEAM_ID` | Apple Team ID |

工作流**只有非空**时才导出 `CSC_LINK`；mac 也只有存在 `APPLE_ID` 时才打开 `notarize`。
不要把空字符串写进 `CSC_LINK`——electron-builder 会把它当成文件路径，
上次 `v0.3.0-alpha.0` 的 macOS job 即因此失败（`.../ts/apps/desktop not a file`）。

Windows 也可另行接入 Azure Trusted Signing，不走 `CSC_LINK`。

## 六、已知手动缺口

- **代码签名证书**：仍需人工把上表 secrets 配进 `oceancolor/openINTJ`。
- **`origin/main` 与实现分支无共同历史**：GitHub 默认 `main` 目前是 2026-04 的 Python 上传，
  不能开 PR。`v0.3.0` 未签名包已从 `rfc-005-007-implementation` 的 tag 发出。
  是否把实现分支设为默认、或重建 `main`，需仓库管理员决定。
- **应用图标**：`ts/apps/desktop/resources/icon.png` 为产品名 openINTJ（1024×1024 像素），由 `scripts/generate-icon.ps1` 绘制。
- **Linux CI**：`release.yml` 矩阵含 ubuntu AppImage。
