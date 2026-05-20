/**
 * Playwright 配置 —— OpenINTJ Desktop E2E
 *
 * 本配置只跑一个 project：用 Playwright 的 `_electron.launch` 启动真实
 * Electron 主进程，加载 `out/main/index.js`，对真实 renderer 跑断言。
 *
 * 启动门槛（避免污染默认 CI 路径）：
 *   - 必须 `OPENINTJ_PLAYWRIGHT=1` 才会执行测试；否则 testignore + skip
 *   - 必须先 `pnpm --filter @openintj/desktop run build` 让 `out/` 存在
 *
 * 跑法：
 *   $env:OPENINTJ_PLAYWRIGHT="1"
 *   pnpm --filter @openintj/desktop run e2e
 *   Remove-Item env:OPENINTJ_PLAYWRIGHT
 */
import { defineConfig } from "@playwright/test";

const enabled = process.env["OPENINTJ_PLAYWRIGHT"] === "1";

export default defineConfig({
  testDir: "./tests",
  // 单 worker：Electron 主进程是单例资源，并发跑会撞窗口和 IPC handler 注册
  workers: 1,
  fullyParallel: false,
  // CI 模式严格化
  forbidOnly: !!process.env["CI"],
  retries: process.env["CI"] ? 1 : 0,
  reporter: process.env["CI"] ? [["github"], ["list"]] : "list",
  // 整体超时给得宽：Electron 冷启动 + LanceDB skip 之外的小启动
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  use: {
    trace: "retain-on-failure",
  },
  // 未设置开关时不抓任何文件，确保默认 CI 路径里 playwright runner 直接 noop
  testIgnore: enabled ? [] : ["**/*"],
});
