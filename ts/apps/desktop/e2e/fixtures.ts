/**
 * Playwright 共享 fixture：启动 Electron 主进程并把第一个 BrowserWindow 给 test。
 *
 * 关键设计：
 *  - 每个 test 都拿到独立的 ElectronApplication + Page；teardown 自动关闭。
 *  - 默认 env：mock LLM + 关持久化 + 关 Dormant，让 boot 快且零依赖。
 *  - 单个 test 可通过 `test.use({ extraEnv: { OPENINTJ_DORMANT: "1" } })` 覆盖。
 */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { type ElectronApplication, type Page, _electron, test as base } from "@playwright/test";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const MAIN_ENTRY = path.resolve(__dirname, "..", "out", "main", "index.js");

const DEFAULT_ENV: NodeJS.ProcessEnv = {
  LLM_PROVIDER: "mock",
  OPENINTJ_DESKTOP_NO_PERSIST: "1",
  // 把 user agent / 任何 GUI auto-pop 关掉，避免 Electron 引导 chrome devtools / autofill
  ELECTRON_DISABLE_SECURITY_WARNINGS: "1",
};

export interface DesktopFixtures {
  extraEnv: NodeJS.ProcessEnv;
  electronApp: ElectronApplication;
  page: Page;
}

export const test = base.extend<DesktopFixtures>({
  // 默认空 env 增量；具体 spec 可以 `test.use({ extraEnv: { OPENINTJ_DORMANT: "1" } })`
  extraEnv: [{}, { option: true }],

  electronApp: async ({ extraEnv }, use) => {
    // Windows + Electron 33：保持参数最小，--no-sandbox / cwd 都会触发 launch 卡住。
    // _electron.launch.env 期望 Record<string,string>，process.env 含 undefined 值要先过滤。
    const mergedEnv: Record<string, string> = {};
    for (const src of [process.env, DEFAULT_ENV, extraEnv]) {
      for (const [k, v] of Object.entries(src)) {
        if (typeof v === "string") mergedEnv[k] = v;
      }
    }
    const app = await _electron.launch({
      args: [MAIN_ENTRY],
      env: mergedEnv,
      timeout: 45_000,
    });
    await use(app);
    await app.close();
  },

  page: async ({ electronApp }, use) => {
    const win = await electronApp.firstWindow();
    // 等待 React 挂载完成（renderer html 里 <div id="root">；body 里 header 出现就算 ready）
    await win.waitForLoadState("domcontentloaded");
    await win.waitForSelector("text=OpenINTJ", { timeout: 20_000 });
    await use(win);
  },
});

export { expect } from "@playwright/test";
