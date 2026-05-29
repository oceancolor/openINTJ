/**
 * Electron Main Process 入口（OpenINTJ Desktop v3.0）
 *
 * - createMainWindow()：BrowserWindow + preload bridge
 * - assembleDesktopAgent()：装配 4-plane + TAO/ReAct
 * - registerIpcHandlers()：按 RFC-004 注册 IPC 通道
 * - 应用生命周期 + 窗口管理
 */

import path from "node:path";
import { fileURLToPath } from "node:url";
import { BrowserWindow, app } from "electron";
import { loadOpenintjEnv, summarizeLlmEnv } from "@openintj/shared";
import { type DesktopAgent, assembleDesktopAgent } from "./agent.js";
import { registerIpcHandlers } from "./ipc-handlers.js";
import { type AutoUpdaterHandle, initAutoUpdater } from "./updater.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 1) 优先从仓库根 .env.local / .env 注入 env（不覆盖已存在键，PowerShell $env: 仍最优先）。
//    打包后 .env 不在 asar 里，靠用户 OS 级 env 注入；这里 silent 失败即可。
loadOpenintjEnv({ logPrefix: "[OpenINTJ desktop env]" });

// 2) 关掉 Chromium 后台探测的网络噪音（国内 GFW 经常让这些请求报 SSL 握手错）。
//    这些组件即使打开也不参与 OpenINTJ 业务，关闭安全无副作用。
//    必须在 app.whenReady() 前调用。
if (process.env["OPENINTJ_DESKTOP_KEEP_BG_NET"] !== "1") {
  app.commandLine.appendSwitch("disable-background-networking");
  app.commandLine.appendSwitch("disable-component-update");
  app.commandLine.appendSwitch("disable-domain-reliability");
  app.commandLine.appendSwitch(
    "disable-features",
    [
      "SafeBrowsing",
      "NetworkTimeServiceQuerying",
      "DialMediaRouteProvider",
      "MediaRouter",
      "OptimizationHints",
      "Translate",
      "InterestFeedContentSuggestions",
    ].join(","),
  );
}

let mainWindow: BrowserWindow | undefined;
let agent: DesktopAgent | undefined;
let updater: AutoUpdaterHandle | undefined;

const createWindow = (): BrowserWindow => {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    show: false,
    webPreferences: {
      contextIsolation: true,
      sandbox: false,
      // electron-vite 把 preload 产物固定打到 index.mjs（ESM）；
      // Electron 28+ 原生支持 .mjs preload，dev/prod 路径一致。
      preload: path.join(__dirname, "../preload/index.mjs"),
    },
  });

  win.once("ready-to-show", () => win.show());

  if (process.env["ELECTRON_RENDERER_URL"]) {
    void win.loadURL(process.env["ELECTRON_RENDERER_URL"]);
  } else {
    void win.loadFile(path.join(__dirname, "../renderer/index.html"));
  }

  return win;
};

void app.whenReady().then(async () => {
  const dataDir =
    process.env["OPENINTJ_DATA_DIR"] ?? path.join(app.getPath("userData"), "memory-store");
  const llmProvider =
    (process.env["LLM_PROVIDER"] as "ollama" | "hunyuan" | "mock" | undefined) ?? "mock";
  const envSummary = summarizeLlmEnv();
  console.log(`[OpenINTJ desktop] llm: ${envSummary.summary}`);
  if (llmProvider === "hunyuan" && !envSummary.hunyuan.hasKey) {
    console.warn(
      "[OpenINTJ desktop] LLM_PROVIDER=hunyuan 但未读到 HUNYUAN_API_KEY —— 客户端会自动降级 mock。\n" +
        "                  把 HUNYUAN_API_KEY 写进仓库根 .env / .env.local，或在启动前 $env:HUNYUAN_API_KEY=... 注入。",
    );
  }
  agent = await assembleDesktopAgent({
    llmProvider,
    dataDir,
  });
  console.log(
    `[OpenINTJ desktop] persistence=${agent.persistenceInfo.mode} dataDir=${agent.persistenceInfo.dataDir ?? "<in-memory>"}`,
  );
  mainWindow = createWindow();
  registerIpcHandlers(agent, mainWindow.webContents);

  // #6 自动更新：仅打包后真正生效；dev/测试为 no-op。
  updater = initAutoUpdater({ getWebContents: () => mainWindow?.webContents });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      mainWindow = createWindow();
      if (agent) registerIpcHandlers(agent, mainWindow.webContents);
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", async () => {
  updater?.dispose();
  if (agent) {
    try {
      await agent.close();
    } catch (e) {
      console.error("[OpenINTJ desktop] persist close failed:", e);
    }
  }
});
