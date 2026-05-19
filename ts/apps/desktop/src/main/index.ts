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
import { type DesktopAgent, assembleDesktopAgent } from "./agent.js";
import { registerIpcHandlers } from "./ipc-handlers.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow: BrowserWindow | undefined;
let agent: DesktopAgent | undefined;

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
      preload: path.join(__dirname, "../preload/index.js"),
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
  agent = await assembleDesktopAgent({
    llmProvider: (process.env["LLM_PROVIDER"] as "ollama" | "hunyuan" | "mock") ?? "mock",
    dataDir,
  });
  console.log(
    `[OpenINTJ desktop] persistence=${agent.persistenceInfo.mode} dataDir=${agent.persistenceInfo.dataDir ?? "<in-memory>"}`,
  );
  mainWindow = createWindow();
  registerIpcHandlers(agent, mainWindow.webContents);

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
  if (agent) {
    try {
      await agent.close();
    } catch (e) {
      console.error("[OpenINTJ desktop] persist close failed:", e);
    }
  }
});
