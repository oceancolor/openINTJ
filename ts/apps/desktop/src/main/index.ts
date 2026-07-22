/**
 * Electron Main Process 入口（OpenINTJ Desktop v0.3）
 *
 * - createMainWindow()：BrowserWindow + preload bridge
 * - assembleDesktopAgent()：装配 4-plane + TAO/ReAct
 * - registerIpcHandlers()：按 RFC-004 注册 IPC 通道
 * - 应用生命周期 + 窗口管理
 */

import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadOpenintjEnv, summarizeLlmEnv } from "@openintj/shared";
import { BrowserWindow, app, dialog, safeStorage } from "electron";
import { DEFAULT_DESKTOP_MODEL_PROFILE_ID, type ModelProvider } from "../shared/ipc-protocol.js";
import { type DesktopAgent, assembleDesktopAgent } from "./agent.js";
import { type ConfigService, createConfigService } from "./config-store.js";
import { createCredentialStore } from "./credential-store.js";
import { type IpcDeps, type IpcRegistration, registerIpcHandlers } from "./ipc-handlers.js";
import { createModelRegistry } from "./model-registry.js";
import { type AutoUpdaterHandle, initAutoUpdater } from "./updater.js";
import { type WorkbenchStore, createWorkbenchStore } from "./workbench-store.js";

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
let config: ConfigService | undefined;
let ipcRegistration: IpcRegistration | undefined;
let agentClosePromise: Promise<void> | undefined;
let workbench: WorkbenchStore | undefined;
let workbenchClosed = false;

const closeAgent = (): Promise<void> => {
  if (!agent) return Promise.resolve();
  agentClosePromise ??= agent.close();
  return agentClosePromise;
};

const closeApplicationState = async (): Promise<void> => {
  await closeAgent();
  if (workbench && !workbenchClosed) {
    workbench.close();
    workbenchClosed = true;
  }
};

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

void app
  .whenReady()
  .then(async () => {
    // 持久化的应用配置（用户在 UI 里改的偏好）。优先级：显式 env > 已存配置 > 默认。
    config = createConfigService(path.join(app.getPath("userData"), "config.json"));
    const credentials = createCredentialStore(
      path.join(app.getPath("userData"), "model-credentials.json"),
      safeStorage,
    );
    const modelRegistry = createModelRegistry({ config, credentials });
    let savedConfig = config.get();
    if (savedConfig.modelDefaultsVersion !== 1) {
      savedConfig = config.update({
        activeModelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
        llmProvider: "hunyuan",
        modelDefaultsVersion: 1,
      });
    }
    if (!process.env["HUNYUAN_API_KEY"]) {
      try {
        const storedKey = credentials.get(DEFAULT_DESKTOP_MODEL_PROFILE_ID);
        if (storedKey) process.env["HUNYUAN_API_KEY"] = storedKey;
      } catch (error) {
        console.warn("[OpenINTJ desktop] 无法读取安全存储中的 Hy3 凭据", error);
      }
    }

    const dataDir =
      process.env["OPENINTJ_DATA_DIR"] ?? path.join(app.getPath("userData"), "memory-store");
    mkdirSync(dataDir, { recursive: true });
    // 工作区根（read_file / write_file 沙箱根）：默认 documents 下的 OpenINTJ 目录，避免落到随机 cwd。
    const workspaceDir =
      process.env["OPENINTJ_WORKSPACE_DIR"] ??
      savedConfig.workspaceDir ??
      path.join(app.getPath("documents"), "OpenINTJ");
    mkdirSync(workspaceDir, { recursive: true });
    workbench = createWorkbenchStore({
      dbPath:
        process.env["OPENINTJ_DESKTOP_NO_PERSIST"] === "1"
          ? ":memory:"
          : path.join(app.getPath("userData"), "workbench.sqlite"),
      defaultWorkspaceRoot: workspaceDir,
      ...(process.env["OPENINTJ_DESKTOP_NO_PERSIST"] === "1" ? {} : { defaultDataDir: dataDir }),
    });
    const llmProvider =
      (process.env["LLM_PROVIDER"] as ModelProvider | undefined) ??
      savedConfig.llmProvider ??
      "hunyuan";
    const embedProvider =
      ((process.env["EMBEDDING_PROVIDER"] ?? process.env["EMBED_PROVIDER"]) as
        | "auto"
        | "ollama"
        | "mock"
        | undefined) ?? savedConfig.embedProvider;
    if (
      process.env["HUNYUAN_MODEL"] === "hy3-preview" ||
      process.env["HUNYUAN_MODEL"] === "hunyuan-turbos-latest"
    ) {
      console.warn(
        `[OpenINTJ desktop] 已将旧 Hunyuan 模型 '${process.env["HUNYUAN_MODEL"]}' 迁移为 'hy3'`,
      );
      process.env["HUNYUAN_MODEL"] = "hy3";
    }
    const envSummary = summarizeLlmEnv();
    console.log(`[OpenINTJ desktop] llm: ${envSummary.summary}`);
    if (llmProvider === "hunyuan" && !envSummary.hunyuan.hasKey) {
      console.warn(
        "[OpenINTJ desktop] LLM_PROVIDER=hunyuan 但未读到 HUNYUAN_API_KEY —— strict 模式将报错。\n" +
          "                  把 HUNYUAN_API_KEY 写进仓库根 .env / .env.local，或用 LLM_PROVIDER=auto。",
      );
    }
    if (savedConfig.ollamaBaseUrl) process.env["OLLAMA_BASE_URL"] = savedConfig.ollamaBaseUrl;
    if (savedConfig.ollamaModel) process.env["OLLAMA_MODEL"] = savedConfig.ollamaModel;
    if (savedConfig.ollamaEmbedModel)
      process.env["OLLAMA_EMBED_MODEL"] = savedConfig.ollamaEmbedModel;
    if (
      savedConfig.embedProvider &&
      !process.env["EMBEDDING_PROVIDER"] &&
      !process.env["EMBED_PROVIDER"]
    ) {
      process.env["EMBED_PROVIDER"] = savedConfig.embedProvider;
    }

    agent = await assembleDesktopAgent({
      llmProvider,
      ...(embedProvider ? { embedProvider } : {}),
      dataDir,
      workspaceDir,
      ...(savedConfig.retrievalMode ? { retrievalMode: savedConfig.retrievalMode } : {}),
      ...(savedConfig.enableCommands !== undefined
        ? { enableCommands: savedConfig.enableCommands }
        : {}),
      ...(savedConfig.allowedCommands ? { allowedCommands: savedConfig.allowedCommands } : {}),
      ...(savedConfig.enableDormant !== undefined
        ? { enableDormant: savedConfig.enableDormant }
        : {}),
      ...(savedConfig.enablePersona !== undefined
        ? { enablePersona: savedConfig.enablePersona }
        : {}),
      ...(savedConfig.enableProductBehavior !== undefined
        ? { enableProductBehavior: savedConfig.enableProductBehavior }
        : {}),
      ...(savedConfig.enableSkills !== undefined ? { enableSkills: savedConfig.enableSkills } : {}),
      ...(savedConfig.enableSkillLearning !== undefined
        ? { enableSkillLearning: savedConfig.enableSkillLearning }
        : {}),
      ...(savedConfig.enableClassifier !== undefined
        ? { enableClassifier: savedConfig.enableClassifier }
        : {}),
      ...(savedConfig.enableTaskPool !== undefined
        ? { enableTaskPool: savedConfig.enableTaskPool }
        : {}),
    });
    console.log(
      `[OpenINTJ desktop] persistence=${agent.persistenceInfo.mode} dataDir=${agent.persistenceInfo.dataDir ?? "<in-memory>"}`,
    );
    mainWindow = createWindow();

    // 弹系统目录选择框，选定后即时持久化为新工作区根（下次启动生效）。
    const pickDirectory = async (): Promise<string | null> => {
      const res = await dialog.showOpenDialog(mainWindow!, {
        properties: ["openDirectory", "createDirectory"],
        title: "选择 OpenINTJ 工作区目录",
      });
      if (res.canceled || res.filePaths.length === 0) return null;
      const picked = res.filePaths[0]!;
      config?.update({ workspaceDir: picked });
      return picked;
    };
    const restart = async (): Promise<void> => {
      ipcRegistration?.unregister();
      await closeApplicationState();
      app.relaunch();
      app.quit();
    };
    const ipcDeps: IpcDeps = {
      pickDirectory,
      restart,
      credentials,
      modelRegistry,
      workbench,
      credentialEnv: {
        hunyuan: Boolean(process.env["HUNYUAN_API_KEY"]),
        kimi: Boolean(process.env["KIMI_API_KEY"] ?? process.env["MOONSHOT_API_KEY"]),
        minimax: Boolean(process.env["MINIMAX_API_KEY"]),
        glm: Boolean(process.env["GLM_API_KEY"] ?? process.env["ZAI_API_KEY"]),
      },
      ...(config ? { config } : {}),
    };
    ipcRegistration = registerIpcHandlers(agent, mainWindow.webContents, undefined, ipcDeps);

    // #6 自动更新：仅打包后真正生效；dev/测试为 no-op。
    updater = initAutoUpdater({ getWebContents: () => mainWindow?.webContents });

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        mainWindow = createWindow();
        if (agent) {
          ipcRegistration?.unregister();
          ipcRegistration = registerIpcHandlers(agent, mainWindow.webContents, undefined, ipcDeps);
        }
      }
    });
  })
  .catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error("[OpenINTJ desktop] startup failed:", error);
    dialog.showErrorBox("OpenINTJ 启动失败", message);
    app.quit();
  });

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", async () => {
  updater?.dispose();
  ipcRegistration?.unregister();
  try {
    await closeApplicationState();
  } catch (e) {
    console.error("[OpenINTJ desktop] persist close failed:", e);
  }
});
