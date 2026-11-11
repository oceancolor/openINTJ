/**
 * Electron Main Process 入口（OpenINTJ Desktop v0.3）
 *
 * - createMainWindow()：BrowserWindow + preload bridge
 * - assembleDesktopAgent()：装配 4-plane + TAO/ReAct
 * - registerIpcHandlers()：按 RFC-004 注册 IPC 通道
 * - 应用生命周期 + 窗口管理
 */

import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadOpenintjEnv, summarizeLlmEnv } from "@openintj/shared";
import { BrowserWindow, Menu, app, dialog, nativeTheme, safeStorage, session } from "electron";
import { CHROME_BG } from "../shared/chrome.js";
import {
  type AppConfig,
  DEFAULT_DESKTOP_MODEL_PROFILE_ID,
  MODEL_DEFAULTS_VERSION,
  type ModelProvider,
  TITLE_BAR_HEIGHT,
} from "../shared/ipc-protocol.js";
import { type DesktopAgent, assembleDesktopAgent } from "./agent.js";
import { type ConfigService, createConfigService } from "./config-store.js";
import { createCredentialStore } from "./credential-store.js";
import { type IpcDeps, type IpcRegistration, registerIpcHandlers } from "./ipc-handlers.js";
import { createModelRegistry } from "./model-registry.js";
import {
  keepDevWindowOnRestart,
  resolveRendererLoad,
  shouldFallbackToRendererFile,
} from "./renderer-boot.js";
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

const applySavedLlmEnv = (savedConfig: AppConfig): void => {
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
};

const resolveDesktopPaths = (savedConfig: AppConfig): { dataDir: string; workspaceDir: string } => {
  const dataDir =
    process.env["OPENINTJ_DATA_DIR"] ?? path.join(app.getPath("userData"), "memory-store");
  mkdirSync(dataDir, { recursive: true });
  const workspaceDir =
    process.env["OPENINTJ_WORKSPACE_DIR"] ??
    savedConfig.workspaceDir ??
    path.join(app.getPath("documents"), "OpenINTJ");
  mkdirSync(workspaceDir, { recursive: true });
  return { dataDir, workspaceDir };
};

const isOllamaEmbedUnavailable = (error: unknown): boolean => {
  if (!error || typeof error !== "object") return false;
  const code = "code" in error ? String((error as { code?: unknown }).code ?? "") : "";
  const provider =
    "provider" in error ? String((error as { provider?: unknown }).provider ?? "") : "";
  const message = error instanceof Error ? error.message : String(error);
  return (
    provider === "ollama" &&
    (code === "MODEL_PROVIDER_UNAVAILABLE" ||
      code === "MODEL_NOT_INSTALLED" ||
      /Embed provider 'ollama'/i.test(message))
  );
};

const isEmbeddingFingerprintBlock = (error: unknown): boolean => {
  const code =
    error && typeof error === "object" && "code" in error
      ? String((error as { code?: unknown }).code ?? "")
      : "";
  const message = error instanceof Error ? error.message : String(error);
  return (
    code === "EMBEDDING_FINGERPRINT_MISSING" ||
    code === "EMBEDDING_FINGERPRINT_MISMATCH" ||
    /EMBEDDING_FINGERPRINT_/i.test(message)
  );
};

const assembleFromSavedConfig = async (
  savedConfig: AppConfig,
  extras: { persistMemory?: boolean } = {},
): Promise<DesktopAgent> => {
  applySavedLlmEnv(savedConfig);
  const { dataDir, workspaceDir } = resolveDesktopPaths(savedConfig);
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
  return assembleDesktopAgent({
    llmProvider,
    ...(embedProvider ? { embedProvider } : {}),
    dataDir,
    workspaceDir,
    ...(extras.persistMemory ? { persistenceMode: "memory" } : {}),
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
    // RFC-013: desktop is the host that can actually ask a human — write/execute
    // tool calls require approval (or a capability grant) unless opted out.
    ...(process.env["OPENINTJ_APPROVAL"] === "0"
      ? {}
      : { approvalRisks: ["write", "execute"] as ("write" | "execute")[] }),
  });
};

/** 本机 Ollama 未开时不要让窗口起不来。不改磁盘配置；指纹冲突则本轮改内存库。 */
const assembleFromSavedConfigWithFallback = async (
  savedConfig: AppConfig,
): Promise<DesktopAgent> => {
  const withAutoEmbed = (persistMemory: boolean): Promise<DesktopAgent> => {
    delete process.env["EMBED_PROVIDER"];
    delete process.env["EMBEDDING_PROVIDER"];
    return assembleFromSavedConfig(
      { ...savedConfig, embedProvider: "auto" },
      persistMemory ? { persistMemory: true } : {},
    );
  };
  try {
    return await assembleFromSavedConfig(savedConfig);
  } catch (error) {
    if (isOllamaEmbedUnavailable(error)) {
      console.warn(
        "[OpenINTJ desktop] Ollama embedding unavailable; falling back to auto/simple. Start Ollama or switch Embed in Settings.",
      );
      try {
        return await withAutoEmbed(false);
      } catch (inner) {
        if (!isEmbeddingFingerprintBlock(inner)) throw inner;
        console.warn(
          "[OpenINTJ desktop] disk memory store fingerprint blocks simple embed; opening with in-memory store.",
        );
        return withAutoEmbed(true);
      }
    }
    if (isEmbeddingFingerprintBlock(error)) {
      console.warn(
        "[OpenINTJ desktop] embedding fingerprint blocked persistence; opening with in-memory store.",
      );
      return withAutoEmbed(true);
    }
    throw error;
  }
};

const rendererFile = (): string => path.join(__dirname, "../renderer/index.html");

const loadRenderer = (win: BrowserWindow): void => {
  const target = resolveRendererLoad({
    rendererUrl: process.env["ELECTRON_RENDERER_URL"],
    rendererFile: rendererFile(),
  });
  if (target.kind === "url") {
    console.log(`[OpenINTJ desktop] loading renderer ${target.url}`);
    void win.loadURL(target.url);
    return;
  }
  console.log(`[OpenINTJ desktop] loading renderer file ${target.file}`);
  void win.loadFile(target.file);
};

const createWindow = (): BrowserWindow => {
  const iconPath = path.join(__dirname, "../../resources/icon.png");
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: `OpenINTJ v${app.getVersion()}`,
    backgroundColor: CHROME_BG,
    show: false,
    ...(process.platform === "darwin"
      ? { titleBarStyle: "hiddenInset" as const }
      : process.platform === "win32"
        ? {
            titleBarStyle: "hidden" as const,
            titleBarOverlay: {
              color: CHROME_BG,
              symbolColor: "#18181b",
              height: TITLE_BAR_HEIGHT,
            },
          }
        : { titleBarStyle: "hidden" as const }),
    ...(existsSync(iconPath) ? { icon: iconPath } : {}),
    webPreferences: {
      contextIsolation: true,
      sandbox: false,
      // electron-vite 把 preload 产物固定打到 index.mjs（ESM）；
      // Electron 28+ 原生支持 .mjs preload，dev/prod 路径一致。
      preload: path.join(__dirname, "../preload/index.mjs"),
    },
  });

  win.once("ready-to-show", () => win.show());
  win.webContents.on(
    "did-fail-load",
    (_event, errorCode, errorDescription, validatedURL, isMainFrame) => {
      console.error(
        `[OpenINTJ desktop] renderer failed to load (${errorCode}): ${errorDescription} url=${validatedURL}`,
      );
      if (
        shouldFallbackToRendererFile({
          isMainFrame,
          errorCode,
          failedUrl: validatedURL,
        }) &&
        existsSync(rendererFile())
      ) {
        console.warn("[OpenINTJ desktop] falling back to built renderer/index.html");
        void win.loadFile(rendererFile());
      }
    },
  );

  return win;
};

/**
 * RFC-019 microphone access for voice input: grant `media` only to the main
 * window's own renderer; every other permission (and any embedded IDE view)
 * is denied. Installing a handler replaces Electron's defaults, so the
 * deny-everything-else branch is explicit.
 */
const installPermissionPolicy = (): void => {
  const isOwnWindow = (webContents: Electron.WebContents | null): boolean =>
    webContents !== null && mainWindow !== undefined && webContents === mainWindow.webContents;
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    callback(isOwnWindow(webContents) && permission === "media");
  });
  session.defaultSession.setPermissionCheckHandler(
    (webContents, permission) => isOwnWindow(webContents) && permission === "media",
  );
};

void app
  .whenReady()
  .then(async () => {
    installPermissionPolicy();
    nativeTheme.themeSource = "light";
    // 持久化的应用配置（用户在 UI 里改的偏好）。优先级：显式 env > 已存配置 > 默认。
    config = createConfigService(path.join(app.getPath("userData"), "config.json"));
    const credentials = createCredentialStore(
      path.join(app.getPath("userData"), "model-credentials.json"),
      safeStorage,
    );
    const modelRegistry = createModelRegistry({ config, credentials });
    let savedConfig = config.get();
    const needsDefaultBump = savedConfig.modelDefaultsVersion !== MODEL_DEFAULTS_VERSION;
    if (needsDefaultBump) {
      savedConfig = config.update({
        activeModelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
        llmProvider: "hunyuan",
        modelDefaultsVersion: MODEL_DEFAULTS_VERSION,
        // 已下线的旧混元配置档若曾被持久化，一并移除（凭据走新 id 回退链读取）。
        ...(savedConfig.modelProfiles
          ? {
              modelProfiles: savedConfig.modelProfiles.filter(
                (p) => p.id !== "hunyuan-hy3" && p.id !== "hunyuan-hy4-preview",
              ),
            }
          : {}),
      });
    }
    if (!process.env["HUNYUAN_API_KEY"]) {
      try {
        const storedKey =
          credentials.get(DEFAULT_DESKTOP_MODEL_PROFILE_ID) ??
          credentials.get("hunyuan-hy4-preview") ??
          credentials.get("hunyuan-hy3");
        if (storedKey) process.env["HUNYUAN_API_KEY"] = storedKey;
      } catch (error) {
        console.warn("[OpenINTJ desktop] 无法读取安全存储中的混元凭据", error);
      }
    }

    const { dataDir, workspaceDir } = resolveDesktopPaths(savedConfig);
    workbench = createWorkbenchStore({
      dbPath:
        process.env["OPENINTJ_DESKTOP_NO_PERSIST"] === "1"
          ? ":memory:"
          : path.join(app.getPath("userData"), "workbench.sqlite"),
      defaultWorkspaceRoot: workspaceDir,
      ...(process.env["OPENINTJ_DESKTOP_NO_PERSIST"] === "1" ? {} : { defaultDataDir: dataDir }),
    });
    if (needsDefaultBump) {
      for (const conversation of workbench.bootstrap().conversations) {
        if (
          conversation.modelProfileId === "hunyuan-hy3" ||
          conversation.modelProfileId === "hunyuan-hy4-preview"
        ) {
          workbench.updateConversation(conversation.id, {
            modelProfileId: DEFAULT_DESKTOP_MODEL_PROFILE_ID,
          });
        }
      }
    }
    const llmProvider =
      (process.env["LLM_PROVIDER"] as ModelProvider | undefined) ??
      savedConfig.llmProvider ??
      "hunyuan";
    if (
      process.env["HUNYUAN_MODEL"] === "hy4-preview" ||
      process.env["HUNYUAN_MODEL"] === "hy3" ||
      process.env["HUNYUAN_MODEL"] === "hy3-preview" ||
      process.env["HUNYUAN_MODEL"] === "hunyuan-turbos-latest"
    ) {
      console.warn(
        `[OpenINTJ desktop] 已将旧 Hunyuan 模型 '${process.env["HUNYUAN_MODEL"]}' 迁移为 'hy4'`,
      );
      process.env["HUNYUAN_MODEL"] = "hy4";
    }
    const envSummary = summarizeLlmEnv();
    console.log(`[OpenINTJ desktop] llm: ${envSummary.summary}`);
    if (llmProvider === "hunyuan" && !envSummary.hunyuan.hasKey) {
      console.warn(
        "[OpenINTJ desktop] LLM_PROVIDER=hunyuan 但未读到 HUNYUAN_API_KEY —— strict 模式将报错。\n" +
          "                  把 HUNYUAN_API_KEY 写进仓库根 .env / .env.local，或用 LLM_PROVIDER=auto。",
      );
    }

    agent = await assembleFromSavedConfigWithFallback(savedConfig);
    console.log(
      `[OpenINTJ desktop] persistence=${agent.persistenceInfo.mode} dataDir=${agent.persistenceInfo.dataDir ?? "<in-memory>"}`,
    );
    mainWindow = createWindow();

    // 弹系统目录选择框，选定后即时持久化并热重装 agent。
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
    const reloadAgent = async (): Promise<void> => {
      if (!config) throw new Error("config_unavailable");
      ipcRegistration?.unregister();
      const previous = agent;
      agent = undefined;
      agentClosePromise = undefined;
      if (previous) {
        try {
          await previous.close();
        } catch (error) {
          console.warn("[OpenINTJ desktop] previous agent close failed during reload", error);
        }
      }
      const next = await assembleFromSavedConfigWithFallback(config.get());
      agent = next;
      if (mainWindow && !mainWindow.isDestroyed()) {
        ipcRegistration = registerIpcHandlers(agent, mainWindow.webContents, undefined, ipcDeps);
        // RFC-015: restart the scheduler for the fresh agent (old one stopped in close()).
        agent.scheduler.start();
        // RFC-016: initialize async connectors (MCP) for the fresh agent.
        void agent.initConnectors();
      }
      console.log(
        `[OpenINTJ desktop] agent reloaded persistence=${agent.persistenceInfo.mode} dataDir=${agent.persistenceInfo.dataDir ?? "<in-memory>"}`,
      );
    };
    const restart = async (): Promise<void> => {
      if (
        keepDevWindowOnRestart({
          isPackaged: app.isPackaged,
          rendererUrl: process.env["ELECTRON_RENDERER_URL"],
        })
      ) {
        // electron-vite 把 Vite URL 注进当前进程。relaunch 会丢掉开发服务器，窗口变白屏。
        await reloadAgent();
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.reloadIgnoringCache();
        }
        return;
      }
      ipcRegistration?.unregister();
      await closeApplicationState();
      app.relaunch();
      app.quit();
    };
    // Windows/Linux：系统标题栏与菜单合成 renderer 一行；macOS 菜单仍在系统栏。
    if (process.platform === "darwin") {
      Menu.setApplicationMenu(
        Menu.buildFromTemplate([
          { role: "appMenu" },
          {
            label: "File",
            submenu: [
              {
                label: "重启",
                click: () => {
                  void restart();
                },
              },
              { type: "separator" },
              { role: "quit", label: "退出" },
            ],
          },
          { role: "editMenu" },
          { role: "viewMenu" },
          { role: "windowMenu" },
        ]),
      );
    } else {
      Menu.setApplicationMenu(null);
    }
    const ipcDeps: IpcDeps = {
      pickDirectory,
      restart,
      reloadAgent,
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
    // RFC-015: start the scheduler (catch-up pass fires anything missed offline).
    agent.scheduler.start();
    // RFC-016: initialize async connectors (MCP) in the background.
    void agent.initConnectors();
    loadRenderer(mainWindow);

    // #6 自动更新：仅打包后真正生效；dev/测试为 no-op。
    updater = initAutoUpdater({ getWebContents: () => mainWindow?.webContents });

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        mainWindow = createWindow();
        if (agent) {
          ipcRegistration?.unregister();
          ipcRegistration = registerIpcHandlers(agent, mainWindow.webContents, undefined, ipcDeps);
        }
        loadRenderer(mainWindow);
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
