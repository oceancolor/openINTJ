/**
 * 自动更新（electron-updater, #6）。
 *
 * 设计要点：
 *  - 仅在**打包后**（app.isPackaged）启用；dev/测试环境直接 no-op，避免找不到 app-update.yml 报错。
 *    可用 OPENINTJ_FORCE_UPDATER=1 在非打包环境强制开启（调试用）。
 *  - 完全防御式：electron-updater 任意异常（无发布源 / 网络受限 / 签名缺失）只 console + 推 "error"/"disabled"
 *    事件给 renderer，绝不抛进主进程生命周期。
 *  - 通过 IPC.EVT_UPDATE 把状态流式推给 renderer；UPDATE_CHECK / UPDATE_INSTALL 供 UI 主动触发。
 *
 * 发布源：electron-builder.yml 的 `publish: github` 会在打包时生成 app-update.yml，
 * autoUpdater 据此拉取 release。owner/repo 由 package.json 的 repository 字段或环境推断。
 */

import { type IpcMain, type WebContents, app, ipcMain } from "electron";
import { IPC, type UpdateEvent } from "../shared/ipc-protocol.js";

export interface AutoUpdaterDeps {
  /** 拿当前可见窗口的 webContents，用于推送 EVT_UPDATE。 */
  getWebContents: () => WebContents | undefined;
  /** 注入 ipcMain（测试可替换）。 */
  ipc?: IpcMain;
  /** 强制开启（绕过 app.isPackaged 判断）。 */
  force?: boolean;
}

export interface AutoUpdaterHandle {
  /** 主动触发一次检查（返回是否真正发起）。 */
  checkNow: () => Promise<boolean>;
  dispose: () => void;
}

// electron-updater 的最小结构子集（CJS 默认导出 { autoUpdater }）。
interface UpdaterLike {
  autoDownload: boolean;
  autoInstallOnAppQuit: boolean;
  logger: unknown;
  on(event: string, cb: (...args: unknown[]) => void): void;
  removeAllListeners(event?: string): void;
  checkForUpdates(): Promise<unknown>;
  quitAndInstall(isSilent?: boolean, isForceRunAfter?: boolean): void;
}

const isEnabled = (force?: boolean): boolean =>
  force === true || process.env["OPENINTJ_FORCE_UPDATER"] === "1" || app.isPackaged;

/**
 * 初始化自动更新。未启用时返回一个 no-op handle（并向 renderer 推 "disabled"）。
 */
export const initAutoUpdater = (deps: AutoUpdaterDeps): AutoUpdaterHandle => {
  const ipc = deps.ipc ?? ipcMain;

  const push = (evt: UpdateEvent): void => {
    try {
      deps.getWebContents()?.send(IPC.EVT_UPDATE, evt);
    } catch {
      // renderer 已销毁；忽略
    }
  };

  if (!isEnabled(deps.force)) {
    const handler = async (): Promise<{ ok: false; reason: string }> => {
      push({ status: "disabled", message: "未打包环境，自动更新已禁用" });
      return { ok: false, reason: "not_packaged" };
    };
    ipc.handle(IPC.UPDATE_CHECK, handler);
    ipc.handle(IPC.UPDATE_INSTALL, async () => ({ ok: false, reason: "not_packaged" }));
    return {
      checkNow: async () => false,
      dispose: () => {
        ipc.removeHandler(IPC.UPDATE_CHECK);
        ipc.removeHandler(IPC.UPDATE_INSTALL);
      },
    };
  }

  let updater: UpdaterLike | undefined;

  const ensureUpdater = async (): Promise<UpdaterLike | undefined> => {
    if (updater) return updater;
    try {
      const mod = (await import("electron-updater")) as unknown as {
        autoUpdater?: UpdaterLike;
        default?: { autoUpdater?: UpdaterLike };
      };
      const au = mod.autoUpdater ?? mod.default?.autoUpdater;
      if (!au) throw new Error("electron-updater: autoUpdater 不可用");
      au.autoDownload = true;
      au.autoInstallOnAppQuit = true;
      au.on("checking-for-update", () => push({ status: "checking" }));
      au.on("update-available", (info: unknown) =>
        push({ status: "available", version: versionOf(info) }),
      );
      au.on("update-not-available", () => push({ status: "not-available" }));
      au.on("download-progress", (p: unknown) =>
        push({ status: "downloading", percent: percentOf(p) }),
      );
      au.on("update-downloaded", (info: unknown) =>
        push({ status: "downloaded", version: versionOf(info) }),
      );
      au.on("error", (err: unknown) =>
        push({ status: "error", message: err instanceof Error ? err.message : String(err) }),
      );
      updater = au;
      return au;
    } catch (e) {
      console.error("[OpenINTJ updater] 初始化失败:", (e as Error).message);
      push({ status: "error", message: (e as Error).message });
      return undefined;
    }
  };

  const checkNow = async (): Promise<boolean> => {
    const au = await ensureUpdater();
    if (!au) return false;
    try {
      await au.checkForUpdates();
      return true;
    } catch (e) {
      console.error("[OpenINTJ updater] checkForUpdates 失败:", (e as Error).message);
      push({ status: "error", message: (e as Error).message });
      return false;
    }
  };

  ipc.handle(IPC.UPDATE_CHECK, async () => ({ ok: await checkNow() }));
  ipc.handle(IPC.UPDATE_INSTALL, async () => {
    const au = await ensureUpdater();
    if (!au) return { ok: false, reason: "updater_unavailable" };
    try {
      au.quitAndInstall(false, true);
      return { ok: true };
    } catch (e) {
      console.error("[OpenINTJ updater] quitAndInstall 失败:", (e as Error).message);
      return { ok: false, reason: (e as Error).message };
    }
  });

  // 启动后延迟首检，避开冷启动 IO 高峰。
  setTimeout(() => void checkNow(), 4_000);

  return {
    checkNow,
    dispose: () => {
      ipc.removeHandler(IPC.UPDATE_CHECK);
      ipc.removeHandler(IPC.UPDATE_INSTALL);
      updater?.removeAllListeners();
    },
  };
};

const versionOf = (info: unknown): string | undefined => {
  if (info && typeof info === "object" && "version" in info) {
    const v = (info as { version?: unknown }).version;
    return typeof v === "string" ? v : undefined;
  }
  return undefined;
};

const percentOf = (p: unknown): number | undefined => {
  if (p && typeof p === "object" && "percent" in p) {
    const v = (p as { percent?: unknown }).percent;
    return typeof v === "number" ? Math.round(v) : undefined;
  }
  return undefined;
};
