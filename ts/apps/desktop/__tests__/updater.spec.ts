import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IPC, type UpdateEvent } from "../src/shared/ipc-protocol.js";

vi.mock("electron", () => ({
  app: { isPackaged: false },
  ipcMain: { handle: vi.fn(), removeHandler: vi.fn() },
}));

// 受控的假 autoUpdater（vi.hoisted 以便 vi.mock 工厂内引用）。
const fake = vi.hoisted(() => {
  const listeners = new Map<string, (...a: unknown[]) => void>();
  return {
    listeners,
    autoUpdater: {
      autoDownload: false,
      autoInstallOnAppQuit: false,
      logger: null as unknown,
      on(event: string, cb: (...a: unknown[]) => void) {
        listeners.set(event, cb);
      },
      removeAllListeners() {
        listeners.clear();
      },
      checkForUpdates: vi.fn(async () => ({})),
      quitAndInstall: vi.fn(),
    },
  };
});

vi.mock("electron-updater", () => ({ autoUpdater: fake.autoUpdater }));

import { initAutoUpdater } from "../src/main/updater.js";

interface SentEvent {
  ch: string;
  payload: unknown;
}

const makeIpc = () => {
  const handlers = new Map<string, (...a: unknown[]) => unknown>();
  return {
    handlers,
    api: {
      handle: (ch: string, cb: (...a: unknown[]) => unknown) => handlers.set(ch, cb),
      removeHandler: (ch: string) => handlers.delete(ch),
    },
  };
};

const makeWc = (sink: SentEvent[]) => ({
  send: (ch: string, payload: unknown) => sink.push({ ch, payload }),
});

describe("initAutoUpdater (未打包 → 禁用)", () => {
  it("注册 no-op handler，UPDATE_CHECK 返回 not_packaged 并推 disabled", async () => {
    const { handlers, api } = makeIpc();
    const sent: SentEvent[] = [];
    const h = initAutoUpdater({
      getWebContents: () => makeWc(sent) as never,
      ipc: api as never,
    });

    expect(handlers.has(IPC.UPDATE_CHECK)).toBe(true);
    expect(handlers.has(IPC.UPDATE_INSTALL)).toBe(true);
    expect(await h.checkNow()).toBe(false);

    const res = (await handlers.get(IPC.UPDATE_CHECK)!({}, undefined)) as { ok: boolean };
    expect(res.ok).toBe(false);
    const pushed = sent.find((s) => s.ch === IPC.EVT_UPDATE);
    expect((pushed?.payload as UpdateEvent | undefined)?.status).toBe("disabled");

    h.dispose();
    expect(handlers.has(IPC.UPDATE_CHECK)).toBe(false);
  });
});

describe("initAutoUpdater (force + mocked electron-updater)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fake.listeners.clear();
    fake.autoUpdater.checkForUpdates.mockClear();
  });
  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it("checkNow 触发 checkForUpdates，事件转发为 EVT_UPDATE", async () => {
    const { api } = makeIpc();
    const sent: SentEvent[] = [];
    const h = initAutoUpdater({
      getWebContents: () => makeWc(sent) as never,
      ipc: api as never,
      force: true,
    });

    expect(await h.checkNow()).toBe(true);
    expect(fake.autoUpdater.checkForUpdates).toHaveBeenCalledTimes(1);
    expect(fake.autoUpdater.autoDownload).toBe(true);

    // 模拟 electron-updater 触发各阶段事件，验证被转发。
    fake.listeners.get("update-available")?.({ version: "9.9.9" });
    fake.listeners.get("download-progress")?.({ percent: 42.7 });
    fake.listeners.get("update-downloaded")?.({ version: "9.9.9" });

    const statuses = sent
      .filter((s) => s.ch === IPC.EVT_UPDATE)
      .map((s) => (s.payload as UpdateEvent).status);
    expect(statuses).toContain("available");
    expect(statuses).toContain("downloading");
    expect(statuses).toContain("downloaded");

    const progress = sent
      .map((s) => s.payload as UpdateEvent)
      .find((p) => p.status === "downloading");
    expect(progress?.percent).toBe(43);

    h.dispose();
  });
});
