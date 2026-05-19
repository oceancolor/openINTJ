import { describe, expect, it, vi } from "vitest";
import { ChatRequestSchema, IPC, MemoryQueryRequestSchema } from "../src/shared/ipc-protocol.js";

// 我们 mock electron 模块（CI 中无法启动真实 Electron）
vi.mock("electron", () => ({
  ipcMain: {
    handle: vi.fn(),
    removeHandler: vi.fn(),
  },
  contextBridge: {},
  ipcRenderer: {},
  app: {},
  BrowserWindow: class {},
}));

import { assembleDesktopAgent } from "../src/main/agent.js";
import { registerIpcHandlers } from "../src/main/ipc-handlers.js";

describe("IPC handler registration", () => {
  it("registers all openintj.* channels", async () => {
    const handle = vi.fn();
    const removeHandler = vi.fn();
    const fakeIpc = {
      handle,
      removeHandler,
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    const reg = registerIpcHandlers(agent, undefined, fakeIpc);

    const calledChannels = handle.mock.calls.map((c) => c[0] as string);
    expect(calledChannels).toContain(IPC.PING);
    expect(calledChannels).toContain(IPC.STATUS);
    expect(calledChannels).toContain(IPC.CHAT);
    expect(calledChannels).toContain(IPC.MEMORY_QUERY);
    expect(calledChannels).toContain(IPC.AUDIT_RECENT);

    reg.unregister();
    expect(removeHandler.mock.calls.length).toBeGreaterThanOrEqual(5);
  });

  it("PING handler returns ok", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => {
        handlers.set(ch, cb);
      },
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const r = (await handlers.get(IPC.PING)?.({}, undefined)) as {
      ok: boolean;
      ts: number;
    };
    expect(r.ok).toBe(true);
    expect(typeof r.ts).toBe("number");
  });

  it("CHAT handler returns final answer in mock mode", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const result = (await handlers.get(IPC.CHAT)?.({}, { query: "你好" })) as {
      finalAnswer: string;
      status: string;
    };
    expect(typeof result.finalAnswer).toBe("string");
    expect(result.finalAnswer.length).toBeGreaterThan(0);
  });

  it("CHAT handler validates request body", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const r = (await handlers.get(IPC.CHAT)?.({}, { invalid: "x" })) as {
      error?: string;
    };
    expect(r.error).toBe("invalid_request");
  });

  it("STATUS returns 4-plane snapshot", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const r = (await handlers.get(IPC.STATUS)?.({}, undefined)) as Record<string, unknown>;
    expect(r).toHaveProperty("llm");
    expect(r).toHaveProperty("memory");
    expect(r).toHaveProperty("governance");
    expect(r).toHaveProperty("tools");
  });

  it("MEMORY_QUERY returns recent fragments after a chat", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    await handlers.get(IPC.CHAT)?.({}, { query: "我想喝咖啡" });
    const r = (await handlers.get(IPC.MEMORY_QUERY)?.({}, { topK: 5 })) as Array<{
      memoryType: string;
    }>;
    expect(Array.isArray(r)).toBe(true);
    expect(r.length).toBeGreaterThan(0);
  });

  it("schema validation: ChatRequest accepts query, MemoryQueryRequest defaults topK", () => {
    expect(ChatRequestSchema.safeParse({ query: "x" }).success).toBe(true);
    expect(ChatRequestSchema.safeParse({}).success).toBe(false);
    const m = MemoryQueryRequestSchema.parse({});
    expect(m.topK).toBe(10);
  });

  // ---------- RFC-003 装配（Phase 3.3.D） ----------

  it("MEMORY_QUERY ?mode=hybrid 走 BM25 + cosine 双路", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    agent.persistentStore.addLongTerm("machine learning frameworks pytorch", {
      taskTags: ["ml"],
      importance: 0.7,
    });
    agent.persistentStore.addLongTerm("cooking pasta recipe italian", {
      taskTags: ["food"],
      importance: 0.4,
    });
    await agent.persistentStore.awaitPendingWrites();

    const r = (await handlers.get(IPC.MEMORY_QUERY)?.(
      {},
      {
        query: "machine learning",
        topK: 5,
        mode: "hybrid",
      },
    )) as Array<{ fragmentId: string; content: string; score?: number }>;
    expect(Array.isArray(r)).toBe(true);
    expect(r.length).toBeGreaterThan(0);
    // hybrid 模式应该把 ML doc 排在 food doc 前
    expect(r[0]!.content).toContain("machine learning");
  });

  it("Dormant IPC 未启用时返回 dormant_not_enabled", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const r = (await handlers.get(IPC.DORMANT_MINE)?.({}, undefined)) as {
      error?: string;
    };
    expect(r.error).toBe("dormant_not_enabled");

    const r2 = (await handlers.get(IPC.DORMANT_LIST)?.({}, undefined)) as {
      error?: string;
    };
    expect(r2.error).toBe("dormant_not_enabled");
  });

  it("Dormant 完整 IPC 链路：mine → list → approve → persona", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({
      llmProvider: "mock",
      enableDormant: true,
      dormantOpts: {
        minerOpts: {
          ngramSize: 2,
          minFrequency: 2,
          minConfidence: 0.3,
          llmExtract: async (ngram) => ({
            description: `desktop 偏好: ${ngram}`,
            category: "preference",
          }),
        },
      },
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    // 喂事件
    for (let i = 0; i < 5; i++) {
      agent.dormant!.record("绿 茶 健 康", "user", { iter: i });
    }

    const mineRes = (await handlers.get(IPC.DORMANT_MINE)?.({}, undefined)) as {
      scannedEvents: number;
      proposals: Array<{ proposalId: string; status: string }>;
    };
    expect(mineRes.scannedEvents).toBe(5);
    expect(mineRes.proposals.length).toBeGreaterThan(0);

    const listRes = (await handlers.get(IPC.DORMANT_LIST)?.(
      {},
      {
        status: "pending",
      },
    )) as { total: number; proposals: Array<{ proposalId: string }> };
    expect(listRes.total).toBeGreaterThan(0);

    const first = listRes.proposals[0]!;
    const approveRes = (await handlers.get(IPC.DORMANT_APPROVE)?.(
      {},
      {
        proposalId: first.proposalId,
      },
    )) as { status: string };
    expect(approveRes.status).toBe("applied");

    const persona = (await handlers.get(IPC.DORMANT_PERSONA)?.({}, undefined)) as {
      preferences: Record<string, unknown>;
      meta: { version: number };
    };
    expect(Object.keys(persona.preferences).length).toBeGreaterThan(0);
    expect(persona.meta.version).toBe(1);
  });

  it("STATUS 暴露 retrievalMode 和 dormant 状态", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({
      llmProvider: "mock",
      retrievalMode: "hybrid",
      enableDormant: true,
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const r = (await handlers.get(IPC.STATUS)?.({}, undefined)) as {
      retrievalMode: string;
      dormant?: { enabled: boolean };
    };
    expect(r.retrievalMode).toBe("hybrid");
    expect(r.dormant?.enabled).toBe(true);
  });

  it("registerIpcHandlers 注册所有 Dormant channel", async () => {
    const handle = vi.fn();
    const removeHandler = vi.fn();
    const fakeIpc = { handle, removeHandler } as unknown as Parameters<
      typeof registerIpcHandlers
    >[2];

    const agent = await assembleDesktopAgent({
      llmProvider: "mock",
      enableDormant: true,
    });
    const reg = registerIpcHandlers(agent, undefined, fakeIpc);
    const channels = handle.mock.calls.map((c) => c[0] as string);
    expect(channels).toContain(IPC.DORMANT_MINE);
    expect(channels).toContain(IPC.DORMANT_LIST);
    expect(channels).toContain(IPC.DORMANT_APPROVE);
    expect(channels).toContain(IPC.DORMANT_REJECT);
    expect(channels).toContain(IPC.DORMANT_PERSONA);
    reg.unregister();
    // 5 个原本的 + 5 个 dormant = 至少 10
    expect(removeHandler.mock.calls.length).toBeGreaterThanOrEqual(10);
  });
});
