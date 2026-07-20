import { describe, expect, it, vi } from "vitest";
import {
  ChatRequestSchema,
  DormantDecisionResponseSchema,
  DormantListResponseSchema,
  DormantMineResponseSchema,
  DormantPersonaResponseSchema,
  IPC,
  MemoryQueryRequestSchema,
  SkillActiveResponseSchema,
  SkillDecisionResponseSchema,
  SkillDistillResponseSchema,
  SkillListResponseSchema,
  StatusResponseSchema,
} from "../src/shared/ipc-protocol.js";

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

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { assembleDesktopAgent } from "../src/main/agent.js";
import { createConfigService } from "../src/main/config-store.js";
import { type IpcDeps, registerIpcHandlers } from "../src/main/ipc-handlers.js";
import { createWorkbenchStore } from "../src/main/workbench-store.js";
import {
  WorkspaceInfoSchema,
  WorkspaceReadResponseSchema,
  WorkspaceWriteResponseSchema,
} from "../src/shared/ipc-protocol.js";

type Handlers = Map<string, (e: unknown, p?: unknown) => unknown>;
const makeFakeIpc = (handlers: Handlers): Parameters<typeof registerIpcHandlers>[2] =>
  ({
    handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
    removeHandler: () => {},
  }) as unknown as Parameters<typeof registerIpcHandlers>[2];

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

  it("CHAT persists a conversation turn and uses its selected model", async () => {
    const handlers: Handlers = new Map();
    const workbench = createWorkbenchStore({
      dbPath: ":memory:",
      defaultWorkspaceRoot: "F:\\workspace",
    });
    const conversation = workbench.bootstrap().conversations[0]!;
    workbench.updateConversation(conversation.id, { modelProfileId: "glm-5.2" });
    const selectedClient = {
      chat: vi.fn(async () => "FINAL: 来自 GLM"),
      visionChat: vi.fn(async () => "vision"),
      getStatus: () => ({
        provider: "glm",
        model: "glm-5.2",
        available: true,
        mode: "live" as const,
        status: "connected",
        visionSupported: false,
      }),
    };
    const modelRegistry = {
      list: () => [],
      resolve: vi.fn(async () => selectedClient),
      test: vi.fn(async () => ({ ok: true, provider: "glm", model: "glm-5.2" })),
      clear: vi.fn(),
    };
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers), {
      workbench,
      modelRegistry,
    });

    const result = (await handlers.get(IPC.CHAT)?.(
      {},
      { query: "继续", conversationId: conversation.id },
    )) as { finalAnswer: string; provider: string };

    expect(result.finalAnswer).toBe("来自 GLM");
    expect(result.provider).toBe("glm");
    expect(modelRegistry.resolve).toHaveBeenCalledWith("glm-5.2");
    expect(workbench.listMessages(conversation.id).map((entry) => entry.role)).toEqual([
      "user",
      "assistant",
    ]);
    await handlers.get(IPC.CHAT)?.({}, { query: "再继续", conversationId: conversation.id });
    const secondCallMessages = selectedClient.chat.mock.calls.at(-1)?.[0] as Array<{
      role: string;
      content: string;
    }>;
    expect(secondCallMessages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ role: "user", content: "继续" }),
        expect.objectContaining({ role: "assistant", content: "来自 GLM" }),
      ]),
    );
    workbench.close();
  });

  it("CHAT materializes a requested file when the model omits write_file", async () => {
    const handlers: Handlers = new Map();
    const root = mkdtempSync(join(tmpdir(), "openintj-artifact-"));
    const workbench = createWorkbenchStore({
      dbPath: ":memory:",
      defaultWorkspaceRoot: root,
    });
    const conversation = workbench.bootstrap().conversations[0]!;
    const selectedClient = {
      chat: vi.fn(async () => "FINAL: # 项目报告\n\n内容已完成。"),
      visionChat: vi.fn(async () => "vision"),
      getStatus: () => ({
        provider: "hunyuan",
        model: "hy3",
        available: true,
        mode: "live" as const,
        status: "connected",
        visionSupported: false,
      }),
    };
    const agent = await assembleDesktopAgent({
      llmProvider: "mock",
      workspaceDir: root,
    });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers), {
      workbench,
      modelRegistry: {
        list: () => [],
        resolve: vi.fn(async () => selectedClient),
        test: vi.fn(async () => ({ ok: true, provider: "hunyuan", model: "hy3" })),
        clear: vi.fn(),
      },
    });

    const result = (await handlers.get(IPC.CHAT)?.(
      {},
      {
        query: "请生成 report.md 文件",
        conversationId: conversation.id,
        modelProfileId: "hunyuan-hy3",
      },
    )) as { finalAnswer: string };

    expect(readFileSync(join(root, "report.md"), "utf8")).toContain("项目报告");
    expect(result.finalAnswer).toContain("已写入工作区：`report.md`");
    await agent.close();
    workbench.close();
    rmSync(root, { recursive: true, force: true });
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

    const agent = await assembleDesktopAgent({ llmProvider: "mock", embedProvider: "simple" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    await agent.persistentStore.addLongTermAsync("machine learning frameworks pytorch", {
      taskTags: ["ml"],
      importance: 0.7,
    });
    await agent.persistentStore.addLongTermAsync("cooking pasta recipe italian", {
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

  // ---------- Phase 3.5 #9.B：UI 装配前的 IPC 形状契约 ----------

  it("STATUS 返回值通过 StatusResponseSchema 全字段校验（含 dormant + retrievalMode + persistence）", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({
      llmProvider: "mock",
      enableDormant: true,
      retrievalMode: "hybrid",
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const raw = await handlers.get(IPC.STATUS)?.({}, undefined);
    const parsed = StatusResponseSchema.safeParse(raw);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.dormant?.enabled).toBe(true);
      expect(parsed.data.retrievalMode).toBe("hybrid");
      // mock 装配 memory 模式
      expect(parsed.data.persistence?.mode ?? "memory").toBe("memory");
    }
  });

  it("DORMANT_MINE 返回通过 DormantMineResponseSchema 校验", async () => {
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
          llmExtract: async (n) => ({ description: n, category: "preference" }),
        },
      },
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    for (let i = 0; i < 3; i++) agent.dormant!.record("绿 茶", "user");
    const raw = await handlers.get(IPC.DORMANT_MINE)?.({}, undefined);
    const parsed = DormantMineResponseSchema.safeParse(raw);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.scannedEvents).toBe(3);
      expect(parsed.data.proposals.length).toBeGreaterThan(0);
    }
  });

  it("DORMANT_LIST 默认（无 status）返回所有 proposals + schema 校验", async () => {
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
          llmExtract: async (n) => ({ description: n, category: "preference" }),
        },
      },
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    for (let i = 0; i < 3; i++) agent.dormant!.record("aa bb", "user");
    await handlers.get(IPC.DORMANT_MINE)?.({}, undefined);

    // 不传 status → all
    const raw = await handlers.get(IPC.DORMANT_LIST)?.({}, {});
    const parsed = DormantListResponseSchema.safeParse(raw);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.total).toBeGreaterThan(0);
      expect(parsed.data.proposals[0]!.patternDescription).toBeDefined();
      expect(parsed.data.proposals[0]!.confidence).toBeGreaterThan(0);
    }
  });

  it("DORMANT_REJECT 返回 status=rejected + 不污染 persona", async () => {
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
          llmExtract: async (n) => ({ description: n, category: "preference" }),
        },
      },
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    for (let i = 0; i < 3; i++) agent.dormant!.record("不 喜 欢", "user");
    const mineRes = (await handlers.get(IPC.DORMANT_MINE)?.({}, undefined)) as {
      proposals: Array<{ proposalId: string }>;
    };
    const id = mineRes.proposals[0]!.proposalId;

    const rejectRaw = await handlers.get(IPC.DORMANT_REJECT)?.({}, { proposalId: id });
    const rejectParsed = DormantDecisionResponseSchema.safeParse(rejectRaw);
    expect(rejectParsed.success).toBe(true);
    if (rejectParsed.success) {
      expect(rejectParsed.data.status).toBe("rejected");
      expect(rejectParsed.data.decidedAt).toBeTypeOf("number");
    }

    const persona = await handlers.get(IPC.DORMANT_PERSONA)?.({}, undefined);
    const personaParsed = DormantPersonaResponseSchema.safeParse(persona);
    expect(personaParsed.success).toBe(true);
    if (personaParsed.success) {
      expect(Object.keys(personaParsed.data.preferences).length).toBe(0);
      expect(personaParsed.data.meta.version).toBe(0);
    }
  });

  it("DORMANT_REVOKE：approve → revoke 后 persona 清空，proposal=revoked", async () => {
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
          llmExtract: async (n) => ({ description: `偏好: ${n}`, category: "preference" }),
        },
      },
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    for (let i = 0; i < 4; i++) agent.dormant!.record("绿 茶", "user");
    const mineRes = (await handlers.get(IPC.DORMANT_MINE)?.({}, undefined)) as {
      proposals: Array<{ proposalId: string }>;
    };
    const id = mineRes.proposals[0]!.proposalId;
    const approveRes = (await handlers.get(IPC.DORMANT_APPROVE)?.({}, { proposalId: id })) as {
      status: string;
    };
    expect(approveRes.status).toBe("applied");

    const revokeRes = (await handlers.get(IPC.DORMANT_REVOKE)?.({}, { proposalId: id })) as {
      status?: string;
    };
    expect(revokeRes.status).toBe("revoked");

    const persona = (await handlers.get(IPC.DORMANT_PERSONA)?.({}, undefined)) as {
      preferences: Record<string, unknown>;
    };
    expect(Object.keys(persona.preferences).length).toBe(0);
  });

  it("DORMANT_REVOKE：未 applied / 不存在 → not_found_or_not_applied；未启用 → dormant_not_enabled", async () => {
    const h1 = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const agent1 = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent1, undefined, {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => h1.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2]);
    const notEnabled = (await h1.get(IPC.DORMANT_REVOKE)?.({}, { proposalId: "x" })) as {
      error?: string;
    };
    expect(notEnabled.error).toBe("dormant_not_enabled");

    const h2 = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const agent2 = await assembleDesktopAgent({ llmProvider: "mock", enableDormant: true });
    registerIpcHandlers(agent2, undefined, {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => h2.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2]);
    const ghost = (await h2.get(IPC.DORMANT_REVOKE)?.({}, { proposalId: "ghost" })) as {
      error?: string;
    };
    expect(ghost.error).toBe("not_found_or_not_applied");
  });

  it("APPROVE / REJECT 不存在 proposalId 时返回 not_found_or_already_decided", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({
      llmProvider: "mock",
      enableDormant: true,
    });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const r1 = (await handlers.get(IPC.DORMANT_APPROVE)?.({}, { proposalId: "ghost" })) as {
      error?: string;
    };
    expect(r1.error).toBe("not_found_or_already_decided");

    const r2 = (await handlers.get(IPC.DORMANT_REJECT)?.({}, { proposalId: "ghost" })) as {
      error?: string;
    };
    expect(r2.error).toBe("not_found_or_already_decided");
  });

  it("DORMANT_PERSONA 未启用时返回 dormant_not_enabled", async () => {
    const handlers = new Map<string, (e: unknown, p?: unknown) => unknown>();
    const fakeIpc = {
      handle: (ch: string, cb: (e: unknown, p?: unknown) => unknown) => handlers.set(ch, cb),
      removeHandler: () => {},
    } as unknown as Parameters<typeof registerIpcHandlers>[2];

    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, fakeIpc);

    const r = (await handlers.get(IPC.DORMANT_PERSONA)?.({}, undefined)) as {
      error?: string;
    };
    expect(r.error).toBe("dormant_not_enabled");
  });

  // ---------- RFC-004 §8 工作区系统能力面 ----------

  it("WORKSPACE_INFO 暴露沙箱根 + 命令策略，通过 schema 校验", async () => {
    const root = mkdtempSync(join(tmpdir(), "ointj-ws-info-"));
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock", workspaceDir: root });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers));

    const raw = await handlers.get(IPC.WORKSPACE_INFO)?.({}, undefined);
    const parsed = WorkspaceInfoSchema.safeParse(raw);
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.root).toBe(root);
      expect(parsed.data.enableCommands).toBe(false);
    }
  });

  it("WORKSPACE_WRITE → WORKSPACE_READ 在沙箱内往返", async () => {
    const root = mkdtempSync(join(tmpdir(), "ointj-ws-rw-"));
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock", workspaceDir: root });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers));

    const wRaw = await handlers.get(IPC.WORKSPACE_WRITE)?.(
      {},
      { path: "notes/a.txt", content: "hello-ws" },
    );
    const wParsed = WorkspaceWriteResponseSchema.safeParse(wRaw);
    expect(wParsed.success).toBe(true);

    const rRaw = await handlers.get(IPC.WORKSPACE_READ)?.({}, { path: "notes/a.txt" });
    const rParsed = WorkspaceReadResponseSchema.safeParse(rRaw);
    expect(rParsed.success).toBe(true);
    if (rParsed.success) expect(rParsed.data.content).toBe("hello-ws");
  });

  it("WORKSPACE_READ 拒绝越界路径，返回 workspace_error", async () => {
    const root = mkdtempSync(join(tmpdir(), "ointj-ws-esc-"));
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock", workspaceDir: root });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers));

    const r = (await handlers.get(IPC.WORKSPACE_READ)?.({}, { path: "../../etc/passwd" })) as {
      error?: string;
    };
    expect(r.error).toBe("workspace_error");
  });

  it("WORKSPACE_PICK_DIR：无 picker 返回 canceled；注入 picker 返回 root", async () => {
    const root = mkdtempSync(join(tmpdir(), "ointj-ws-pick-"));
    const agent = await assembleDesktopAgent({ llmProvider: "mock", workspaceDir: root });

    const h1: Handlers = new Map();
    registerIpcHandlers(agent, undefined, makeFakeIpc(h1));
    const noPick = (await h1.get(IPC.WORKSPACE_PICK_DIR)?.({}, undefined)) as {
      canceled: boolean;
    };
    expect(noPick.canceled).toBe(true);

    const h2: Handlers = new Map();
    const deps: IpcDeps = { pickDirectory: async () => "C:/picked/dir" };
    registerIpcHandlers(agent, undefined, makeFakeIpc(h2), deps);
    const picked = (await h2.get(IPC.WORKSPACE_PICK_DIR)?.({}, undefined)) as {
      canceled: boolean;
      root?: string;
    };
    expect(picked.canceled).toBe(false);
    expect(picked.root).toBe("C:/picked/dir");
  });

  // ---------- 应用配置面 ----------

  it("CONFIG_GET / CONFIG_UPDATE 经注入的 ConfigService 往返并落盘", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ointj-cfg-"));
    const cfgPath = join(dir, "config.json");
    const config = createConfigService(cfgPath);
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers), { config });

    const before = (await handlers.get(IPC.CONFIG_GET)?.({}, undefined)) as Record<string, unknown>;
    expect(before).toEqual({});

    const updated = (await handlers.get(IPC.CONFIG_UPDATE)?.(
      {},
      { retrievalMode: "hybrid", enableCommands: true },
    )) as { retrievalMode?: string; enableCommands?: boolean };
    expect(updated.retrievalMode).toBe("hybrid");
    expect(updated.enableCommands).toBe(true);

    // 落盘可被独立读回
    const onDisk = JSON.parse(readFileSync(cfgPath, "utf8")) as { retrievalMode?: string };
    expect(onDisk.retrievalMode).toBe("hybrid");

    const after = (await handlers.get(IPC.CONFIG_GET)?.({}, undefined)) as {
      retrievalMode?: string;
    };
    expect(after.retrievalMode).toBe("hybrid");
  });

  it("CONFIG_UPDATE 拒绝非法补丁", async () => {
    const dir = mkdtempSync(join(tmpdir(), "ointj-cfg-bad-"));
    const config = createConfigService(join(dir, "config.json"));
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers), { config });

    const r = (await handlers.get(IPC.CONFIG_UPDATE)?.({}, { retrievalMode: "not-a-mode" })) as {
      error?: string;
    };
    expect(r.error).toBe("invalid_request");
  });

  it("APP_RESTART delegates to the graceful restart dependency", async () => {
    const handlers: Handlers = new Map();
    const restart = vi.fn(async () => {});
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers), { restart });

    await expect(handlers.get(IPC.APP_RESTART)?.({}, undefined)).resolves.toEqual({ ok: true });
    expect(restart).toHaveBeenCalledOnce();
  });

  it("APP_RESTART aborts in-flight chat before relaunching", async () => {
    const handlers: Handlers = new Map();
    const restart = vi.fn(async () => {});
    const hangingClient = {
      chat: vi.fn(
        async (_messages: unknown, opts?: { signal?: AbortSignal }): Promise<string> =>
          new Promise((_resolve, reject) => {
            opts?.signal?.addEventListener(
              "abort",
              () => reject(new DOMException("aborted", "AbortError")),
              { once: true },
            );
          }),
      ),
      visionChat: vi.fn(async () => "vision"),
      getStatus: () => ({
        provider: "glm",
        model: "glm-5.2",
        available: true,
        mode: "live" as const,
        status: "connected",
        visionSupported: false,
      }),
    };
    const modelRegistry = {
      list: () => [],
      resolve: vi.fn(async () => hangingClient),
      test: vi.fn(async () => ({ ok: true, provider: "glm", model: "glm-5.2" })),
      clear: vi.fn(),
    };
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers), {
      restart,
      modelRegistry,
    });

    const chat = handlers.get(IPC.CHAT)?.(
      {},
      { query: "深入分析这个复杂系统", modelProfileId: "glm-5.2" },
    );
    await vi.waitFor(() => expect(hangingClient.chat).toHaveBeenCalled());
    await handlers.get(IPC.APP_RESTART)?.({}, undefined);

    await expect(chat).rejects.toThrow("aborted");
    expect(restart).toHaveBeenCalledOnce();
  });

  it("MODEL_PROFILES reports credential presence without exposing the key", async () => {
    const handlers: Handlers = new Map();
    const values = new Map<string, string>();
    const credentials = {
      has: (id: string) => values.has(id),
      get: (id: string) => values.get(id),
      set: (id: string, value: string) => void values.set(id, value),
      delete: (id: string) => values.delete(id),
    };
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers), { credentials });
    const key = "super-secret-key";

    await expect(
      handlers.get(IPC.MODEL_CREDENTIAL_SET)?.({}, { profileId: "kimi-k3", apiKey: key }),
    ).resolves.toEqual({ ok: true });
    const profiles = (await handlers.get(IPC.MODEL_PROFILES)?.({}, undefined)) as Array<{
      id: string;
      hasCredential: boolean;
    }>;
    expect(profiles.find((profile) => profile.id === "kimi-k3")?.hasCredential).toBe(true);
    expect(JSON.stringify(profiles)).not.toContain(key);
  });

  // ---------- 技能自学习 Phase 2：审批 IPC 契约 ----------

  it("技能自学习 IPC 未启用时统一返回 skills_learning_not_enabled", async () => {
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers));

    for (const ch of [
      IPC.SKILLS_DISTILL,
      IPC.SKILLS_LIST,
      IPC.SKILLS_APPROVE,
      IPC.SKILLS_REJECT,
      IPC.SKILLS_REVOKE,
      IPC.SKILLS_ACTIVE,
    ]) {
      const r = (await handlers.get(ch)?.({}, { proposalId: "x" })) as { error?: string };
      expect(r.error).toBe("skills_learning_not_enabled");
    }
  });

  it("注册了所有 skill channel", async () => {
    const handle = vi.fn();
    const removeHandler = vi.fn();
    const fakeIpc = { handle, removeHandler } as unknown as Parameters<
      typeof registerIpcHandlers
    >[2];
    const agent = await assembleDesktopAgent({ llmProvider: "mock", enableSkillLearning: true });
    const reg = registerIpcHandlers(agent, undefined, fakeIpc);
    const channels = handle.mock.calls.map((c) => c[0] as string);
    expect(channels).toContain(IPC.SKILLS_DISTILL);
    expect(channels).toContain(IPC.SKILLS_LIST);
    expect(channels).toContain(IPC.SKILLS_APPROVE);
    expect(channels).toContain(IPC.SKILLS_REJECT);
    expect(channels).toContain(IPC.SKILLS_REVOKE);
    expect(channels).toContain(IPC.SKILLS_ACTIVE);
    reg.unregister();
    await agent.close();
  });

  it("技能完整链路：成功轨迹 → distill → list → approve → active + status.skills", async () => {
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock", enableSkillLearning: true });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers));

    // 直接喂成功轨迹（默认 minSamplesToDistill=3，taskType 缺省聚成 general 簇）。
    for (const q of [
      "review my python function for bugs",
      "review this java function for bugs",
      "review the typescript function for correctness",
    ]) {
      agent.skillLearning!.recordOutcome(q, undefined, "completed", {
        finalAnswer: "done",
        toolsUsed: ["read_file"],
      });
    }

    const distillRaw = await handlers.get(IPC.SKILLS_DISTILL)?.({}, undefined);
    const distillParsed = SkillDistillResponseSchema.safeParse(distillRaw);
    expect(distillParsed.success).toBe(true);
    if (!distillParsed.success) return;
    expect(distillParsed.data.produced).toBeGreaterThan(0);

    const listRaw = await handlers.get(IPC.SKILLS_LIST)?.({}, { status: "pending" });
    const listParsed = SkillListResponseSchema.safeParse(listRaw);
    expect(listParsed.success).toBe(true);
    if (!listParsed.success) return;
    expect(listParsed.data.total).toBeGreaterThan(0);
    const first = listParsed.data.proposals[0]!;
    expect(first.evidence.count).toBeGreaterThan(0);

    const approveRaw = await handlers.get(IPC.SKILLS_APPROVE)?.(
      {},
      { proposalId: first.proposalId },
    );
    const approveParsed = SkillDecisionResponseSchema.safeParse(approveRaw);
    expect(approveParsed.success).toBe(true);
    if (approveParsed.success) expect(approveParsed.data.status).toBe("approved");

    const activeRaw = await handlers.get(IPC.SKILLS_ACTIVE)?.({}, undefined);
    const activeParsed = SkillActiveResponseSchema.safeParse(activeRaw);
    expect(activeParsed.success).toBe(true);
    if (activeParsed.success) {
      expect(activeParsed.data.total).toBeGreaterThan(0);
      expect(activeParsed.data.skills[0]!.id).toBe(first.skillId);
    }

    // status.skills 反映生效技能数，且通过全字段 schema 校验。
    const statusRaw = await handlers.get(IPC.STATUS)?.({}, undefined);
    const statusParsed = StatusResponseSchema.safeParse(statusRaw);
    expect(statusParsed.success).toBe(true);
    if (statusParsed.success) {
      expect(statusParsed.data.skills?.enabled).toBe(true);
      expect(statusParsed.data.skills?.activeSkills).toBeGreaterThan(0);
    }

    await agent.close();
  });

  it("SKILLS_APPROVE 不存在 proposalId 返回 not_found_or_already_decided", async () => {
    const handlers: Handlers = new Map();
    const agent = await assembleDesktopAgent({ llmProvider: "mock", enableSkillLearning: true });
    registerIpcHandlers(agent, undefined, makeFakeIpc(handlers));

    const r = (await handlers.get(IPC.SKILLS_APPROVE)?.({}, { proposalId: "ghost" })) as {
      error?: string;
    };
    expect(r.error).toBe("not_found_or_already_decided");
    await agent.close();
  });

  it("注册了所有 workspace + config channel", async () => {
    const handle = vi.fn();
    const removeHandler = vi.fn();
    const fakeIpc = { handle, removeHandler } as unknown as Parameters<
      typeof registerIpcHandlers
    >[2];
    const agent = await assembleDesktopAgent({ llmProvider: "mock" });
    const reg = registerIpcHandlers(agent, undefined, fakeIpc);
    const channels = handle.mock.calls.map((c) => c[0] as string);
    expect(channels).toContain(IPC.WORKSPACE_READ);
    expect(channels).toContain(IPC.WORKSPACE_WRITE);
    expect(channels).toContain(IPC.WORKSPACE_INFO);
    expect(channels).toContain(IPC.WORKSPACE_PICK_DIR);
    expect(channels).toContain(IPC.CONFIG_GET);
    expect(channels).toContain(IPC.CONFIG_UPDATE);
    reg.unregister();
  });
});
