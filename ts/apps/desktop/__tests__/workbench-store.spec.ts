import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createWorkbenchStore } from "../src/main/workbench-store.js";

const require = createRequire(import.meta.url);

describe("WorkbenchStore", () => {
  it("seeds 新创建的任务 and persists task conversations with ordered messages", () => {
    let seq = 0;
    const root = mkdtempSync(join(tmpdir(), "openintj-workbench-"));
    const store = createWorkbenchStore({
      dbPath: ":memory:",
      defaultWorkspaceRoot: root,
      now: () => 1_000 + seq,
      idFactory: () => `id-${++seq}`,
    });
    const seeded = store.bootstrap();
    expect(seeded.workspaces).toHaveLength(1);
    expect(seeded.workspaces[0]?.rootPath).toBe(root);
    expect(seeded.conversations[0]?.modelProfileId).toBe("hunyuan-hy4");
    expect(seeded.tasks[0]?.title).toBe("新创建的任务");
    expect(seeded.conversations).toHaveLength(1);

    const task = store.createTask({
      workspaceId: seeded.workspaces[0]!.id,
      title: "发布桌面版",
    });
    const conversation = store.createConversation({
      taskId: task.id,
      title: "签名排查",
      modelProfileId: "glm-5.2",
    });
    store.appendMessage({
      conversationId: conversation.id,
      role: "user",
      content: "检查签名",
    });
    store.appendMessage({
      conversationId: conversation.id,
      role: "assistant",
      content: "开始检查",
      traceId: "trace-1",
      tokens: 12,
      status: "completed",
    });

    expect(store.listMessages(conversation.id)).toMatchObject([
      { role: "user", content: "检查签名" },
      { role: "assistant", traceId: "trace-1", tokens: 12 },
    ]);
    expect(store.listJournal(conversation.id).map((event) => event.type)).toEqual([
      "turn.start",
      "user",
      "assistant",
      "turn.end",
    ]);
    store.clearMessages(conversation.id);
    expect(store.listMessages(conversation.id)).toEqual([]);
    expect(store.listJournal(conversation.id).at(-1)).toMatchObject({
      type: "session.cleared",
      command: "/clear",
    });
    expect(store.getConversation(conversation.id).workspace.id).toBe(seeded.workspaces[0]!.id);
    store.close();
    rmSync(root, { recursive: true, force: true });
  });

  it("migrates legacy Inbox task titles to 新创建的任务 on reopen", () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-workbench-"));
    const dbPath = join(root, "workbench.sqlite");
    const first = createWorkbenchStore({ dbPath, defaultWorkspaceRoot: root });
    const workspaceId = first.bootstrap().workspaces[0]!.id;
    const legacy = first.createTask({ workspaceId, title: "Inbox" });
    const custom = first.createTask({ workspaceId, title: "我的任务" });
    first.close();

    const reopened = createWorkbenchStore({ dbPath, defaultWorkspaceRoot: root });
    const tasks = reopened.bootstrap().tasks;
    expect(tasks.find((item) => item.id === legacy.id)?.title).toBe("新创建的任务");
    expect(tasks.find((item) => item.id === custom.id)?.title).toBe("我的任务");
    reopened.close();
    rmSync(root, { recursive: true, force: true });
  });

  it("deletes tasks and conversations with their messages; seed workspace is not named 默认工作区", () => {
    let seq = 0;
    const root = mkdtempSync(join(tmpdir(), "openintj-workbench-"));
    const store = createWorkbenchStore({
      dbPath: ":memory:",
      defaultWorkspaceRoot: root,
      now: () => 3_000 + seq++,
    });
    const seeded = store.bootstrap();
    expect(seeded.workspaces[0]?.name).not.toBe("默认工作区");
    expect(seeded.workspaces[0]?.name).toBe(root.split(/[\\/]/).filter(Boolean).at(-1));
    expect(store.updateWorkspace(seeded.workspaces[0]!.id, { name: "改名后的工作区" }).name).toBe(
      "改名后的工作区",
    );

    const workspaceId = seeded.workspaces[0]!.id;
    const task = store.createTask({ workspaceId, title: "待删除任务" });
    const conversation = store.createConversation({
      taskId: task.id,
      title: "待删除会话",
      modelProfileId: "hunyuan-hy4",
    });
    store.appendMessage({ conversationId: conversation.id, role: "user", content: "你好" });

    store.deleteConversation(conversation.id);
    expect(store.bootstrap().conversations.map((item) => item.id)).not.toContain(conversation.id);
    expect(() => store.listMessages(conversation.id)).toThrow();

    const conversation2 = store.createConversation({
      taskId: task.id,
      title: "随任务删除",
      modelProfileId: "hunyuan-hy4",
    });
    store.appendMessage({ conversationId: conversation2.id, role: "user", content: "再见" });
    store.deleteTask(task.id);
    const after = store.bootstrap();
    expect(after.tasks.map((item) => item.id)).not.toContain(task.id);
    expect(after.conversations.map((item) => item.id)).not.toContain(conversation2.id);
    expect(() => store.deleteTask(task.id)).toThrow();
    store.close();
    rmSync(root, { recursive: true, force: true });
  });

  it("persists understanding cards and clarification message kinds", () => {
    let seq = 0;
    const root = mkdtempSync(join(tmpdir(), "openintj-workbench-"));
    const store = createWorkbenchStore({
      dbPath: ":memory:",
      defaultWorkspaceRoot: root,
      now: () => 2_000 + seq++,
    });
    const conversation = store.bootstrap().conversations[0]!;
    store.appendMessage({
      conversationId: conversation.id,
      role: "user",
      content: "部署到生产",
    });
    store.appendMessage({
      conversationId: conversation.id,
      role: "assistant",
      content: "1. 部署到哪个环境？",
      messageKind: "clarification",
      status: "completed",
      inputStructure: {
        action: "clarify",
        mode: "clarification",
        executionInput: "",
        structure: {
          goal: "部署到生产",
          context: [],
          relations: [],
          constraints: [],
          deliverables: [],
          dependencies: [],
          assumptions: [],
        },
        ambiguityScore: 1,
        questions: ["部署到哪个环境？"],
        tokensSpent: 0,
        durationMs: 0,
      },
    });

    const messages = store.listMessages(conversation.id);
    expect(messages[1]).toMatchObject({
      messageKind: "clarification",
      inputStructure: {
        action: "clarify",
        questions: ["部署到哪个环境？"],
      },
    });
    store.close();
    rmSync(root, { recursive: true, force: true });
  });

  it("physically creates workspace directories and archives without deleting conversations", () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-workbench-"));
    const createdRoot = join(root, "created-workspace");
    const store = createWorkbenchStore({
      dbPath: ":memory:",
      defaultWorkspaceRoot: root,
    });
    const seeded = store.bootstrap();
    const conversationId = seeded.conversations[0]!.id;
    store.createWorkspace({ name: "Created", rootPath: createdRoot });
    expect(existsSync(createdRoot)).toBe(true);
    const open = store.bootstrap();
    expect(open.workspaces).toHaveLength(1);
    expect(open.workspaces[0]?.name).toBe("Created");
    expect(open.tasks).toHaveLength(0);
    expect(store.getConversation(conversationId).conversation.id).toBe(conversationId);
    const archived = store.updateTask(seeded.tasks[0]!.id, { status: "archived" });
    expect(archived.status).toBe("archived");
    store.close();
    rmSync(root, { recursive: true, force: true });
  });

  it("keeps only one open workspace and restores the same id by root path", () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-workbench-"));
    const firstRoot = join(root, "first");
    const secondRoot = join(root, "second");
    const store = createWorkbenchStore({
      dbPath: ":memory:",
      defaultWorkspaceRoot: firstRoot,
    });
    const seeded = store.bootstrap();
    const firstId = seeded.workspaces[0]!.id;
    const second = store.createWorkspace({ name: "Second", rootPath: secondRoot });
    const afterOpen = store.bootstrap();
    expect(afterOpen.workspaces).toHaveLength(1);
    expect(afterOpen.workspaces[0]?.id).toBe(second.id);
    expect(existsSync(firstRoot)).toBe(true);

    store.closeWorkspace(second.id);
    expect(store.bootstrap().workspaces).toHaveLength(0);
    expect(existsSync(secondRoot)).toBe(true);
    expect(() => store.createTask({ workspaceId: second.id, title: "x" })).toThrow(/closed/);

    const restored = store.createWorkspace({ name: "First again", rootPath: firstRoot });
    expect(restored.id).toBe(firstId);
    expect(store.bootstrap().workspaces).toEqual([
      expect.objectContaining({ id: firstId, name: "First again" }),
    ]);
    store.close();
    rmSync(root, { recursive: true, force: true });
  });

  it("bootstrap closes extra open workspaces left from older databases", () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-workbench-"));
    const dbPath = join(root, "workbench.sqlite");
    const extraRoot = join(root, "extra");
    const store = createWorkbenchStore({
      dbPath,
      defaultWorkspaceRoot: root,
    });
    const originalId = store.bootstrap().workspaces[0]!.id;
    store.close();

    const Database = require("better-sqlite3") as new (
      path: string,
    ) => {
      prepare(sql: string): { run(...params: unknown[]): unknown };
      close(): void;
    };
    const db = new Database(dbPath);
    const later = Date.now() + 10_000;
    db.prepare(
      "INSERT INTO workspaces(id,name,root_path,created_at,updated_at,closed_at) VALUES(?,?,?,?,?,NULL)",
    ).run("extra", "Extra", extraRoot, later, later);
    db.close();

    const reopened = createWorkbenchStore({
      dbPath,
      defaultWorkspaceRoot: root,
    });
    const snapshot = reopened.bootstrap();
    expect(snapshot.workspaces).toHaveLength(1);
    expect(snapshot.workspaces[0]?.id).toBe("extra");
    expect(snapshot.workspaces[0]?.id).not.toBe(originalId);
    reopened.close();
    rmSync(root, { recursive: true, force: true });
  });
});
