import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createWorkbenchStore } from "../src/main/workbench-store.js";

const require = createRequire(import.meta.url);

describe("WorkbenchStore", () => {
  it("seeds Inbox and persists task conversations with ordered messages", () => {
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
    expect(seeded.conversations[0]?.modelProfileId).toBe("hunyuan-hy4-preview");
    expect(seeded.tasks[0]?.title).toBe("Inbox");
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
    expect(store.getConversation(conversation.id).workspace.id).toBe(seeded.workspaces[0]!.id);
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

    const Database = require("better-sqlite3") as new (path: string) => {
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
