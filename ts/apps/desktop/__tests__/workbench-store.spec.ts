import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createWorkbenchStore } from "../src/main/workbench-store.js";

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
    expect(seeded.conversations[0]?.modelProfileId).toBe("hunyuan-hy3");
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
    store.createWorkspace({ name: "Created", rootPath: createdRoot });
    expect(existsSync(createdRoot)).toBe(true);
    const seeded = store.bootstrap();
    const archived = store.updateTask(seeded.tasks[0]!.id, { status: "archived" });
    expect(archived.status).toBe("archived");
    expect(store.bootstrap().conversations).toHaveLength(1);
    store.close();
    rmSync(root, { recursive: true, force: true });
  });
});
