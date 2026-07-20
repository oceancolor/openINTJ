import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  DESKTOP_PERSISTENCE_SCHEMA_VERSION,
  openDesktopPersistenceRepository,
} from "../src/main/desktop-persistence-repository.js";
import { DEFAULT_WORKSPACE_ID, INBOX_TASK_ID } from "../src/main/desktop-persistence-types.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories
      .splice(0)
      .map((directory) => rm(directory, { recursive: true, force: true })),
  );
});

async function temporaryDatabase(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "openintj-desktop-repository-"));
  temporaryDirectories.push(directory);
  return join(directory, "nested", "desktop.sqlite");
}

describe("desktop persistence repository", () => {
  it("migrates an in-memory database and seeds Default plus Inbox", async () => {
    const repository = await openDesktopPersistenceRepository({ dbPath: ":memory:" });

    expect(repository.schemaVersion).toBe(DESKTOP_PERSISTENCE_SCHEMA_VERSION);
    expect(repository.getDefaultWorkspace()).toMatchObject({
      id: DEFAULT_WORKSPACE_ID,
      name: "Default",
      archivedAt: undefined,
    });
    expect(repository.getInboxTask()).toMatchObject({
      id: INBOX_TASK_ID,
      workspaceId: DEFAULT_WORKSPACE_ID,
      title: "Inbox",
    });
    expect(repository.archiveWorkspace(DEFAULT_WORKSPACE_ID)).toBe(false);
    expect(repository.archiveTask(INBOX_TASK_ID)).toBe(false);

    repository.close();
  });

  it("supports workspace, task, and conversation CRUD without destructive deletes", async () => {
    let now = 100;
    const repository = await openDesktopPersistenceRepository({
      dbPath: ":memory:",
      now: () => now++,
    });
    const workspace = repository.createWorkspace({
      id: "workspace:product",
      name: "Product",
      rootPath: "F:\\product",
    });
    const task = repository.createTask({
      id: "task:roadmap",
      workspaceId: workspace.id,
      title: "Roadmap",
      status: "in_progress",
      modelProfileId: "profile:reasoning",
    });
    const conversation = repository.createConversation({
      id: "conversation:q3",
      taskId: task.id,
      title: "Q3 planning",
      modelProfileId: "profile:fast",
    });

    expect(repository.updateWorkspace(workspace.id, { name: "Product Team" })?.name).toBe(
      "Product Team",
    );
    expect(
      repository.updateTask(task.id, { status: "completed", modelProfileId: null }),
    ).toMatchObject({ status: "completed", modelProfileId: undefined });
    expect(
      repository.updateConversation(conversation.id, { modelProfileId: "profile:reasoning" }),
    ).toMatchObject({ modelProfileId: "profile:reasoning" });

    expect(repository.archiveWorkspace(workspace.id)).toBe(true);
    expect(repository.getWorkspace(workspace.id)).toBeUndefined();
    expect(repository.getTask(task.id)).toBeUndefined();
    expect(repository.getConversation(conversation.id)).toBeUndefined();
    expect(repository.getWorkspace(workspace.id, true)?.archivedAt).toBeDefined();
    expect(repository.getTask(task.id, true)?.archivedAt).toBeDefined();
    expect(repository.getConversation(conversation.id, true)?.archivedAt).toBeDefined();
    expect(repository.archiveWorkspace(workspace.id)).toBe(false);

    repository.close();
  });

  it("stores ordered messages and operational metadata", async () => {
    const repository = await openDesktopPersistenceRepository({ dbPath: ":memory:" });
    const conversation = repository.createConversation({
      id: "conversation:metadata",
      taskId: INBOX_TASK_ID,
      title: "Metadata",
      modelProfileId: "profile:conversation",
    });
    const first = repository.appendMessage({
      id: "message:user",
      conversationId: conversation.id,
      role: "user",
      content: "Plan this",
      traceId: "trace-1",
      status: "completed",
      metadata: { source: "desktop" },
    });
    const second = repository.appendMessage({
      id: "message:assistant",
      conversationId: conversation.id,
      role: "assistant",
      content: "Working",
      modelProfileId: "profile:response",
      traceId: "trace-1",
      status: "streaming",
    });

    expect([first.sequence, second.sequence]).toEqual([0, 1]);
    expect(
      repository.updateMessage(second.id, {
        content: "Done",
        status: "completed",
        tokenUsage: { prompt: 7, completion: 5, total: 12 },
        metadata: { finishReason: "stop" },
      }),
    ).toMatchObject({
      sequence: 1,
      content: "Done",
      status: "completed",
      modelProfileId: "profile:response",
      traceId: "trace-1",
      tokenUsage: { prompt: 7, completion: 5, total: 12 },
      metadata: { finishReason: "stop" },
    });
    expect(repository.listMessages(conversation.id).map((message) => message.id)).toEqual([
      first.id,
      second.id,
    ]);

    expect(repository.archiveMessage(first.id)).toBe(true);
    expect(repository.listMessages(conversation.id).map((message) => message.id)).toEqual([
      second.id,
    ]);
    expect(
      repository
        .listMessages(conversation.id, { includeArchived: true })
        .map((message) => message.id),
    ).toEqual([first.id, second.id]);

    repository.close();
  });

  it("bounds history and paginates backwards while returning chronological order", async () => {
    const repository = await openDesktopPersistenceRepository({
      dbPath: ":memory:",
      maxHistoryMessages: 3,
    });
    const conversation = repository.createConversation({
      taskId: INBOX_TASK_ID,
      title: "Bounded",
    });
    for (let index = 0; index < 6; index++) {
      repository.appendMessage({
        conversationId: conversation.id,
        role: index % 2 === 0 ? "user" : "assistant",
        content: String(index),
      });
    }

    expect(repository.listMessages(conversation.id).map((message) => message.sequence)).toEqual([
      3, 4, 5,
    ]);
    expect(
      repository
        .listMessages(conversation.id, { limit: 2, beforeSequence: 4 })
        .map((message) => message.sequence),
    ).toEqual([2, 3]);
    expect(
      repository.listMessages(conversation.id, { limit: 4 }).map((message) => message.sequence),
    ).toEqual([3, 4, 5]);

    repository.close();
  });

  it("persists records across close and reopen in file mode", async () => {
    const dbPath = await temporaryDatabase();
    const first = await openDesktopPersistenceRepository({ dbPath });
    const conversation = first.createConversation({
      id: "conversation:persisted",
      taskId: INBOX_TASK_ID,
      title: "Persisted",
      modelProfileId: "profile:file",
    });
    first.appendMessage({
      conversationId: conversation.id,
      role: "assistant",
      content: "Survives restart",
      status: "completed",
      traceId: "trace-file",
    });
    first.close();

    const second = await openDesktopPersistenceRepository({ dbPath });
    expect(second.getDefaultWorkspace().id).toBe(DEFAULT_WORKSPACE_ID);
    expect(second.getConversation(conversation.id)?.modelProfileId).toBe("profile:file");
    expect(second.listMessages(conversation.id)[0]).toMatchObject({
      content: "Survives restart",
      traceId: "trace-file",
    });
    second.close();
  });

  it("refuses databases created by a newer schema version", async () => {
    const dbPath = await temporaryDatabase();
    const repository = await openDesktopPersistenceRepository({ dbPath });
    repository.close();

    const moduleName = "better-sqlite3";
    const imported = (await import(moduleName)) as unknown as {
      default: new (
        filename: string,
      ) => {
        pragma(value: string): unknown;
        close(): void;
      };
    };
    const raw = new imported.default(dbPath);
    raw.pragma(`user_version = ${DESKTOP_PERSISTENCE_SCHEMA_VERSION + 1}`);
    raw.close();

    await expect(openDesktopPersistenceRepository({ dbPath })).rejects.toThrow(
      "newer than supported",
    );
  });

  it("closes idempotently and rejects later operations", async () => {
    const repository = await openDesktopPersistenceRepository({ dbPath: ":memory:" });
    repository.close();
    repository.close();

    expect(() => repository.listWorkspaces()).toThrow("is closed");
  });
});
