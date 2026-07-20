import { randomUUID } from "node:crypto";
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname } from "node:path";
import type {
  WorkbenchConversation,
  WorkbenchMessage,
  WorkbenchTask,
  WorkbenchWorkspace,
} from "../shared/ipc-protocol.js";
import { DEFAULT_DESKTOP_MODEL_PROFILE_ID, InputStructureSchema } from "../shared/ipc-protocol.js";

interface Statement {
  run(...params: unknown[]): { changes: number };
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
}
interface Database {
  pragma(source: string): unknown;
  exec(source: string): void;
  prepare(source: string): Statement;
  transaction<T extends (...args: never[]) => unknown>(fn: T): T;
  close(): void;
}
type DatabaseCtor = new (path: string) => Database;

const require = createRequire(import.meta.url);
const loadDatabase = (): DatabaseCtor => require("better-sqlite3") as DatabaseCtor;

interface WorkspaceRow {
  id: string;
  name: string;
  root_path: string;
  data_dir: string | null;
  created_at: number;
  updated_at: number;
}
interface TaskRow {
  id: string;
  workspace_id: string;
  title: string;
  status: WorkbenchTask["status"];
  taskpool_run_id: string | null;
  created_at: number;
  updated_at: number;
}
interface ConversationRow {
  id: string;
  task_id: string;
  title: string;
  model_profile_id: string;
  created_at: number;
  updated_at: number;
}
interface MessageRow {
  id: string;
  conversation_id: string;
  role: WorkbenchMessage["role"];
  content: string;
  trace_id: string | null;
  tokens: number | null;
  status: string | null;
  message_kind: WorkbenchMessage["messageKind"] | null;
  input_structure_json: string | null;
  created_at: number;
}

const workspace = (row: WorkspaceRow): WorkbenchWorkspace => ({
  id: row.id,
  name: row.name,
  rootPath: row.root_path,
  ...(row.data_dir ? { dataDir: row.data_dir } : {}),
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});
const task = (row: TaskRow): WorkbenchTask => ({
  id: row.id,
  workspaceId: row.workspace_id,
  title: row.title,
  status: row.status,
  ...(row.taskpool_run_id ? { taskPoolRunId: row.taskpool_run_id } : {}),
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});
const conversation = (row: ConversationRow): WorkbenchConversation => ({
  id: row.id,
  taskId: row.task_id,
  title: row.title,
  modelProfileId: row.model_profile_id,
  createdAt: row.created_at,
  updatedAt: row.updated_at,
});
const message = (row: MessageRow): WorkbenchMessage => {
  const parsedStructure = row.input_structure_json
    ? InputStructureSchema.safeParse(JSON.parse(row.input_structure_json))
    : undefined;
  return {
    id: row.id,
    conversationId: row.conversation_id,
    role: row.role,
    content: row.content,
    ...(row.trace_id ? { traceId: row.trace_id } : {}),
    ...(row.tokens !== null ? { tokens: row.tokens } : {}),
    ...(row.status ? { status: row.status } : {}),
    messageKind: row.message_kind ?? "message",
    ...(parsedStructure?.success ? { inputStructure: parsedStructure.data } : {}),
    createdAt: row.created_at,
  };
};

export interface WorkbenchStore {
  bootstrap(): {
    workspaces: WorkbenchWorkspace[];
    tasks: WorkbenchTask[];
    conversations: WorkbenchConversation[];
  };
  createWorkspace(input: { name: string; rootPath: string; dataDir?: string }): WorkbenchWorkspace;
  createTask(input: { workspaceId: string; title: string }): WorkbenchTask;
  updateTask(
    id: string,
    patch: Partial<Pick<WorkbenchTask, "title" | "status" | "taskPoolRunId">>,
  ): WorkbenchTask;
  createConversation(input: {
    taskId: string;
    title: string;
    modelProfileId: string;
  }): WorkbenchConversation;
  updateConversation(
    id: string,
    patch: Partial<Pick<WorkbenchConversation, "title" | "modelProfileId">>,
  ): WorkbenchConversation;
  getConversation(id: string): {
    workspace: WorkbenchWorkspace;
    task: WorkbenchTask;
    conversation: WorkbenchConversation;
  };
  listMessages(conversationId: string, limit?: number): WorkbenchMessage[];
  appendMessage(
    input: Omit<WorkbenchMessage, "id" | "createdAt" | "messageKind"> & {
      id?: string;
      createdAt?: number;
      messageKind?: WorkbenchMessage["messageKind"];
    },
  ): WorkbenchMessage;
  close(): void;
}

export const createWorkbenchStore = (opts: {
  dbPath: string;
  defaultWorkspaceRoot: string;
  defaultDataDir?: string;
  now?: () => number;
  idFactory?: () => string;
}): WorkbenchStore => {
  if (opts.dbPath !== ":memory:") mkdirSync(dirname(opts.dbPath), { recursive: true });
  mkdirSync(opts.defaultWorkspaceRoot, { recursive: true });
  const db = new (loadDatabase())(opts.dbPath);
  const now = opts.now ?? Date.now;
  const id = opts.idFactory ?? randomUUID;
  db.pragma("foreign_keys = ON");
  if (opts.dbPath !== ":memory:") db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS workspaces (
      id TEXT PRIMARY KEY, name TEXT NOT NULL, root_path TEXT NOT NULL,
      data_dir TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tasks (
      id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id),
      title TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
      taskpool_run_id TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_tasks_workspace ON tasks(workspace_id, updated_at DESC);
    CREATE TABLE IF NOT EXISTS conversations (
      id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
      title TEXT NOT NULL, model_profile_id TEXT NOT NULL,
      created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_conversations_task ON conversations(task_id, updated_at DESC);
    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
      role TEXT NOT NULL, content TEXT NOT NULL, trace_id TEXT, tokens INTEGER,
      status TEXT, message_kind TEXT NOT NULL DEFAULT 'message',
      input_structure_json TEXT, created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_messages_conversation
      ON messages(conversation_id, created_at ASC);
    PRAGMA user_version = 2;
  `);
  const messageColumns = new Set(
    (
      db.prepare("PRAGMA table_info(messages)").all() as Array<{
        name: string;
      }>
    ).map((column) => column.name),
  );
  if (!messageColumns.has("message_kind")) {
    db.exec("ALTER TABLE messages ADD COLUMN message_kind TEXT NOT NULL DEFAULT 'message'");
  }
  if (!messageColumns.has("input_structure_json")) {
    db.exec("ALTER TABLE messages ADD COLUMN input_structure_json TEXT");
  }
  db.pragma("user_version = 2");
  db.prepare("UPDATE conversations SET model_profile_id=? WHERE model_profile_id='auto'").run(
    DEFAULT_DESKTOP_MODEL_PROFILE_ID,
  );

  const seed = db.transaction(() => {
    const count = db.prepare("SELECT COUNT(*) AS count FROM workspaces").get() as { count: number };
    if (count.count > 0) return;
    const timestamp = now();
    const workspaceId = id();
    const taskId = id();
    const conversationId = id();
    db.prepare(
      "INSERT INTO workspaces(id,name,root_path,data_dir,created_at,updated_at) VALUES(?,?,?,?,?,?)",
    ).run(
      workspaceId,
      "默认工作区",
      opts.defaultWorkspaceRoot,
      opts.defaultDataDir ?? null,
      timestamp,
      timestamp,
    );
    db.prepare(
      "INSERT INTO tasks(id,workspace_id,title,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
    ).run(taskId, workspaceId, "Inbox", "active", timestamp, timestamp);
    db.prepare(
      "INSERT INTO conversations(id,task_id,title,model_profile_id,created_at,updated_at) VALUES(?,?,?,?,?,?)",
    ).run(conversationId, taskId, "新对话", DEFAULT_DESKTOP_MODEL_PROFILE_ID, timestamp, timestamp);
  });
  seed();

  const getWorkspace = (workspaceId: string): WorkbenchWorkspace => {
    const row = db.prepare("SELECT * FROM workspaces WHERE id=?").get(workspaceId) as
      | WorkspaceRow
      | undefined;
    if (!row) throw new Error(`workspace not found: ${workspaceId}`);
    return workspace(row);
  };
  const getTask = (taskId: string): WorkbenchTask => {
    const row = db.prepare("SELECT * FROM tasks WHERE id=?").get(taskId) as TaskRow | undefined;
    if (!row) throw new Error(`task not found: ${taskId}`);
    return task(row);
  };
  const getConversationRow = (conversationId: string): WorkbenchConversation => {
    const row = db.prepare("SELECT * FROM conversations WHERE id=?").get(conversationId) as
      | ConversationRow
      | undefined;
    if (!row) throw new Error(`conversation not found: ${conversationId}`);
    return conversation(row);
  };

  return {
    bootstrap() {
      return {
        workspaces: (
          db.prepare("SELECT * FROM workspaces ORDER BY updated_at DESC").all() as WorkspaceRow[]
        ).map(workspace),
        tasks: (db.prepare("SELECT * FROM tasks ORDER BY updated_at DESC").all() as TaskRow[]).map(
          task,
        ),
        conversations: (
          db
            .prepare("SELECT * FROM conversations ORDER BY updated_at DESC")
            .all() as ConversationRow[]
        ).map(conversation),
      };
    },
    createWorkspace(input) {
      mkdirSync(input.rootPath, { recursive: true });
      const timestamp = now();
      const workspaceId = id();
      db.prepare(
        "INSERT INTO workspaces(id,name,root_path,data_dir,created_at,updated_at) VALUES(?,?,?,?,?,?)",
      ).run(workspaceId, input.name, input.rootPath, input.dataDir ?? null, timestamp, timestamp);
      return getWorkspace(workspaceId);
    },
    createTask(input) {
      getWorkspace(input.workspaceId);
      const timestamp = now();
      const taskId = id();
      db.prepare(
        "INSERT INTO tasks(id,workspace_id,title,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
      ).run(taskId, input.workspaceId, input.title, "active", timestamp, timestamp);
      return getTask(taskId);
    },
    updateTask(taskId, patch) {
      const current = getTask(taskId);
      db.prepare("UPDATE tasks SET title=?,status=?,taskpool_run_id=?,updated_at=? WHERE id=?").run(
        patch.title ?? current.title,
        patch.status ?? current.status,
        patch.taskPoolRunId ?? current.taskPoolRunId ?? null,
        now(),
        taskId,
      );
      return getTask(taskId);
    },
    createConversation(input) {
      getTask(input.taskId);
      const timestamp = now();
      const conversationId = id();
      db.prepare(
        "INSERT INTO conversations(id,task_id,title,model_profile_id,created_at,updated_at) VALUES(?,?,?,?,?,?)",
      ).run(conversationId, input.taskId, input.title, input.modelProfileId, timestamp, timestamp);
      return getConversationRow(conversationId);
    },
    updateConversation(conversationId, patch) {
      const current = getConversationRow(conversationId);
      db.prepare("UPDATE conversations SET title=?,model_profile_id=?,updated_at=? WHERE id=?").run(
        patch.title ?? current.title,
        patch.modelProfileId ?? current.modelProfileId,
        now(),
        conversationId,
      );
      return getConversationRow(conversationId);
    },
    getConversation(conversationId) {
      const selected = getConversationRow(conversationId);
      const selectedTask = getTask(selected.taskId);
      return {
        workspace: getWorkspace(selectedTask.workspaceId),
        task: selectedTask,
        conversation: selected,
      };
    },
    listMessages(conversationId, limit = 80) {
      getConversationRow(conversationId);
      const rows = db
        .prepare(
          `SELECT * FROM (
             SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?
           ) ORDER BY created_at ASC`,
        )
        .all(conversationId, Math.max(1, Math.min(500, limit))) as MessageRow[];
      return rows.map(message);
    },
    appendMessage(input) {
      getConversationRow(input.conversationId);
      const messageId = input.id ?? id();
      const createdAt = input.createdAt ?? now();
      db.prepare(
        `INSERT INTO messages(
          id,conversation_id,role,content,trace_id,tokens,status,message_kind,
          input_structure_json,created_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?)`,
      ).run(
        messageId,
        input.conversationId,
        input.role,
        input.content,
        input.traceId ?? null,
        input.tokens ?? null,
        input.status ?? null,
        input.messageKind ?? "message",
        input.inputStructure ? JSON.stringify(input.inputStructure) : null,
        createdAt,
      );
      db.prepare("UPDATE conversations SET updated_at=? WHERE id=?").run(
        createdAt,
        input.conversationId,
      );
      return message(db.prepare("SELECT * FROM messages WHERE id=?").get(messageId) as MessageRow);
    },
    close() {
      db.close();
    },
  };
};
