import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import {
  type AppendMessageInput,
  type ConversationRecord,
  DEFAULT_WORKSPACE_ID,
  type DesktopPersistenceOptions,
  type DesktopPersistenceRepository,
  INBOX_TASK_ID,
  type ListOptions,
  type MessageHistoryOptions,
  type MessageRecord,
  type MessageRole,
  type MessageStatus,
  type TaskPersistenceStatus,
  type TaskRecord,
  type TokenUsage,
  type UpdateConversationInput,
  type UpdateMessageInput,
  type UpdateTaskInput,
  type UpdateWorkspaceInput,
  type WorkspaceRecord,
} from "./desktop-persistence-types.js";

interface Statement {
  run(...params: unknown[]): { changes: number | bigint };
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
}

interface Database {
  exec(sql: string): void;
  prepare(sql: string): Statement;
  pragma(value: string, options?: { simple?: boolean }): unknown;
  close(): void;
}

type DatabaseConstructor = new (filename: string) => Database;

interface Migration {
  version: number;
  name: string;
  sql: string;
}

interface WorkspaceRow {
  id: string;
  name: string;
  root_path: string | null;
  created_at: number;
  updated_at: number;
  archived_at: number | null;
}

interface TaskRow {
  id: string;
  workspace_id: string;
  title: string;
  status: TaskPersistenceStatus;
  model_profile_id: string | null;
  created_at: number;
  updated_at: number;
  archived_at: number | null;
}

interface ConversationRow {
  id: string;
  task_id: string;
  title: string;
  model_profile_id: string | null;
  created_at: number;
  updated_at: number;
  archived_at: number | null;
}

interface MessageRow {
  id: string;
  conversation_id: string;
  sequence: number;
  role: MessageRole;
  content: string;
  model_profile_id: string | null;
  trace_id: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  status: MessageStatus;
  error_message: string | null;
  metadata_json: string;
  created_at: number;
  updated_at: number;
  archived_at: number | null;
}

const MIGRATIONS: readonly Migration[] = [
  {
    version: 1,
    name: "workspace_task_conversation_message_schema",
    sql: `
      CREATE TABLE desktop_schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at INTEGER NOT NULL
      );

      CREATE TABLE workspaces (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL CHECK(length(trim(name)) > 0),
        root_path TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        archived_at INTEGER
      );

      CREATE TABLE tasks (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL REFERENCES workspaces(id),
        title TEXT NOT NULL CHECK(length(trim(title)) > 0),
        status TEXT NOT NULL CHECK(status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
        model_profile_id TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        archived_at INTEGER
      );
      CREATE INDEX idx_desktop_tasks_workspace_updated
        ON tasks(workspace_id, archived_at, updated_at DESC, id);

      CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL REFERENCES tasks(id),
        title TEXT NOT NULL CHECK(length(trim(title)) > 0),
        model_profile_id TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        archived_at INTEGER
      );
      CREATE INDEX idx_desktop_conversations_task_updated
        ON conversations(task_id, archived_at, updated_at DESC, id);

      CREATE TABLE messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        sequence INTEGER NOT NULL CHECK(sequence >= 0),
        role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant', 'tool')),
        content TEXT NOT NULL,
        model_profile_id TEXT,
        trace_id TEXT,
        prompt_tokens INTEGER CHECK(prompt_tokens IS NULL OR prompt_tokens >= 0),
        completion_tokens INTEGER CHECK(completion_tokens IS NULL OR completion_tokens >= 0),
        total_tokens INTEGER CHECK(total_tokens IS NULL OR total_tokens >= 0),
        status TEXT NOT NULL CHECK(status IN ('pending', 'streaming', 'completed', 'failed', 'cancelled')),
        error_message TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        archived_at INTEGER,
        UNIQUE(conversation_id, sequence)
      );
      CREATE INDEX idx_desktop_messages_conversation_sequence
        ON messages(conversation_id, archived_at, sequence);
      CREATE INDEX idx_desktop_messages_trace ON messages(trace_id) WHERE trace_id IS NOT NULL;
    `,
  },
  {
    version: 2,
    name: "default_workspace_and_inbox",
    sql: `
      INSERT INTO workspaces (id, name, root_path, created_at, updated_at, archived_at)
      VALUES ('workspace:default', 'Default', NULL, 0, 0, NULL)
      ON CONFLICT(id) DO NOTHING;

      INSERT INTO tasks (
        id, workspace_id, title, status, model_profile_id, created_at, updated_at, archived_at
      )
      VALUES ('task:inbox', 'workspace:default', 'Inbox', 'pending', NULL, 0, 0, NULL)
      ON CONFLICT(id) DO NOTHING;
    `,
  },
];

export const DESKTOP_PERSISTENCE_SCHEMA_VERSION = MIGRATIONS.length;
const DEFAULT_MAX_HISTORY_MESSAGES = 200;
const MAX_CONFIGURED_HISTORY_MESSAGES = 10_000;

export async function openDesktopPersistenceRepository(
  options: DesktopPersistenceOptions,
): Promise<DesktopPersistenceRepository> {
  const dbPath = requireText(options.dbPath, "dbPath");
  if (dbPath !== ":memory:") await mkdir(dirname(dbPath), { recursive: true });

  const moduleName = "better-sqlite3";
  const imported = (await import(moduleName).catch((error) => {
    throw new Error(
      `DesktopPersistenceRepository: better-sqlite3 is required: ${(error as Error).message}`,
    );
  })) as { default?: DatabaseConstructor } & DatabaseConstructor;
  const Constructor = (imported.default ?? imported) as DatabaseConstructor;
  const db = new Constructor(dbPath);

  try {
    db.pragma("foreign_keys = ON");
    db.pragma("busy_timeout = 5000");
    if (dbPath !== ":memory:" && options.wal !== false) db.pragma("journal_mode = WAL");
    migrate(db);
    return new SqliteDesktopPersistenceRepository(db, { ...options, dbPath });
  } catch (error) {
    db.close();
    throw error;
  }
}

class SqliteDesktopPersistenceRepository implements DesktopPersistenceRepository {
  readonly dbPath: string;
  readonly schemaVersion = DESKTOP_PERSISTENCE_SCHEMA_VERSION;
  private readonly maxHistoryMessages: number;
  private readonly now: () => number;
  private readonly idFactory: () => string;
  private closed = false;

  constructor(
    private readonly db: Database,
    options: DesktopPersistenceOptions,
  ) {
    this.dbPath = options.dbPath;
    this.maxHistoryMessages = boundedInteger(
      options.maxHistoryMessages ?? DEFAULT_MAX_HISTORY_MESSAGES,
      1,
      MAX_CONFIGURED_HISTORY_MESSAGES,
      "maxHistoryMessages",
    );
    this.now = options.now ?? Date.now;
    this.idFactory = options.idFactory ?? (() => crypto.randomUUID());
  }

  getDefaultWorkspace(): WorkspaceRecord {
    return this.getWorkspace(DEFAULT_WORKSPACE_ID) ?? invariantMissing("default workspace");
  }

  getInboxTask(): TaskRecord {
    return this.getTask(INBOX_TASK_ID) ?? invariantMissing("Inbox task");
  }

  createWorkspace(input: {
    id?: string;
    name: string;
    rootPath?: string;
  }): WorkspaceRecord {
    this.assertOpen();
    const id = optionalText(input.id, "workspace id") ?? this.idFactory();
    const name = requireText(input.name, "workspace name");
    const rootPath = optionalText(input.rootPath, "rootPath") ?? null;
    const timestamp = this.timestamp();
    this.db
      .prepare(
        `INSERT INTO workspaces (id, name, root_path, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?)`,
      )
      .run(id, name, rootPath, timestamp, timestamp);
    return this.getWorkspace(id) ?? invariantMissing("created workspace");
  }

  getWorkspace(id: string, includeArchived = false): WorkspaceRecord | undefined {
    this.assertOpen();
    const row = this.db
      .prepare(
        `SELECT * FROM workspaces
         WHERE id = ? AND (? = 1 OR archived_at IS NULL)`,
      )
      .get(requireText(id, "workspace id"), includeArchived ? 1 : 0) as WorkspaceRow | undefined;
    return row ? mapWorkspace(row) : undefined;
  }

  listWorkspaces(options: ListOptions = {}): WorkspaceRecord[] {
    this.assertOpen();
    const limit = listLimit(options.limit);
    return (
      this.db
        .prepare(
          `SELECT * FROM workspaces
           WHERE (? = 1 OR archived_at IS NULL)
           ORDER BY updated_at DESC, id
           LIMIT ?`,
        )
        .all(options.includeArchived ? 1 : 0, limit) as WorkspaceRow[]
    ).map(mapWorkspace);
  }

  updateWorkspace(id: string, patch: UpdateWorkspaceInput): WorkspaceRecord | undefined {
    this.assertOpen();
    const current = this.getWorkspace(id, true);
    if (!current || current.archivedAt !== undefined) return undefined;
    if (patch.name === undefined && patch.rootPath === undefined) return current;
    const name =
      patch.name === undefined ? current.name : requireText(patch.name, "workspace name");
    const rootPath =
      patch.rootPath === undefined
        ? (current.rootPath ?? null)
        : patch.rootPath === null
          ? null
          : requireText(patch.rootPath, "rootPath");
    this.db
      .prepare("UPDATE workspaces SET name = ?, root_path = ?, updated_at = ? WHERE id = ?")
      .run(name, rootPath, this.timestamp(), current.id);
    return this.getWorkspace(current.id);
  }

  archiveWorkspace(id: string): boolean {
    this.assertOpen();
    const workspaceId = requireText(id, "workspace id");
    if (workspaceId === DEFAULT_WORKSPACE_ID) return false;
    const timestamp = this.timestamp();
    return this.transaction(() => {
      const result = this.db
        .prepare(
          `UPDATE workspaces SET archived_at = ?, updated_at = ?
           WHERE id = ? AND archived_at IS NULL`,
        )
        .run(timestamp, timestamp, workspaceId);
      if (!changed(result)) return false;
      this.db
        .prepare(
          `UPDATE tasks SET archived_at = ?, updated_at = ?
           WHERE workspace_id = ? AND archived_at IS NULL`,
        )
        .run(timestamp, timestamp, workspaceId);
      this.db
        .prepare(
          `UPDATE conversations SET archived_at = ?, updated_at = ?
           WHERE archived_at IS NULL
             AND task_id IN (SELECT id FROM tasks WHERE workspace_id = ?)`,
        )
        .run(timestamp, timestamp, workspaceId);
      return true;
    });
  }

  createTask(input: {
    id?: string;
    workspaceId: string;
    title: string;
    status?: TaskPersistenceStatus;
    modelProfileId?: string;
  }): TaskRecord {
    this.assertOpen();
    const workspaceId = requireText(input.workspaceId, "workspaceId");
    if (!this.getWorkspace(workspaceId))
      throw new Error(`Active workspace not found: ${workspaceId}`);
    const id = optionalText(input.id, "task id") ?? this.idFactory();
    const title = requireText(input.title, "task title");
    const status = input.status ?? "pending";
    const modelProfileId = optionalText(input.modelProfileId, "modelProfileId") ?? null;
    const timestamp = this.timestamp();
    this.db
      .prepare(
        `INSERT INTO tasks
          (id, workspace_id, title, status, model_profile_id, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(id, workspaceId, title, status, modelProfileId, timestamp, timestamp);
    return this.getTask(id) ?? invariantMissing("created task");
  }

  getTask(id: string, includeArchived = false): TaskRecord | undefined {
    this.assertOpen();
    const row = this.db
      .prepare("SELECT * FROM tasks WHERE id = ? AND (? = 1 OR archived_at IS NULL)")
      .get(requireText(id, "task id"), includeArchived ? 1 : 0) as TaskRow | undefined;
    return row ? mapTask(row) : undefined;
  }

  listTasks(workspaceId: string, options: ListOptions = {}): TaskRecord[] {
    this.assertOpen();
    return (
      this.db
        .prepare(
          `SELECT * FROM tasks
           WHERE workspace_id = ? AND (? = 1 OR archived_at IS NULL)
           ORDER BY updated_at DESC, id
           LIMIT ?`,
        )
        .all(
          requireText(workspaceId, "workspaceId"),
          options.includeArchived ? 1 : 0,
          listLimit(options.limit),
        ) as TaskRow[]
    ).map(mapTask);
  }

  updateTask(id: string, patch: UpdateTaskInput): TaskRecord | undefined {
    this.assertOpen();
    const current = this.getTask(id, true);
    if (!current || current.archivedAt !== undefined) return undefined;
    if (
      patch.title === undefined &&
      patch.status === undefined &&
      patch.modelProfileId === undefined
    ) {
      return current;
    }
    const title =
      patch.title === undefined ? current.title : requireText(patch.title, "task title");
    const modelProfileId =
      patch.modelProfileId === undefined
        ? (current.modelProfileId ?? null)
        : patch.modelProfileId === null
          ? null
          : requireText(patch.modelProfileId, "modelProfileId");
    this.db
      .prepare(
        `UPDATE tasks SET title = ?, status = ?, model_profile_id = ?, updated_at = ?
         WHERE id = ?`,
      )
      .run(title, patch.status ?? current.status, modelProfileId, this.timestamp(), current.id);
    return this.getTask(current.id);
  }

  archiveTask(id: string): boolean {
    this.assertOpen();
    const taskId = requireText(id, "task id");
    if (taskId === INBOX_TASK_ID) return false;
    const timestamp = this.timestamp();
    return this.transaction(() => {
      const result = this.db
        .prepare(
          "UPDATE tasks SET archived_at = ?, updated_at = ? WHERE id = ? AND archived_at IS NULL",
        )
        .run(timestamp, timestamp, taskId);
      if (!changed(result)) return false;
      this.db
        .prepare(
          `UPDATE conversations SET archived_at = ?, updated_at = ?
           WHERE task_id = ? AND archived_at IS NULL`,
        )
        .run(timestamp, timestamp, taskId);
      return true;
    });
  }

  createConversation(input: {
    id?: string;
    taskId: string;
    title: string;
    modelProfileId?: string;
  }): ConversationRecord {
    this.assertOpen();
    const taskId = requireText(input.taskId, "taskId");
    if (!this.getTask(taskId)) throw new Error(`Active task not found: ${taskId}`);
    const id = optionalText(input.id, "conversation id") ?? this.idFactory();
    const title = requireText(input.title, "conversation title");
    const modelProfileId = optionalText(input.modelProfileId, "modelProfileId") ?? null;
    const timestamp = this.timestamp();
    this.db
      .prepare(
        `INSERT INTO conversations
          (id, task_id, title, model_profile_id, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(id, taskId, title, modelProfileId, timestamp, timestamp);
    return this.getConversation(id) ?? invariantMissing("created conversation");
  }

  getConversation(id: string, includeArchived = false): ConversationRecord | undefined {
    this.assertOpen();
    const row = this.db
      .prepare("SELECT * FROM conversations WHERE id = ? AND (? = 1 OR archived_at IS NULL)")
      .get(requireText(id, "conversation id"), includeArchived ? 1 : 0) as
      | ConversationRow
      | undefined;
    return row ? mapConversation(row) : undefined;
  }

  listConversations(taskId: string, options: ListOptions = {}): ConversationRecord[] {
    this.assertOpen();
    return (
      this.db
        .prepare(
          `SELECT * FROM conversations
           WHERE task_id = ? AND (? = 1 OR archived_at IS NULL)
           ORDER BY updated_at DESC, id
           LIMIT ?`,
        )
        .all(
          requireText(taskId, "taskId"),
          options.includeArchived ? 1 : 0,
          listLimit(options.limit),
        ) as ConversationRow[]
    ).map(mapConversation);
  }

  updateConversation(id: string, patch: UpdateConversationInput): ConversationRecord | undefined {
    this.assertOpen();
    const current = this.getConversation(id, true);
    if (!current || current.archivedAt !== undefined) return undefined;
    if (patch.title === undefined && patch.modelProfileId === undefined) return current;
    const title =
      patch.title === undefined ? current.title : requireText(patch.title, "conversation title");
    const modelProfileId =
      patch.modelProfileId === undefined
        ? (current.modelProfileId ?? null)
        : patch.modelProfileId === null
          ? null
          : requireText(patch.modelProfileId, "modelProfileId");
    this.db
      .prepare(
        "UPDATE conversations SET title = ?, model_profile_id = ?, updated_at = ? WHERE id = ?",
      )
      .run(title, modelProfileId, this.timestamp(), current.id);
    return this.getConversation(current.id);
  }

  archiveConversation(id: string): boolean {
    this.assertOpen();
    const timestamp = this.timestamp();
    const result = this.db
      .prepare(
        `UPDATE conversations SET archived_at = ?, updated_at = ?
         WHERE id = ? AND archived_at IS NULL`,
      )
      .run(timestamp, timestamp, requireText(id, "conversation id"));
    return changed(result);
  }

  appendMessage(input: AppendMessageInput): MessageRecord {
    this.assertOpen();
    const conversationId = requireText(input.conversationId, "conversationId");
    if (!this.getConversation(conversationId)) {
      throw new Error(`Active conversation not found: ${conversationId}`);
    }
    const id = optionalText(input.id, "message id") ?? this.idFactory();
    const content =
      typeof input.content === "string" ? input.content : requireText(input.content, "content");
    const modelProfileId = optionalText(input.modelProfileId, "modelProfileId") ?? null;
    const traceId = optionalText(input.traceId, "traceId") ?? null;
    const tokens =
      input.tokenUsage === undefined ? undefined : validateTokenUsage(input.tokenUsage);
    const metadataJson = JSON.stringify(input.metadata ?? {});
    const timestamp = this.timestamp();

    return this.transaction(() => {
      const next = this.db
        .prepare(
          `SELECT COALESCE(MAX(sequence), -1) + 1 AS sequence
           FROM messages WHERE conversation_id = ?`,
        )
        .get(conversationId) as { sequence: number };
      this.db
        .prepare(
          `INSERT INTO messages (
            id, conversation_id, sequence, role, content, model_profile_id, trace_id,
            prompt_tokens, completion_tokens, total_tokens, status, error_message,
            metadata_json, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          id,
          conversationId,
          next.sequence,
          input.role,
          content,
          modelProfileId,
          traceId,
          tokens?.prompt ?? null,
          tokens?.completion ?? null,
          tokens?.total ?? null,
          input.status ?? "completed",
          input.errorMessage ?? null,
          metadataJson,
          timestamp,
          timestamp,
        );
      this.touchConversationAncestors(conversationId, timestamp);
      return this.getMessage(id) ?? invariantMissing("appended message");
    });
  }

  getMessage(id: string, includeArchived = false): MessageRecord | undefined {
    this.assertOpen();
    const row = this.db
      .prepare("SELECT * FROM messages WHERE id = ? AND (? = 1 OR archived_at IS NULL)")
      .get(requireText(id, "message id"), includeArchived ? 1 : 0) as MessageRow | undefined;
    return row ? mapMessage(row) : undefined;
  }

  listMessages(conversationId: string, options: MessageHistoryOptions = {}): MessageRecord[] {
    this.assertOpen();
    const requestedLimit =
      options.limit === undefined
        ? this.maxHistoryMessages
        : boundedInteger(options.limit, 1, this.maxHistoryMessages, "history limit");
    const before = options.beforeSequence ?? Number.MAX_SAFE_INTEGER;
    if (!Number.isSafeInteger(before) || before < 0) {
      throw new Error("beforeSequence must be a non-negative safe integer");
    }
    return (
      this.db
        .prepare(
          `SELECT * FROM (
             SELECT * FROM messages
             WHERE conversation_id = ?
               AND sequence < ?
               AND (? = 1 OR archived_at IS NULL)
             ORDER BY sequence DESC
             LIMIT ?
           ) ORDER BY sequence ASC`,
        )
        .all(
          requireText(conversationId, "conversationId"),
          before,
          options.includeArchived ? 1 : 0,
          requestedLimit,
        ) as MessageRow[]
    ).map(mapMessage);
  }

  updateMessage(id: string, patch: UpdateMessageInput): MessageRecord | undefined {
    this.assertOpen();
    const current = this.getMessage(id, true);
    if (!current || current.archivedAt !== undefined) return undefined;
    if (Object.keys(patch).length === 0) return current;
    const modelProfileId = nullableTextPatch(
      patch.modelProfileId,
      current.modelProfileId,
      "modelProfileId",
    );
    const traceId = nullableTextPatch(patch.traceId, current.traceId, "traceId");
    const errorMessage = nullableTextPatch(
      patch.errorMessage,
      current.errorMessage,
      "errorMessage",
    );
    const tokens =
      patch.tokenUsage === undefined
        ? current.tokenUsage
        : patch.tokenUsage === null
          ? undefined
          : validateTokenUsage(patch.tokenUsage);
    const timestamp = this.timestamp();
    this.db
      .prepare(
        `UPDATE messages SET
          content = ?, model_profile_id = ?, trace_id = ?,
          prompt_tokens = ?, completion_tokens = ?, total_tokens = ?,
          status = ?, error_message = ?, metadata_json = ?, updated_at = ?
         WHERE id = ?`,
      )
      .run(
        patch.content ?? current.content,
        modelProfileId ?? null,
        traceId ?? null,
        tokens?.prompt ?? null,
        tokens?.completion ?? null,
        tokens?.total ?? null,
        patch.status ?? current.status,
        errorMessage ?? null,
        JSON.stringify(patch.metadata ?? current.metadata),
        timestamp,
        current.id,
      );
    this.touchConversationAncestors(current.conversationId, timestamp);
    return this.getMessage(current.id);
  }

  archiveMessage(id: string): boolean {
    this.assertOpen();
    const timestamp = this.timestamp();
    const result = this.db
      .prepare(
        "UPDATE messages SET archived_at = ?, updated_at = ? WHERE id = ? AND archived_at IS NULL",
      )
      .run(timestamp, timestamp, requireText(id, "message id"));
    return changed(result);
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.db.close();
  }

  private touchConversationAncestors(conversationId: string, timestamp: number): void {
    this.db
      .prepare("UPDATE conversations SET updated_at = ? WHERE id = ?")
      .run(timestamp, conversationId);
    this.db
      .prepare(
        `UPDATE tasks SET updated_at = ?
         WHERE id = (SELECT task_id FROM conversations WHERE id = ?)`,
      )
      .run(timestamp, conversationId);
    this.db
      .prepare(
        `UPDATE workspaces SET updated_at = ?
         WHERE id = (
           SELECT t.workspace_id
           FROM tasks t JOIN conversations c ON c.task_id = t.id
           WHERE c.id = ?
         )`,
      )
      .run(timestamp, conversationId);
  }

  private transaction<T>(operation: () => T): T {
    this.db.exec("BEGIN IMMEDIATE");
    try {
      const result = operation();
      this.db.exec("COMMIT");
      return result;
    } catch (error) {
      try {
        this.db.exec("ROLLBACK");
      } catch {
        // Preserve the original operation error.
      }
      throw error;
    }
  }

  private timestamp(): number {
    const value = this.now();
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error("now() must return a non-negative safe integer");
    }
    return value;
  }

  private assertOpen(): void {
    if (this.closed) throw new Error("DesktopPersistenceRepository is closed");
  }
}

function migrate(db: Database): void {
  const current = Number(db.pragma("user_version", { simple: true }));
  if (!Number.isSafeInteger(current) || current < 0) {
    throw new Error(`Invalid desktop persistence schema version: ${String(current)}`);
  }
  if (current > DESKTOP_PERSISTENCE_SCHEMA_VERSION) {
    throw new Error(
      `Desktop persistence database version ${current} is newer than supported version ${DESKTOP_PERSISTENCE_SCHEMA_VERSION}`,
    );
  }

  if (current > 0) {
    const table = db
      .prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'desktop_schema_migrations'",
      )
      .get();
    if (!table) throw new Error("Desktop persistence schema is missing migration metadata");
    const row = db
      .prepare("SELECT MAX(version) AS version FROM desktop_schema_migrations")
      .get() as { version: number | null };
    if (row.version !== current) {
      throw new Error(
        `Desktop persistence migration metadata (${String(row.version)}) does not match user_version (${current})`,
      );
    }
  }

  for (const migration of MIGRATIONS) {
    if (migration.version <= current) continue;
    db.exec("BEGIN IMMEDIATE");
    try {
      db.exec(migration.sql);
      db.prepare(
        `INSERT INTO desktop_schema_migrations (version, name, applied_at)
         VALUES (?, ?, ?)`,
      ).run(migration.version, migration.name, Date.now());
      db.pragma(`user_version = ${migration.version}`);
      db.exec("COMMIT");
    } catch (error) {
      try {
        db.exec("ROLLBACK");
      } catch {
        // Preserve the migration failure.
      }
      throw new Error(
        `Desktop persistence migration ${migration.version} (${migration.name}) failed: ${(error as Error).message}`,
        { cause: error },
      );
    }
  }
}

function mapWorkspace(row: WorkspaceRow): WorkspaceRecord {
  return {
    id: row.id,
    name: row.name,
    rootPath: row.root_path ?? undefined,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    archivedAt: row.archived_at ?? undefined,
  };
}

function mapTask(row: TaskRow): TaskRecord {
  return {
    id: row.id,
    workspaceId: row.workspace_id,
    title: row.title,
    status: row.status,
    modelProfileId: row.model_profile_id ?? undefined,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    archivedAt: row.archived_at ?? undefined,
  };
}

function mapConversation(row: ConversationRow): ConversationRecord {
  return {
    id: row.id,
    taskId: row.task_id,
    title: row.title,
    modelProfileId: row.model_profile_id ?? undefined,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    archivedAt: row.archived_at ?? undefined,
  };
}

function mapMessage(row: MessageRow): MessageRecord {
  const metadata = JSON.parse(row.metadata_json) as unknown;
  if (metadata === null || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error(`Message ${row.id} has invalid metadata JSON`);
  }
  const hasTokens =
    row.prompt_tokens !== null || row.completion_tokens !== null || row.total_tokens !== null;
  if (
    hasTokens &&
    (row.prompt_tokens === null || row.completion_tokens === null || row.total_tokens === null)
  ) {
    throw new Error(`Message ${row.id} has incomplete token metadata`);
  }
  return {
    id: row.id,
    conversationId: row.conversation_id,
    sequence: row.sequence,
    role: row.role,
    content: row.content,
    modelProfileId: row.model_profile_id ?? undefined,
    traceId: row.trace_id ?? undefined,
    tokenUsage: hasTokens
      ? {
          prompt: row.prompt_tokens as number,
          completion: row.completion_tokens as number,
          total: row.total_tokens as number,
        }
      : undefined,
    status: row.status,
    errorMessage: row.error_message ?? undefined,
    metadata: metadata as Record<string, unknown>,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    archivedAt: row.archived_at ?? undefined,
  };
}

function requireText(value: string, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function optionalText(value: string | undefined, label: string): string | undefined {
  return value === undefined ? undefined : requireText(value, label);
}

function nullableTextPatch(
  value: string | null | undefined,
  current: string | undefined,
  label: string,
): string | undefined {
  if (value === undefined) return current;
  if (value === null) return undefined;
  return requireText(value, label);
}

function validateTokenUsage(tokens: TokenUsage): TokenUsage {
  for (const [name, value] of Object.entries(tokens)) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`tokenUsage.${name} must be a non-negative safe integer`);
    }
  }
  if (tokens.total < tokens.prompt || tokens.total < tokens.completion) {
    throw new Error("tokenUsage.total cannot be less than prompt or completion tokens");
  }
  return tokens;
}

function boundedInteger(value: number, min: number, max: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < min) {
    throw new Error(`${label} must be a safe integer greater than or equal to ${min}`);
  }
  return Math.min(value, max);
}

function listLimit(value: number | undefined): number {
  return value === undefined
    ? MAX_CONFIGURED_HISTORY_MESSAGES
    : boundedInteger(value, 1, MAX_CONFIGURED_HISTORY_MESSAGES, "list limit");
}

function changed(result: { changes: number | bigint }): boolean {
  return Number(result.changes) > 0;
}

function invariantMissing(name: string): never {
  throw new Error(`Desktop persistence invariant failed: missing ${name}`);
}
