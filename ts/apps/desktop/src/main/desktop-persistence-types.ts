export const DEFAULT_WORKSPACE_ID = "workspace:default";
export const INBOX_TASK_ID = "task:inbox";

export type TaskPersistenceStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed"
  | "cancelled";

export type MessageRole = "system" | "user" | "assistant" | "tool";
export type MessageStatus = "pending" | "streaming" | "completed" | "failed" | "cancelled";

export interface WorkspaceRecord {
  id: string;
  name: string;
  rootPath: string | undefined;
  createdAt: number;
  updatedAt: number;
  archivedAt: number | undefined;
}

export interface TaskRecord {
  id: string;
  workspaceId: string;
  title: string;
  status: TaskPersistenceStatus;
  modelProfileId: string | undefined;
  createdAt: number;
  updatedAt: number;
  archivedAt: number | undefined;
}

export interface ConversationRecord {
  id: string;
  taskId: string;
  title: string;
  modelProfileId: string | undefined;
  createdAt: number;
  updatedAt: number;
  archivedAt: number | undefined;
}

export interface TokenUsage {
  prompt: number;
  completion: number;
  total: number;
}

export interface MessageRecord {
  id: string;
  conversationId: string;
  sequence: number;
  role: MessageRole;
  content: string;
  modelProfileId: string | undefined;
  traceId: string | undefined;
  tokenUsage: TokenUsage | undefined;
  status: MessageStatus;
  errorMessage: string | undefined;
  metadata: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
  archivedAt: number | undefined;
}

export interface CreateWorkspaceInput {
  id?: string;
  name: string;
  rootPath?: string;
}

export interface UpdateWorkspaceInput {
  name?: string;
  rootPath?: string | null;
}

export interface CreateTaskInput {
  id?: string;
  workspaceId: string;
  title: string;
  status?: TaskPersistenceStatus;
  modelProfileId?: string;
}

export interface UpdateTaskInput {
  title?: string;
  status?: TaskPersistenceStatus;
  modelProfileId?: string | null;
}

export interface CreateConversationInput {
  id?: string;
  taskId: string;
  title: string;
  modelProfileId?: string;
}

export interface UpdateConversationInput {
  title?: string;
  modelProfileId?: string | null;
}

export interface AppendMessageInput {
  id?: string;
  conversationId: string;
  role: MessageRole;
  content: string;
  modelProfileId?: string;
  traceId?: string;
  tokenUsage?: TokenUsage;
  status?: MessageStatus;
  errorMessage?: string;
  metadata?: Record<string, unknown>;
}

export interface UpdateMessageInput {
  content?: string;
  modelProfileId?: string | null;
  traceId?: string | null;
  tokenUsage?: TokenUsage | null;
  status?: MessageStatus;
  errorMessage?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ListOptions {
  includeArchived?: boolean;
  limit?: number;
}

export interface MessageHistoryOptions {
  limit?: number;
  beforeSequence?: number;
  includeArchived?: boolean;
}

export interface DesktopPersistenceOptions {
  dbPath: string;
  wal?: boolean;
  maxHistoryMessages?: number;
  now?: () => number;
  idFactory?: () => string;
}

export interface DesktopPersistenceRepository {
  readonly dbPath: string;
  readonly schemaVersion: number;

  getDefaultWorkspace(): WorkspaceRecord;
  getInboxTask(): TaskRecord;

  createWorkspace(input: CreateWorkspaceInput): WorkspaceRecord;
  getWorkspace(id: string, includeArchived?: boolean): WorkspaceRecord | undefined;
  listWorkspaces(options?: ListOptions): WorkspaceRecord[];
  updateWorkspace(id: string, patch: UpdateWorkspaceInput): WorkspaceRecord | undefined;
  archiveWorkspace(id: string): boolean;

  createTask(input: CreateTaskInput): TaskRecord;
  getTask(id: string, includeArchived?: boolean): TaskRecord | undefined;
  listTasks(workspaceId: string, options?: ListOptions): TaskRecord[];
  updateTask(id: string, patch: UpdateTaskInput): TaskRecord | undefined;
  archiveTask(id: string): boolean;

  createConversation(input: CreateConversationInput): ConversationRecord;
  getConversation(id: string, includeArchived?: boolean): ConversationRecord | undefined;
  listConversations(taskId: string, options?: ListOptions): ConversationRecord[];
  updateConversation(id: string, patch: UpdateConversationInput): ConversationRecord | undefined;
  archiveConversation(id: string): boolean;

  appendMessage(input: AppendMessageInput): MessageRecord;
  getMessage(id: string, includeArchived?: boolean): MessageRecord | undefined;
  listMessages(conversationId: string, options?: MessageHistoryOptions): MessageRecord[];
  updateMessage(id: string, patch: UpdateMessageInput): MessageRecord | undefined;
  archiveMessage(id: string): boolean;

  close(): void;
}
