/**
 * Electron Preload Script
 *
 * 通过 contextBridge 在 renderer 暴露受限的 OpenintjAPI。
 * 任何 main ↔ renderer 通信都必须通过此层（renderer 不能直接 require electron）。
 */

import { contextBridge, ipcRenderer } from "electron";
import {
  type AgentCancelRequest,
  type AgentSteerRequest,
  type AppConfig,
  type AppConfigPatch,
  type ApprovalDecideRequest,
  type ApprovalRequestView,
  type CapabilityGrantView,
  type ChannelStatusView,
  type ChatRequest,
  type ChatResponse,
  type CommandsInvokeRequest,
  type CommandsInvokeResponse,
  type ConnectorStatusView,
  type DormantDecisionError,
  type DormantDecisionResponse,
  type DormantError,
  type DormantListRequest,
  type DormantListResponse,
  type DormantMineResponse,
  type DormantPersonaResponse,
  type DormantProposalDecision,
  type GrantCreateRequest,
  IPC,
  type IdentityAssertRequest,
  type IdentityConfirmRequest,
  type IdentityPersonView,
  type IdentityResolveRequest,
  type IdentityResolveResultView,
  type LinkedAccountView,
  type MemoryQueryRequest,
  type MemoryQueryResult,
  type ModelCredentialSet,
  type ModelProfile,
  type ProfileEntryView,
  type ProfileUpsertRequest,
  type ScheduleAddRequest,
  type ScheduleUpdateRequest,
  type ScheduledJobView,
  type SkillActiveResponse,
  type SkillCatalogResponse,
  type SkillDecisionResponse,
  type SkillDistillResponse,
  type SkillLearningError,
  type SkillListRequest,
  type SkillListResponse,
  type SkillProposalDecision,
  type OllamaCheckResponse,
  type StatusResponse,
  type SubagentRecordView,
  type SyncApplyReportView,
  type SyncBundleView,
  type TaskCheckpointView,
  type TaskRunResultView,
  type VoiceTranscribeRequest,
  type WindowChromeAction,
  type WorkbenchConversation,
  type WorkbenchMessage,
  type WorkbenchTask,
  type WorkbenchWorkspace,
  type WorkspaceBytesRequest,
  type WorkspaceBytesResponse,
  type WorkspaceError,
  type WorkspaceInfo,
  type WorkspaceListResponse,
  type WorkspaceMkdirRequest,
  type WorkspaceMkdirResponse,
  type WorkspacePickResponse,
  type WorkspaceReadRequest,
  type WorkspaceReadResponse,
  type WorkspaceRenameDirRequest,
  type WorkspaceRenameDirResponse,
  type WorkspaceRevealRequest,
  type WorkspaceRevealResponse,
  type WorkspaceWriteRequest,
  type WorkspaceWriteResponse,
} from "../shared/ipc-protocol.js";

const onEvent = (
  channel:
    | typeof IPC.EVT_TAO
    | typeof IPC.EVT_REACT
    | typeof IPC.EVT_AUDIT
    | typeof IPC.EVT_UPDATE
    | typeof IPC.EVT_WORKSPACE
    | typeof IPC.EVT_SKILL
    | typeof IPC.EVT_ASSISTANT_DELTA
    | typeof IPC.EVT_APPROVAL
    | typeof IPC.EVT_SCHEDULE,
  cb: (payload: unknown) => void,
): (() => void) => {
  const listener = (_evt: unknown, payload: unknown): void => cb(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
};

const api = {
  hostPlatform: process.platform,
  ping(): Promise<{ ok: boolean; ts: number }> {
    return ipcRenderer.invoke(IPC.PING);
  },
  status(): Promise<StatusResponse> {
    return ipcRenderer.invoke(IPC.STATUS);
  },
  /** Ollama 自检：服务是否在跑、embedding 模型是否已安装。 */
  ollamaCheck(): Promise<OllamaCheckResponse> {
    return ipcRenderer.invoke(IPC.OLLAMA_CHECK);
  },
  /** 尝试拉起本机 Ollama 并复检；成功后主进程会热重装 agent 切回真实 embedding。 */
  ollamaLaunch(): Promise<OllamaCheckResponse> {
    return ipcRenderer.invoke(IPC.OLLAMA_LAUNCH);
  },
  chat(req: ChatRequest): Promise<ChatResponse> {
    return ipcRenderer.invoke(IPC.CHAT, req);
  },
  invokeCommand(req: CommandsInvokeRequest): Promise<CommandsInvokeResponse | { error: string }> {
    return ipcRenderer.invoke(IPC.COMMANDS_INVOKE, req);
  },
  steer(req: AgentSteerRequest): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.AGENT_STEER, req);
  },
  inject(req: AgentSteerRequest): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.AGENT_INJECT, req);
  },
  cancelAgent(req: AgentCancelRequest): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.AGENT_CANCEL, req);
  },
  onAssistantDelta(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_ASSISTANT_DELTA, cb);
  },
  // RFC-013 interactive approval + capability grants
  approvalList(): Promise<ApprovalRequestView[]> {
    return ipcRenderer.invoke(IPC.APPROVAL_LIST);
  },
  approvalDecide(req: ApprovalDecideRequest): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.APPROVAL_DECIDE, req);
  },
  grantsList(): Promise<CapabilityGrantView[]> {
    return ipcRenderer.invoke(IPC.GRANTS_LIST);
  },
  grantsGrant(req: GrantCreateRequest): Promise<CapabilityGrantView | { error: string }> {
    return ipcRenderer.invoke(IPC.GRANTS_GRANT, req);
  },
  grantsRevoke(id: string): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.GRANTS_REVOKE, id);
  },
  onApprovalEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_APPROVAL, cb);
  },
  // RFC-014 personal profile (memory control plane)
  profileList(): Promise<ProfileEntryView[]> {
    return ipcRenderer.invoke(IPC.PROFILE_LIST);
  },
  profileUpsert(req: ProfileUpsertRequest): Promise<ProfileEntryView | { error: string }> {
    return ipcRenderer.invoke(IPC.PROFILE_UPSERT, req);
  },
  profileRemove(id: string): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.PROFILE_REMOVE, id);
  },
  profileExport(): Promise<{ profile: string }> {
    return ipcRenderer.invoke(IPC.PROFILE_EXPORT);
  },
  // RFC-015 scheduler (persistent jobs + notifications)
  scheduleList(): Promise<ScheduledJobView[]> {
    return ipcRenderer.invoke(IPC.SCHEDULE_LIST);
  },
  scheduleAdd(req: ScheduleAddRequest): Promise<ScheduledJobView | { error: string }> {
    return ipcRenderer.invoke(IPC.SCHEDULE_ADD, req);
  },
  scheduleUpdate(
    id: string,
    patch: ScheduleUpdateRequest,
  ): Promise<ScheduledJobView | { error: string }> {
    return ipcRenderer.invoke(IPC.SCHEDULE_UPDATE, { id, patch });
  },
  scheduleRemove(id: string): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.SCHEDULE_REMOVE, id);
  },
  onScheduleEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_SCHEDULE, cb);
  },
  // RFC-020 identity graph (proposal-confirm account linking)
  identityPersons(): Promise<IdentityPersonView[]> {
    return ipcRenderer.invoke(IPC.IDENTITY_PERSONS);
  },
  identityResolve(
    req: IdentityResolveRequest,
  ): Promise<IdentityResolveResultView | { error: string }> {
    return ipcRenderer.invoke(IPC.IDENTITY_RESOLVE, req);
  },
  identityConfirm(req: IdentityConfirmRequest): Promise<
    | {
        persons: IdentityPersonView[];
        linked: LinkedAccountView[];
        skipped: string[];
      }
    | { error: string }
  > {
    return ipcRenderer.invoke(IPC.IDENTITY_CONFIRM, req);
  },
  identityAssert(req: IdentityAssertRequest): Promise<IdentityPersonView | { error: string }> {
    return ipcRenderer.invoke(IPC.IDENTITY_ASSERT, req);
  },
  identityUnlink(
    accountId: string,
  ): Promise<{ removed: boolean; personRemoved: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.IDENTITY_UNLINK, { accountId });
  },
  // RFC-016 connectors (status surface)
  connectorsList(): Promise<ConnectorStatusView[]> {
    return ipcRenderer.invoke(IPC.CONNECTORS_LIST);
  },
  // RFC-012 TaskPool checkpoints
  taskpoolCheckpoints(
    ...args: [limit?: number]
  ): Promise<TaskCheckpointView[] | { error: string }> {
    return ipcRenderer.invoke(IPC.TASKPOOL_CHECKPOINTS, args[0]);
  },
  taskpoolResume(runId: string): Promise<TaskRunResultView | { error: string }> {
    return ipcRenderer.invoke(IPC.TASKPOOL_RESUME, runId);
  },
  taskpoolCancel(runId: string): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.TASKPOOL_CANCEL, runId);
  },
  // RFC-012 §3 subagents
  subagentsList(): Promise<SubagentRecordView[]> {
    return ipcRenderer.invoke(IPC.SUBAGENTS_LIST);
  },
  // RFC-018 outbound channels
  channelsList(): Promise<ChannelStatusView[]> {
    return ipcRenderer.invoke(IPC.CHANNELS_LIST);
  },
  channelsTest(id: string): Promise<{ ok: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.CHANNELS_TEST, id);
  },
  // RFC-019 cross-device sync + voice
  syncExport(): Promise<SyncBundleView> {
    return ipcRenderer.invoke(IPC.SYNC_EXPORT);
  },
  syncImport(bundle: SyncBundleView): Promise<{ report: SyncApplyReportView } | { error: string }> {
    return ipcRenderer.invoke(IPC.SYNC_IMPORT, bundle);
  },
  voiceTranscribe(
    req: VoiceTranscribeRequest,
  ): Promise<{ text: string; durationMs?: number } | { error: string }> {
    return ipcRenderer.invoke(IPC.VOICE_TRANSCRIBE, req);
  },
  memoryQuery(req: MemoryQueryRequest): Promise<MemoryQueryResult[]> {
    return ipcRenderer.invoke(IPC.MEMORY_QUERY, req);
  },
  // G5.2 记忆审核台
  memoryStatus(): Promise<unknown> {
    return ipcRenderer.invoke(IPC.MEMORY_STATUS);
  },
  memoryCandidates(): Promise<unknown[]> {
    return ipcRenderer.invoke(IPC.MEMORY_CANDIDATES);
  },
  memoryCandidateApprove(
    candidateId: string,
  ): Promise<{ candidateId: string; status: string; fragmentId?: string } | { error: string }> {
    return ipcRenderer.invoke(IPC.MEMORY_CANDIDATE_APPROVE, { candidateId });
  },
  memoryCandidateReject(
    candidateId: string,
  ): Promise<{ candidateId: string; status: string; fragmentId?: string } | { error: string }> {
    return ipcRenderer.invoke(IPC.MEMORY_CANDIDATE_REJECT, { candidateId });
  },
  // G5.3 技能反馈（对该会话最近一轮回答）
  skillsFeedback(req: {
    conversationId: string;
    sentiment: "positive" | "negative";
  }): Promise<{ applied: boolean } | { error: string }> {
    return ipcRenderer.invoke(IPC.SKILLS_FEEDBACK, req);
  },
  auditRecent(limit?: number): Promise<{
    stats: { totalEvents: number; blockedCount: number };
    recent: unknown[];
  }> {
    return ipcRenderer.invoke(IPC.AUDIT_RECENT, limit);
  },
  // ---------- Dormant Memory Learning (Phase 3.5 #9.B) ----------
  /** 触发一次 mine：分析 PassiveStore → 产出 proposals。 */
  dormantMine(): Promise<DormantMineResponse | DormantError> {
    return ipcRenderer.invoke(IPC.DORMANT_MINE);
  },
  /** 列出 proposals；不传 status 时返回所有状态。 */
  dormantList(req?: DormantListRequest): Promise<DormantListResponse | DormantError> {
    return ipcRenderer.invoke(IPC.DORMANT_LIST, req ?? {});
  },
  /** 批准一条 proposal → 写入 PersonaConfig；返回 status='applied'。 */
  dormantApprove(
    req: DormantProposalDecision,
  ): Promise<DormantDecisionResponse | DormantDecisionError> {
    return ipcRenderer.invoke(IPC.DORMANT_APPROVE, req);
  },
  /** 拒绝一条 proposal；不污染 PersonaConfig；返回 status='rejected'。 */
  dormantReject(
    req: DormantProposalDecision,
  ): Promise<DormantDecisionResponse | DormantDecisionError> {
    return ipcRenderer.invoke(IPC.DORMANT_REJECT, req);
  },
  /** 撤销一条已批准（applied）的 persona 条目 → 从 PersonaConfig 删除；返回 status='revoked'。 */
  dormantRevoke(
    req: DormantProposalDecision,
  ): Promise<DormantDecisionResponse | DormantDecisionError> {
    return ipcRenderer.invoke(IPC.DORMANT_REVOKE, req);
  },
  /** 拿当前 PersonaConfig 快照。 */
  dormantPersona(): Promise<DormantPersonaResponse | DormantError> {
    return ipcRenderer.invoke(IPC.DORMANT_PERSONA);
  },
  // ---------- 技能自学习 (Phase 2) ----------
  /** 触发一次蒸馏：成功轨迹 → 候选技能 pending 提案。 */
  skillsDistill(): Promise<SkillDistillResponse | SkillLearningError> {
    return ipcRenderer.invoke(IPC.SKILLS_DISTILL);
  },
  /** 列出技能提案；不传 status 返回所有状态。 */
  skillsList(req?: SkillListRequest): Promise<SkillListResponse | SkillLearningError> {
    return ipcRenderer.invoke(IPC.SKILLS_LIST, req ?? {});
  },
  /** 批准一条提案 → 写入生效技能并重载注册表。 */
  skillsApprove(req: SkillProposalDecision): Promise<SkillDecisionResponse | SkillLearningError> {
    return ipcRenderer.invoke(IPC.SKILLS_APPROVE, req);
  },
  /** 拒绝一条提案。 */
  skillsReject(req: SkillProposalDecision): Promise<SkillDecisionResponse | SkillLearningError> {
    return ipcRenderer.invoke(IPC.SKILLS_REJECT, req);
  },
  /** 撤销一条已批准技能 → 从生效集移除并重载注册表。 */
  skillsRevoke(req: SkillProposalDecision): Promise<SkillDecisionResponse | SkillLearningError> {
    return ipcRenderer.invoke(IPC.SKILLS_REVOKE, req);
  },
  /** 当前生效的学习技能 + 权重。 */
  skillsActive(): Promise<SkillActiveResponse | SkillLearningError> {
    return ipcRenderer.invoke(IPC.SKILLS_ACTIVE);
  },
  /** 已装技能目录（作者包 + 已批准学习技能）。技能系统关闭时 total=0。 */
  skillsCatalog(): Promise<SkillCatalogResponse> {
    return ipcRenderer.invoke(IPC.SKILLS_CATALOG);
  },
  onSkillEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_SKILL, cb);
  },
  onTaoEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_TAO, cb);
  },
  onReactEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_REACT, cb);
  },
  onAuditEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_AUDIT, cb);
  },
  // ---------- 自动更新 (#6) ----------
  /** 主动触发一次更新检查。 */
  updateCheck(): Promise<{ ok: boolean; reason?: string }> {
    return ipcRenderer.invoke(IPC.UPDATE_CHECK);
  },
  /** 退出并安装已下载的更新。 */
  updateInstall(): Promise<{ ok: boolean; reason?: string }> {
    return ipcRenderer.invoke(IPC.UPDATE_INSTALL);
  },
  /** 订阅更新状态事件（checking / available / downloading / downloaded / error）。 */
  onUpdateEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_UPDATE, cb);
  },
  // ---------- 工作区系统能力面 (RFC-004 §8) ----------
  /** 当前工作区配置（根目录 / 命令开关 / 白名单）。 */
  workspaceInfo(): Promise<WorkspaceInfo> {
    return ipcRenderer.invoke(IPC.WORKSPACE_INFO);
  },
  /** 读取工作区内文件（path 相对工作区根，越界 / 过大会被拒绝）。 */
  workspaceRead(req: WorkspaceReadRequest): Promise<WorkspaceReadResponse | WorkspaceError> {
    return ipcRenderer.invoke(IPC.WORKSPACE_READ, req);
  },
  /** 写入工作区内文件。 */
  workspaceWrite(req: WorkspaceWriteRequest): Promise<WorkspaceWriteResponse | WorkspaceError> {
    return ipcRenderer.invoke(IPC.WORKSPACE_WRITE, req);
  },
  /** 弹出系统目录选择对话框，选定新的工作区根。 */
  workspacePickDir(): Promise<WorkspacePickResponse> {
    return ipcRenderer.invoke(IPC.WORKSPACE_PICK_DIR);
  },
  /** 列出工作区文件树（沙箱内、有深度/数量上限）。 */
  workspaceList(): Promise<WorkspaceListResponse | WorkspaceError> {
    return ipcRenderer.invoke(IPC.WORKSPACE_LIST);
  },
  /** 读取工作区二进制文件（预览图片用 base64）。 */
  workspaceReadBytes(req: WorkspaceBytesRequest): Promise<WorkspaceBytesResponse | WorkspaceError> {
    return ipcRenderer.invoke(IPC.WORKSPACE_READ_BYTES, req);
  },
  /** 在系统文件管理器中打开工作区根或定位某个文件。 */
  workspaceReveal(req?: WorkspaceRevealRequest): Promise<WorkspaceRevealResponse | WorkspaceError> {
    return ipcRenderer.invoke(IPC.WORKSPACE_REVEAL, req ?? {});
  },
  /** 在工作区内建目录（任务同名文件工作区）。 */
  workspaceMkdir(
    req: WorkspaceMkdirRequest,
  ): Promise<WorkspaceMkdirResponse | WorkspaceError> {
    return ipcRenderer.invoke(IPC.WORKSPACE_MKDIR, req);
  },
  /** 工作区内目录改名（任务改名同步）。 */
  workspaceRenameDir(
    req: WorkspaceRenameDirRequest,
  ): Promise<WorkspaceRenameDirResponse | WorkspaceError> {
    return ipcRenderer.invoke(IPC.WORKSPACE_RENAME_DIR, req);
  },
  /** 订阅工作区文件变更事件（fs.watch）。 */
  onWorkspaceEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_WORKSPACE, cb);
  },
  // ---------- 应用配置面 ----------
  /** 读取持久化的应用配置。 */
  getConfig(): Promise<AppConfig> {
    return ipcRenderer.invoke(IPC.CONFIG_GET);
  },
  /** 浅合并更新应用配置并持久化，返回合并后的完整配置。 */
  updateConfig(patch: AppConfigPatch): Promise<AppConfig | { error: string }> {
    return ipcRenderer.invoke(IPC.CONFIG_UPDATE, patch);
  },
  /** Gracefully close stores and relaunch the desktop process. */
  restartApp(): Promise<{ ok: boolean; reason?: string }> {
    return ipcRenderer.invoke(IPC.APP_RESTART);
  },
  windowChrome(action: WindowChromeAction): Promise<{ ok: boolean }> {
    return ipcRenderer.invoke(IPC.WINDOW_CHROME, action);
  },
  modelProfiles(): Promise<ModelProfile[]> {
    return ipcRenderer.invoke(IPC.MODEL_PROFILES);
  },
  setModelCredential(req: ModelCredentialSet): Promise<{ ok: boolean; error?: string }> {
    return ipcRenderer.invoke(IPC.MODEL_CREDENTIAL_SET, req);
  },
  deleteModelCredential(
    profileId: string,
  ): Promise<{ ok: boolean; deleted?: boolean; error?: string }> {
    return ipcRenderer.invoke(IPC.MODEL_CREDENTIAL_DELETE, profileId);
  },
  testModelProfile(
    profileId: string,
  ): Promise<{ ok: boolean; provider?: string; model?: string; error?: string }> {
    return ipcRenderer.invoke(IPC.MODEL_TEST, profileId);
  },
  workbenchBootstrap(): Promise<{
    workspaces: WorkbenchWorkspace[];
    tasks: WorkbenchTask[];
    conversations: WorkbenchConversation[];
  }> {
    return ipcRenderer.invoke(IPC.WORKBENCH_BOOTSTRAP);
  },
  createWorkbenchWorkspace(req: {
    name: string;
    rootPath: string;
  }): Promise<WorkbenchWorkspace> {
    return ipcRenderer.invoke(IPC.WORKBENCH_WORKSPACE_CREATE, req);
  },
  updateWorkbenchWorkspace(req: {
    id: string;
    name: string;
  }): Promise<WorkbenchWorkspace> {
    return ipcRenderer.invoke(IPC.WORKBENCH_WORKSPACE_UPDATE, req);
  },
  closeWorkbenchWorkspace(req: { id: string }): Promise<WorkbenchWorkspace> {
    return ipcRenderer.invoke(IPC.WORKBENCH_WORKSPACE_CLOSE, req);
  },
  createWorkbenchTask(req: {
    parentId: string;
    title: string;
  }): Promise<WorkbenchTask> {
    return ipcRenderer.invoke(IPC.WORKBENCH_TASK_CREATE, req);
  },
  updateWorkbenchTask(req: {
    id: string;
    title?: string;
    status?: "active" | "completed" | "archived";
    taskPoolRunId?: string;
  }): Promise<WorkbenchTask> {
    return ipcRenderer.invoke(IPC.WORKBENCH_TASK_UPDATE, req);
  },
  deleteWorkbenchTask(req: { id: string }): Promise<{ ok: boolean; error?: string }> {
    return ipcRenderer.invoke(IPC.WORKBENCH_TASK_DELETE, req);
  },
  createWorkbenchConversation(req: {
    parentId: string;
    title: string;
    modelProfileId?: string;
  }): Promise<WorkbenchConversation> {
    return ipcRenderer.invoke(IPC.WORKBENCH_CONVERSATION_CREATE, req);
  },
  updateWorkbenchConversation(req: {
    id: string;
    title?: string;
    modelProfileId?: string;
  }): Promise<WorkbenchConversation> {
    return ipcRenderer.invoke(IPC.WORKBENCH_CONVERSATION_UPDATE, req);
  },
  deleteWorkbenchConversation(req: { id: string }): Promise<{ ok: boolean; error?: string }> {
    return ipcRenderer.invoke(IPC.WORKBENCH_CONVERSATION_DELETE, req);
  },
  workbenchMessages(conversationId: string): Promise<WorkbenchMessage[]> {
    return ipcRenderer.invoke(IPC.WORKBENCH_MESSAGES, conversationId);
  },
  clearWorkbenchMessages(conversationId: string): Promise<{ ok: boolean; error?: string }> {
    return ipcRenderer.invoke(IPC.WORKBENCH_MESSAGES_CLEAR, conversationId);
  },
};

contextBridge.exposeInMainWorld("openintj", api);

export type OpenintjAPI = typeof api;
