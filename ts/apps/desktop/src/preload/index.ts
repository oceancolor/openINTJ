/**
 * Electron Preload Script
 *
 * 通过 contextBridge 在 renderer 暴露受限的 OpenintjAPI。
 * 任何 main ↔ renderer 通信都必须通过此层（renderer 不能直接 require electron）。
 */

import { contextBridge, ipcRenderer } from "electron";
import {
  type AppConfig,
  type AppConfigPatch,
  type ChatRequest,
  type ChatResponse,
  type DormantDecisionError,
  type DormantDecisionResponse,
  type DormantError,
  type DormantListRequest,
  type DormantListResponse,
  type DormantMineResponse,
  type DormantPersonaResponse,
  type DormantProposalDecision,
  IPC,
  type MemoryQueryRequest,
  type MemoryQueryResult,
  type ModelCredentialSet,
  type ModelProfile,
  type SkillActiveResponse,
  type SkillDecisionResponse,
  type SkillDistillResponse,
  type SkillLearningError,
  type SkillListRequest,
  type SkillListResponse,
  type SkillProposalDecision,
  type StatusResponse,
  type WorkbenchConversation,
  type WorkbenchMessage,
  type WorkbenchTask,
  type WorkbenchWorkspace,
  type WorkspaceError,
  type WorkspaceInfo,
  type WorkspacePickResponse,
  type WorkspaceReadRequest,
  type WorkspaceReadResponse,
  type WorkspaceWriteRequest,
  type WorkspaceWriteResponse,
} from "../shared/ipc-protocol.js";

const onEvent = (
  channel:
    | typeof IPC.EVT_TAO
    | typeof IPC.EVT_REACT
    | typeof IPC.EVT_AUDIT
    | typeof IPC.EVT_UPDATE
    | typeof IPC.EVT_WORKSPACE,
  cb: (payload: unknown) => void,
): (() => void) => {
  const listener = (_evt: unknown, payload: unknown): void => cb(payload);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
};

const api = {
  ping(): Promise<{ ok: boolean; ts: number }> {
    return ipcRenderer.invoke(IPC.PING);
  },
  status(): Promise<StatusResponse> {
    return ipcRenderer.invoke(IPC.STATUS);
  },
  chat(req: ChatRequest): Promise<ChatResponse> {
    return ipcRenderer.invoke(IPC.CHAT, req);
  },
  memoryQuery(req: MemoryQueryRequest): Promise<MemoryQueryResult[]> {
    return ipcRenderer.invoke(IPC.MEMORY_QUERY, req);
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
  workbenchMessages(conversationId: string): Promise<WorkbenchMessage[]> {
    return ipcRenderer.invoke(IPC.WORKBENCH_MESSAGES, conversationId);
  },
};

contextBridge.exposeInMainWorld("openintj", api);

export type OpenintjAPI = typeof api;
