/**
 * Electron Preload Script
 *
 * 通过 contextBridge 在 renderer 暴露受限的 OpenintjAPI。
 * 任何 main ↔ renderer 通信都必须通过此层（renderer 不能直接 require electron）。
 */

import { contextBridge, ipcRenderer } from "electron";
import {
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
  type StatusResponse,
} from "../shared/ipc-protocol.js";

const onEvent = (
  channel:
    | typeof IPC.EVT_TAO
    | typeof IPC.EVT_REACT
    | typeof IPC.EVT_AUDIT
    | typeof IPC.EVT_UPDATE,
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
  /** 拿当前 PersonaConfig 快照。 */
  dormantPersona(): Promise<DormantPersonaResponse | DormantError> {
    return ipcRenderer.invoke(IPC.DORMANT_PERSONA);
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
};

contextBridge.exposeInMainWorld("openintj", api);

export type OpenintjAPI = typeof api;
