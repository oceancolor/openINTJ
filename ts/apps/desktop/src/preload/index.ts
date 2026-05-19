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
  IPC,
  type MemoryQueryRequest,
  type MemoryQueryResult,
  type StatusResponse,
} from "../shared/ipc-protocol.js";

const onEvent = (
  channel: typeof IPC.EVT_TAO | typeof IPC.EVT_REACT | typeof IPC.EVT_AUDIT,
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
  onTaoEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_TAO, cb);
  },
  onReactEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_REACT, cb);
  },
  onAuditEvent(cb: (payload: unknown) => void): () => void {
    return onEvent(IPC.EVT_AUDIT, cb);
  },
};

contextBridge.exposeInMainWorld("openintj", api);

export type OpenintjAPI = typeof api;
