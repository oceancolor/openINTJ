/**
 * Host-agnostic workspace UI contract.
 *
 * Desktop chat and a future IDE shell both inject this client so explorer /
 * preview components never import Electron or `window.openintj` directly.
 */
import type {
  WorkspaceBytesResponse,
  WorkspaceError,
  WorkspaceInfo,
  WorkspaceListResponse,
  WorkspaceMkdirResponse,
  WorkspaceReadResponse,
  WorkspaceRenameDirResponse,
  WorkspaceRevealResponse,
  WorkspaceWriteResponse,
} from "../../shared/ipc-protocol.js";

export interface WorkspaceClient {
  info(): Promise<WorkspaceInfo>;
  list(): Promise<WorkspaceListResponse | WorkspaceError>;
  read(path: string): Promise<WorkspaceReadResponse | WorkspaceError>;
  readBytes(path: string): Promise<WorkspaceBytesResponse | WorkspaceError>;
  write(path: string, content: string): Promise<WorkspaceWriteResponse | WorkspaceError>;
  reveal(path?: string): Promise<WorkspaceRevealResponse | WorkspaceError>;
  mkdir(path: string): Promise<WorkspaceMkdirResponse | WorkspaceError>;
  renameDir(from: string, to: string): Promise<WorkspaceRenameDirResponse | WorkspaceError>;
  /**
   * 订阅工作区文件变更（桌面端来自主进程 fs.watch）。
   * 可选：无订阅时调用方仍可通过 refreshKey / 对话结束强制重拉。
   */
  onChange?(cb: (payload: { event: string; path: string }) => void): () => void;
}

export const isWorkspaceError = (value: unknown): value is WorkspaceError =>
  typeof value === "object" && value !== null && "error" in value;

export const desktopWorkspaceClient = (): WorkspaceClient => ({
  info: () => window.openintj.workspaceInfo(),
  list: () => window.openintj.workspaceList(),
  read: (path) => window.openintj.workspaceRead({ path }),
  readBytes: (path) => window.openintj.workspaceReadBytes({ path }),
  write: (path, content) => window.openintj.workspaceWrite({ path, content }),
  reveal: (path) => window.openintj.workspaceReveal(path ? { path } : {}),
  mkdir: (path) => window.openintj.workspaceMkdir({ path }),
  renameDir: (from, to) => window.openintj.workspaceRenameDir({ from, to }),
  onChange: (cb) =>
    window.openintj.onWorkspaceEvent((raw) => {
      const p = (raw ?? {}) as { event?: string; path?: string };
      cb({ event: p.event ?? "change", path: p.path ?? "" });
    }),
});
