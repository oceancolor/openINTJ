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
  WorkspaceReadResponse,
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
});
