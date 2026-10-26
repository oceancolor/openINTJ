import type { WorkspacePreviewKind } from "../../shared/workspace-view.js";
import { editorKindForPath } from "../../shared/workspace-view.js";

export interface EditorBuffer {
  path: string;
  kind: WorkspacePreviewKind;
  content: string;
  savedContent: string;
  loading: boolean;
  error?: string;
}

export const bufferDirty = (buffer: EditorBuffer): boolean =>
  buffer.kind === "text" && !buffer.loading && buffer.content !== buffer.savedContent;

export const openBuffer = (
  buffers: readonly EditorBuffer[],
  path: string,
): EditorBuffer[] => {
  if (buffers.some((buffer) => buffer.path === path)) return [...buffers];
  const kind = editorKindForPath(path);
  return [
    ...buffers,
    {
      path,
      kind,
      content: "",
      savedContent: "",
      loading: kind === "text",
    },
  ];
};

export const patchBuffer = (
  buffers: readonly EditorBuffer[],
  path: string,
  patch: Partial<EditorBuffer>,
): EditorBuffer[] =>
  buffers.map((buffer) => (buffer.path === path ? { ...buffer, ...patch } : buffer));

export const closeBuffer = (buffers: readonly EditorBuffer[], path: string): EditorBuffer[] =>
  buffers.filter((buffer) => buffer.path !== path);

export const nextActivePath = (
  buffers: readonly EditorBuffer[],
  closedPath: string,
  current?: string,
): string | undefined => {
  if (current !== closedPath) return current;
  const index = buffers.findIndex((buffer) => buffer.path === closedPath);
  const remaining = buffers.filter((buffer) => buffer.path !== closedPath);
  if (remaining.length === 0) return undefined;
  return remaining[Math.min(Math.max(index, 0), remaining.length - 1)]?.path;
};
