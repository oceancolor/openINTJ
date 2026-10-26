import React from "react";
import {
  bufferDirty,
  closeBuffer,
  type EditorBuffer,
  nextActivePath,
  openBuffer,
  patchBuffer,
} from "./editor-buffers.js";
import { FilePreview } from "./FilePreview.js";
import { type WorkspaceClient, isWorkspaceError } from "./types.js";
import { WorkspaceExplorer } from "./WorkspaceExplorer.js";

const FileEditor = React.lazy(async () => {
  const mod = await import("./FileEditor.js");
  return { default: mod.FileEditor };
});

const fileName = (path: string): string => path.split("/").pop() ?? path;

export const EditorWorkbench: React.FC<{
  client: WorkspaceClient;
  openPath?: string | undefined;
  rootLabel?: string | undefined;
  refreshKey?: number | undefined;
  workspaceOpen?: boolean | undefined;
  onActivePathChange: (path: string | undefined) => void;
}> = ({
  client,
  openPath,
  rootLabel,
  refreshKey = 0,
  workspaceOpen = true,
  onActivePathChange,
}) => {
  const [buffers, setBuffers] = React.useState<EditorBuffer[]>(() =>
    openPath ? openBuffer([], openPath) : [],
  );
  const [activePath, setActivePath] = React.useState<string | undefined>(openPath);
  const [saving, setSaving] = React.useState(false);

  React.useEffect(() => {
    if (!openPath) return;
    setBuffers((current) => openBuffer(current, openPath));
    setActivePath(openPath);
  }, [openPath]);

  const active = buffers.find((buffer) => buffer.path === activePath);

  React.useEffect(() => {
    if (!active || active.kind !== "text" || !active.loading) return;
    const path = active.path;
    let cancelled = false;
    void client.read(path).then((result) => {
      if (cancelled) return;
      if (isWorkspaceError(result)) {
        setBuffers((current) =>
          patchBuffer(current, path, { loading: false, error: result.message }),
        );
        return;
      }
      setBuffers((current) =>
        patchBuffer(current, path, {
          loading: false,
          content: result.content,
          savedContent: result.content,
          error: undefined,
        }),
      );
    });
    return () => {
      cancelled = true;
    };
  }, [active?.path, active?.kind, active?.loading, client]);

  React.useEffect(() => {
    if (refreshKey === 0) return;
    setBuffers((current) =>
      current.map((buffer) =>
        buffer.kind === "text" && !bufferDirty(buffer)
          ? { ...buffer, loading: true, error: undefined }
          : buffer,
      ),
    );
  }, [refreshKey]);

  const save = React.useCallback(async (): Promise<void> => {
    const buffer = buffers.find((item) => item.path === activePath);
    if (!buffer || buffer.kind !== "text" || buffer.loading) return;
    setSaving(true);
    const result = await client.write(buffer.path, buffer.content);
    setSaving(false);
    if (isWorkspaceError(result)) {
      setBuffers((current) => patchBuffer(current, buffer.path, { error: result.message }));
      return;
    }
    setBuffers((current) =>
      patchBuffer(current, buffer.path, {
        savedContent: buffer.content,
        error: undefined,
      }),
    );
  }, [activePath, buffers, client]);

  React.useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void save();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save]);

  const openFile = (path: string): void => {
    setBuffers((current) => openBuffer(current, path));
    setActivePath(path);
    onActivePathChange(path);
  };

  const requestClose = (path: string): void => {
    const buffer = buffers.find((item) => item.path === path);
    if (buffer && bufferDirty(buffer)) {
      const discard = window.confirm(`${fileName(path)} 有未保存更改。关闭并丢弃？`);
      if (!discard) return;
    }
    const next = nextActivePath(buffers, path, activePath);
    setBuffers((current) => closeBuffer(current, path));
    setActivePath(next);
    onActivePathChange(next);
  };

  return (
    <section
      className="h-full min-h-0 flex flex-col bg-[#11111b]"
      data-testid="workspace-editor"
    >
      <div className="shrink-0 h-9 px-2 border-b border-gray-800 flex items-center gap-2 text-xs min-w-0">
        <span className="truncate text-gray-400" title={rootLabel}>
          {workspaceOpen ? (rootLabel ?? "工作区") : "未打开工作区"}
        </span>
        {workspaceOpen ? (
          <button
            type="button"
            className="ml-auto px-2 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-200"
            onClick={() => void client.reveal()}
          >
            打开目录
          </button>
        ) : null}
      </div>
      {!workspaceOpen ? (
        <div className="flex-1 min-h-0 text-xs text-gray-500 p-3">
          当前没有打开的工作区。请在左侧打开一个目录。
        </div>
      ) : (
      <div className="flex-1 min-h-0 flex">
        <aside className="w-[200px] shrink-0 border-r border-gray-800 min-h-0">
          <WorkspaceExplorer
            client={client}
            selectedPath={activePath}
            refreshKey={refreshKey}
            onSelectPath={(path, type) => {
              if (type === "file") openFile(path);
            }}
          />
        </aside>
        <div className="flex-1 min-w-0 min-h-0 flex flex-col">
          <div className="shrink-0 h-9 border-b border-gray-800 flex items-center min-w-0">
            <div className="flex-1 min-w-0 h-full overflow-x-auto flex items-stretch">
              {buffers.map((buffer) => {
                const selected = buffer.path === activePath;
                const dirty = bufferDirty(buffer);
                return (
                  <button
                    key={buffer.path}
                    type="button"
                    title={buffer.path}
                    className={`shrink-0 px-2 text-xs border-r border-gray-800 flex items-center gap-1 ${
                      selected
                        ? "bg-[#1e1e2e] text-gray-100"
                        : "text-gray-400 hover:text-gray-200"
                    }`}
                    onClick={() => openFile(buffer.path)}
                  >
                    <span className="max-w-[10rem] truncate">{fileName(buffer.path)}</span>
                    {dirty ? <span className="text-yellow-400">●</span> : null}
                    <span
                      role="button"
                      tabIndex={0}
                      className="text-gray-500 hover:text-gray-200 px-0.5"
                      onClick={(event) => {
                        event.stopPropagation();
                        requestClose(buffer.path);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.stopPropagation();
                          requestClose(buffer.path);
                        }
                      }}
                    >
                      ×
                    </span>
                  </button>
                );
              })}
            </div>
            <div className="shrink-0 flex items-center gap-2 px-2 text-xs">
              {active?.kind === "text" ? (
                <button
                  type="button"
                  className="px-2 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-200 disabled:opacity-40"
                  disabled={saving || !active || !bufferDirty(active)}
                  onClick={() => void save()}
                >
                  {saving ? "保存中" : "保存"}
                </button>
              ) : null}
              {activePath ? (
                <button
                  type="button"
                  className="text-gray-400 hover:text-gray-200"
                  onClick={() => void client.reveal(activePath)}
                >
                  打开所在目录
                </button>
              ) : null}
            </div>
          </div>
          <div className="flex-1 min-h-0">
            {!active ? (
              <div className="text-xs text-gray-500 p-3">
                从左侧目录树选择文件以查看或编辑。
              </div>
            ) : active.error ? (
              <div className="text-xs text-red-300 p-3">{active.error}</div>
            ) : active.loading ? (
              <div className="text-xs text-gray-500 p-3">加载中…</div>
            ) : active.kind === "text" ? (
              <React.Suspense
                fallback={<div className="text-xs text-gray-500 p-3">加载编辑器…</div>}
              >
                <FileEditor
                  key={active.path}
                  path={active.path}
                  value={active.content}
                  onChange={(content) =>
                    setBuffers((current) => patchBuffer(current, active.path, { content }))
                  }
                />
              </React.Suspense>
            ) : active.kind === "image" ? (
              <FilePreview client={client} path={active.path} showHeader={false} />
            ) : (
              <div className="text-xs text-gray-400 p-3">
                此类型暂不在编辑器中打开，请用「打开所在目录」查看。
              </div>
            )}
          </div>
        </div>
      </div>
      )}
    </section>
  );
};
