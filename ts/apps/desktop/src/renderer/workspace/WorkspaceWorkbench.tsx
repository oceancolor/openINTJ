import React from "react";
import { FilePreview } from "./FilePreview.js";
import { WorkspaceExplorer } from "./WorkspaceExplorer.js";
import type { WorkspaceClient } from "./types.js";

/**
 * Composable workspace chrome: toolbar + recent-files strip + file tree +
 * optional read-only preview. Writing stays with the Agent `write_file`
 * gate — this surface never edits files.
 */
export const WorkspaceWorkbench: React.FC<{
  client: WorkspaceClient;
  rootLabel?: string | undefined;
  selectedPath?: string | undefined;
  onSelectPath: (path: string) => void;
  showPreview?: boolean;
  refreshKey?: number | undefined;
  workspaceOpen?: boolean | undefined;
}> = ({
  client,
  rootLabel,
  selectedPath,
  onSelectPath,
  showPreview = true,
  refreshKey,
  workspaceOpen = true,
}) => {
  const reveal = (path?: string): void => {
    void client.reveal(path);
  };

  // Recent-files strip: fast switching between the files you actually open.
  const [recents, setRecents] = React.useState<string[]>([]);
  React.useEffect(() => {
    if (!selectedPath) return;
    setRecents((prev) => [selectedPath, ...prev.filter((p) => p !== selectedPath)].slice(0, 8));
  }, [selectedPath]);

  return (
    <div className="h-full min-h-0 flex flex-col bg-canvas" data-testid="workspace-workbench">
      <div className="shrink-0 px-3 py-2 border-b border-line bg-panel flex items-center gap-2 text-xs">
        <span className="truncate font-mono text-[11px] text-muted" title={rootLabel}>
          {workspaceOpen ? (rootLabel ?? "工作区") : "未打开工作区"}
        </span>
        {workspaceOpen ? (
          <button
            type="button"
            className="ml-auto px-2 py-1 rounded-lg border border-line bg-white hover:border-linedark text-ink"
            onClick={() => reveal()}
          >
            打开目录
          </button>
        ) : null}
        {workspaceOpen && selectedPath ? (
          <button
            type="button"
            className="px-2 py-1 rounded-lg border border-line bg-white hover:border-linedark text-ink"
            onClick={() => reveal(selectedPath)}
          >
            定位文件
          </button>
        ) : null}
      </div>
      {workspaceOpen && recents.length > 1 ? (
        <div
          className="shrink-0 px-2 py-1.5 border-b border-line bg-panel flex items-center gap-1 overflow-x-auto"
          data-testid="workspace-recents"
        >
          <span className="shrink-0 text-muted text-[10px] font-bold uppercase tracking-[0.13em]">
            最近
          </span>
          {recents.map((path) => (
            <span
              key={path}
              className={`shrink-0 inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] ${
                path === selectedPath
                  ? "border-transparent bg-mosstint text-mossdeep"
                  : "border-line bg-white text-muted hover:border-linedark"
              }`}
            >
              <button type="button" title={path} onClick={() => onSelectPath(path)}>
                {path.split("/").pop() ?? path}
              </button>
              <button
                type="button"
                aria-label={`从最近中移除 ${path}`}
                className="text-faint hover:text-ink"
                onClick={() => setRecents((prev) => prev.filter((p) => p !== path))}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      ) : null}
      {!workspaceOpen ? (
        <div className="flex-1 min-h-0 text-xs text-muted p-3">
          当前没有打开的工作区。请在上方打开一个目录。
        </div>
      ) : (
        <div className={`flex-1 min-h-0 ${showPreview ? "grid grid-rows-2" : ""}`}>
          <WorkspaceExplorer
            client={client}
            selectedPath={selectedPath}
            refreshKey={refreshKey}
            onSelectPath={(path, type) => {
              if (type === "file") onSelectPath(path);
            }}
          />
          {showPreview && selectedPath ? (
            <div className="min-h-0 border-t border-line">
              <FilePreview client={client} path={selectedPath} onReveal={(path) => reveal(path)} />
            </div>
          ) : showPreview ? (
            <div className="min-h-0 border-t border-line text-xs text-muted p-3 bg-panel">
              选择文件以预览。对话里的工作区路径也可以点开。
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};
