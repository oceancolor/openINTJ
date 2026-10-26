import React from "react";
import { FilePreview } from "./FilePreview.js";
import { WorkspaceExplorer } from "./WorkspaceExplorer.js";
import type { WorkspaceClient } from "./types.js";

/**
 * Composable workspace chrome: toolbar + file tree + optional preview.
 * Desktop hosts this in the right drawer; an IDE shell can put the explorer
 * on the left and FilePreview in the editor pane without rewriting the pieces.
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

  return (
    <div className="h-full min-h-0 flex flex-col" data-testid="workspace-workbench">
      <div className="shrink-0 px-3 py-2 border-b border-gray-800 flex items-center gap-2 text-xs">
        <span className="truncate text-gray-400" title={rootLabel}>
          {workspaceOpen ? (rootLabel ?? "工作区") : "未打开工作区"}
        </span>
        {workspaceOpen ? (
          <button
            type="button"
            className="ml-auto px-2 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-200"
            onClick={() => reveal()}
          >
            打开目录
          </button>
        ) : null}
        {workspaceOpen && selectedPath ? (
          <button
            type="button"
            className="px-2 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-200"
            onClick={() => reveal(selectedPath)}
          >
            定位文件
          </button>
        ) : null}
      </div>
      {!workspaceOpen ? (
        <div className="flex-1 min-h-0 text-xs text-gray-500 p-3">
          当前没有打开的工作区。请在左侧打开一个目录。
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
            <div className="min-h-0 border-t border-gray-800">
              <FilePreview
                client={client}
                path={selectedPath}
                onReveal={(path) => reveal(path)}
              />
            </div>
          ) : showPreview ? (
            <div className="min-h-0 border-t border-gray-800 text-xs text-gray-500 p-3">
              选择文件以预览。对话里的工作区路径也可以点开。
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};
