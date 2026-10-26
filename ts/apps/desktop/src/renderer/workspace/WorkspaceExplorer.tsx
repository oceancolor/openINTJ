import React from "react";
import type { WorkspaceTreeNode } from "../../shared/workspace-view.js";
import { buildWorkspaceTree } from "../../shared/workspace-view.js";
import { type WorkspaceClient, isWorkspaceError } from "./types.js";

const TreeNode: React.FC<{
  node: WorkspaceTreeNode;
  depth: number;
  selectedPath?: string | undefined;
  onSelect: (path: string, type: "file" | "dir") => void;
}> = ({ node, depth, selectedPath, onSelect }) => {
  const [open, setOpen] = React.useState(node.path.startsWith("artifacts") || depth < 1);
  const selected = selectedPath === node.path;
  if (node.type === "file") {
    return (
      <button
        type="button"
        className={`block w-full text-left truncate rounded px-1.5 py-0.5 ${
          selected ? "bg-cyan-900 text-cyan-100" : "text-gray-300 hover:bg-gray-800"
        }`}
        style={{ paddingLeft: 8 + depth * 12 }}
        onClick={() => onSelect(node.path, "file")}
        title={node.path}
      >
        {node.name}
      </button>
    );
  }
  return (
    <div>
      <button
        type="button"
        className={`block w-full text-left truncate rounded px-1.5 py-0.5 ${
          selected ? "bg-gray-800 text-gray-100" : "text-gray-400 hover:bg-gray-800"
        }`}
        style={{ paddingLeft: 8 + depth * 12 }}
        onClick={() => {
          setOpen((value) => !value);
          onSelect(node.path, "dir");
        }}
        title={node.path}
      >
        {open ? "▾" : "▸"} {node.name}
      </button>
      {open
        ? (node.children ?? []).map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ))
        : null}
    </div>
  );
};

export const WorkspaceExplorer: React.FC<{
  client: WorkspaceClient;
  selectedPath?: string | undefined;
  onSelectPath: (path: string, type: "file" | "dir") => void;
  refreshKey?: number | undefined;
}> = ({ client, selectedPath, onSelectPath, refreshKey = 0 }) => {
  const [nodes, setNodes] = React.useState<WorkspaceTreeNode[]>([]);
  const [error, setError] = React.useState<string>();

  React.useEffect(() => {
    let cancelled = false;
    void client.list().then((result) => {
      if (cancelled) return;
      if (isWorkspaceError(result)) {
        setError(result.message);
        setNodes([]);
        return;
      }
      setError(undefined);
      setNodes(buildWorkspaceTree(result.entries));
    });
    return () => {
      cancelled = true;
    };
  }, [client, refreshKey]);

  return (
    <div className="h-full min-h-0 overflow-y-auto text-xs" data-testid="workspace-explorer">
      {error ? <div className="px-2 py-1 text-red-300">{error}</div> : null}
      {nodes.length === 0 && !error ? (
        <div className="px-2 py-3 text-gray-500">工作区还没有文件。</div>
      ) : (
        nodes.map((node) => (
          <TreeNode
            key={node.path}
            node={node}
            depth={0}
            selectedPath={selectedPath}
            onSelect={onSelectPath}
          />
        ))
      )}
    </div>
  );
};
