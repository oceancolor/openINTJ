import React from "react";
import type { WorkspaceEntry, WorkspaceTreeNode } from "../../shared/workspace-view.js";
import { buildWorkspaceTree, filterWorkspaceFiles } from "../../shared/workspace-view.js";
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
        className={`block w-full text-left truncate rounded-md px-1.5 py-1 ${
          selected
            ? "bg-white font-medium text-ink shadow-[inset_3px_0_0_#ca5b3f]"
            : "text-muted hover:bg-white/70 hover:text-ink"
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
        className={`block w-full text-left truncate rounded-md px-1.5 py-1 ${
          selected ? "bg-white/80 font-medium text-ink" : "text-muted hover:bg-white/70"
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
  const [entries, setEntries] = React.useState<WorkspaceEntry[]>([]);
  const [nodes, setNodes] = React.useState<WorkspaceTreeNode[]>([]);
  const [error, setError] = React.useState<string>();
  const [query, setQuery] = React.useState("");

  React.useEffect(() => {
    let cancelled = false;
    void client.list().then((result) => {
      if (cancelled) return;
      if (isWorkspaceError(result)) {
        setError(result.message);
        setEntries([]);
        setNodes([]);
        return;
      }
      setError(undefined);
      setEntries(result.entries);
      setNodes(buildWorkspaceTree(result.entries));
    });
    return () => {
      cancelled = true;
    };
  }, [client, refreshKey]);

  const matches = React.useMemo(() => filterWorkspaceFiles(entries, query), [entries, query]);
  const filtering = query.trim().length > 0;

  return (
    <div
      className="h-full min-h-0 flex flex-col bg-rail/70 text-xs"
      data-testid="workspace-explorer"
    >
      <div className="shrink-0 p-2 border-b border-line">
        <input
          aria-label="过滤或快速打开文件"
          className="w-full bg-white border border-line rounded-lg px-2.5 py-1.5 text-ink placeholder:text-faint focus:outline-none focus:border-moss"
          placeholder="过滤 / 快速打开文件…（回车打开首个）"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && matches[0]) {
              onSelectPath(matches[0].path, "file");
            }
            if (event.key === "Escape") setQuery("");
          }}
        />
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto px-1.5 py-1.5">
        {error ? <div className="px-2 py-1 text-signal">{error}</div> : null}
        {filtering ? (
          matches.length === 0 ? (
            <div className="px-2 py-3 text-muted">没有匹配「{query.trim()}」的文件。</div>
          ) : (
            <>
              <div className="px-2 py-1 font-mono text-[10px] text-faint">
                {matches.length} 个匹配
              </div>
              {matches.map((entry) => (
                <button
                  key={entry.path}
                  type="button"
                  className={`block w-full text-left truncate rounded-md px-2 py-1 font-mono text-[11px] ${
                    selectedPath === entry.path
                      ? "bg-white font-medium text-ink shadow-[inset_3px_0_0_#ca5b3f]"
                      : "text-muted hover:bg-white/70 hover:text-ink"
                  }`}
                  title={entry.path}
                  onClick={() => onSelectPath(entry.path, "file")}
                >
                  {entry.path}
                </button>
              ))}
            </>
          )
        ) : nodes.length === 0 && !error ? (
          <div className="px-2 py-3 text-muted">工作区还没有文件。</div>
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
    </div>
  );
};
