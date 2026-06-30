import React from "react";

interface MemoryHit {
  fragmentId: string;
  content: string;
  score?: number;
  memoryType: string;
  taskTags: string[];
}

/**
 * 记忆面板：直接复用 IPC `memoryQuery` 检索持久化记忆，
 * 让用户无需 DevTools 即可确认「会话能访问到哪些已落盘的记忆」。
 * - 有查询词 → 语义检索（带 score）
 * - 空查询 → 列出最近片段（仅元数据，content 可能为空）
 */
export const MemoryPanel: React.FC = () => {
  const [query, setQuery] = React.useState("");
  const [hits, setHits] = React.useState<MemoryHit[] | undefined>();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | undefined>();

  const search = React.useCallback(async (q: string): Promise<void> => {
    const api = window.openintj;
    if (!api?.memoryQuery) return;
    setLoading(true);
    setError(undefined);
    try {
      const res = await api.memoryQuery(q.trim() ? { query: q.trim(), topK: 10 } : { topK: 20 });
      setHits(res as MemoryHit[]);
    } catch (e) {
      setError((e as Error).message);
      setHits([]);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="flex flex-col h-full text-xs">
      <div className="p-2 border-b border-gray-800 flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void search(query);
          }}
          placeholder="检索记忆（留空=最近片段）"
          className="flex-1 bg-[#1e1e2e] border border-gray-700 rounded px-2 py-1 text-gray-200 focus:outline-none focus:border-purple-500"
        />
        <button
          type="button"
          onClick={() => void search(query)}
          disabled={loading}
          className="px-3 py-1 rounded bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white"
        >
          {loading ? "…" : "检索"}
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {error ? <div className="text-red-400">[错误] {error}</div> : null}
        {hits === undefined ? (
          <div className="text-gray-500">输入关键词检索，或留空点「检索」查看最近记忆。</div>
        ) : hits.length === 0 ? (
          <div className="text-gray-500">没有命中的记忆。</div>
        ) : (
          hits.map((h) => (
            <div key={h.fragmentId} className="border-l-2 border-gray-700 pl-2">
              <div className="flex items-center gap-2 text-[10px] text-gray-500">
                <span className="px-1 rounded bg-gray-800 text-gray-300">{h.memoryType}</span>
                {typeof h.score === "number" ? <span>score={h.score.toFixed(3)}</span> : null}
                {h.taskTags.length > 0 ? <span>tags: {h.taskTags.join(", ")}</span> : null}
              </div>
              <div className="text-gray-300 whitespace-pre-wrap break-words mt-0.5">
                {h.content?.trim() ? (
                  h.content
                ) : (
                  <span className="text-gray-600">（仅元数据，无正文）</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
