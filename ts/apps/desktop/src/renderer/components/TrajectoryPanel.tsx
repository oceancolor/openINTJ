import React from "react";

export type TrajectoryEntryKind =
  | "tao.beforeThink"
  | "tao.afterAct"
  | "react.thought"
  | "react.action"
  | "react.observation"
  | "audit";

export interface TrajectoryEntry {
  kind: TrajectoryEntryKind;
  payload: Record<string, unknown>;
  ts: number;
}

const KIND_LABELS: Record<TrajectoryEntryKind, string> = {
  "tao.beforeThink": "Think",
  "tao.afterAct": "After Act",
  "react.thought": "Thought",
  "react.action": "Action",
  "react.observation": "Observation",
  audit: "Audit",
};

const KIND_COLORS: Record<TrajectoryEntryKind, string> = {
  "tao.beforeThink": "text-cyan-400",
  "tao.afterAct": "text-blue-400",
  "react.thought": "text-purple-400",
  "react.action": "text-orange-400",
  "react.observation": "text-green-400",
  audit: "text-yellow-400",
};

interface SearchSource {
  title?: string;
  url?: string;
}

/** 若 observation 来自 search 工具，抽取 { answer, sources }；否则返回 null。 */
const extractSearchHit = (
  payload: Record<string, unknown>,
): { answer?: string; sources: SearchSource[] } | null => {
  const tr = payload["toolResult"] as
    | { toolName?: string; success?: boolean; output?: unknown }
    | undefined;
  if (!tr || tr.toolName !== "search") return null;
  const output = tr.output as { answer?: unknown; sources?: unknown } | undefined;
  const rawSources = Array.isArray(output?.sources) ? (output.sources as SearchSource[]) : [];
  const sources = rawSources.filter((s) => typeof s?.url === "string" || typeof s?.title === "string");
  const answer = typeof output?.answer === "string" ? output.answer : undefined;
  return { ...(answer ? { answer } : {}), sources };
};

const SearchObservation: React.FC<{ hit: { answer?: string; sources: SearchSource[] } }> = ({
  hit,
}) => (
  <div className="space-y-1.5">
    {hit.answer ? (
      <div className="text-gray-300 whitespace-pre-wrap text-[11px] leading-relaxed">
        {hit.answer.length > 400 ? `${hit.answer.slice(0, 400)}…` : hit.answer}
      </div>
    ) : null}
    {hit.sources.length > 0 ? (
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wide text-green-500">
          联网来源 · {hit.sources.length}
        </div>
        <ol className="space-y-0.5 list-decimal list-inside">
          {hit.sources.map((s, i) => (
            <li key={i} className="text-gray-400 break-all text-[11px]">
              <span className="text-gray-300">{s.title?.trim() || s.url}</span>
              {s.url ? <span className="block text-cyan-500/80 break-all">{s.url}</span> : null}
            </li>
          ))}
        </ol>
      </div>
    ) : (
      <div className="text-gray-500 text-[11px]">（本次未命中联网搜索结果）</div>
    )}
  </div>
);

export const TrajectoryPanel: React.FC<{ entries: TrajectoryEntry[] }> = ({ entries }) => {
  const ref = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [entries.length]);

  return (
    <div ref={ref} className="h-full overflow-y-auto p-3 space-y-2 text-xs">
      {entries.length === 0 ? (
        <div className="text-gray-500">暂无轨迹</div>
      ) : (
        entries.map((e, i) => {
          const searchHit =
            e.kind === "react.observation" ? extractSearchHit(e.payload) : null;
          return (
            <div key={i} className="border-l-2 border-gray-700 pl-2">
              <div className={`font-semibold ${KIND_COLORS[e.kind] ?? "text-gray-400"}`}>
                {KIND_LABELS[e.kind] ?? e.kind}
                {searchHit ? <span className="ml-1.5 text-[10px] text-green-500">search</span> : null}
              </div>
              {searchHit ? (
                <SearchObservation hit={searchHit} />
              ) : (
                <pre className="text-gray-400 whitespace-pre-wrap break-all text-[11px]">
                  {JSON.stringify(e.payload, null, 2)}
                </pre>
              )}
            </div>
          );
        })
      )}
    </div>
  );
};
