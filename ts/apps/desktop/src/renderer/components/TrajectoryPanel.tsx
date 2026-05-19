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

export const TrajectoryPanel: React.FC<{ entries: TrajectoryEntry[] }> = ({ entries }) => {
  const ref = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [entries.length]);

  return (
    <div className="flex flex-col h-full bg-[#11111b] border-l border-gray-800">
      <div className="px-3 py-2 text-xs font-semibold text-gray-300 border-b border-gray-800 uppercase tracking-wide">
        推理轨迹
      </div>
      <div ref={ref} className="flex-1 overflow-y-auto p-3 space-y-2 text-xs">
        {entries.length === 0 ? (
          <div className="text-gray-500">暂无轨迹</div>
        ) : (
          entries.map((e, i) => (
            <div key={i} className="border-l-2 border-gray-700 pl-2">
              <div className={`font-semibold ${KIND_COLORS[e.kind] ?? "text-gray-400"}`}>
                {KIND_LABELS[e.kind] ?? e.kind}
              </div>
              <pre className="text-gray-400 whitespace-pre-wrap break-all text-[11px]">
                {JSON.stringify(e.payload, null, 2)}
              </pre>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
