import type React from "react";
import { type TrajectoryEntry, TrajectoryPanel } from "./TrajectoryPanel.js";

/**
 * 时间线（右栏）：把 TAO/ReAct/审计轨迹按时间合并成一条流，顶部
 * 给并行与协作状态筹码（任务池、待审批、轨迹量）。本轮为会话内
 * 视图（轨迹事件由宿主实时推送）；跨会话落盘是后续深度项。
 */
export const TimelinePanel: React.FC<{
  entries: TrajectoryEntry[];
  taskPoolActive: boolean;
  pendingApprovals: number;
  running: boolean;
}> = ({ entries, taskPoolActive, pendingApprovals, running }) => (
  <div className="flex h-full min-h-0 flex-col">
    <div className="shrink-0 flex items-center gap-1.5 border-b border-line px-3 py-2 text-[11px]">
      <span className="rounded-full bg-rail px-2 py-0.5 text-muted">轨迹 {entries.length}</span>
      <span className={`rounded-full px-2 py-0.5 ${taskPoolActive ? "bg-mosstint text-mossdeep" : "bg-rail text-muted"}`}>
        任务池 {taskPoolActive ? "开" : "关"}
      </span>
      {running ? <span className="rounded-full bg-ambertint px-2 py-0.5 text-amberdeep">运行中…</span> : null}
      {pendingApprovals > 0 ? (
        <span className="rounded-full bg-ambertint px-2 py-0.5 text-amberdeep">待审批 {pendingApprovals}</span>
      ) : null}
    </div>
    <div className="flex-1 min-h-0">
      <TrajectoryPanel entries={entries} />
    </div>
  </div>
);
