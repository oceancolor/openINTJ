import type React from "react";

export interface StatusSnapshot {
  llm: { provider: string; status: string; model?: string };
  memory: { counts: Record<string, number>; total: number };
  governance: {
    audit: { totalEvents: number; blockedCount: number };
  };
  tools: string[];
}

export const StatusBar: React.FC<{ status: StatusSnapshot | undefined }> = ({ status }) => {
  if (!status) {
    return (
      <div className="px-4 py-1 text-xs text-gray-500 border-t border-gray-800">加载状态中...</div>
    );
  }
  const dotColor =
    status.llm.status === "connected"
      ? "bg-green-400"
      : status.llm.status === "degraded"
        ? "bg-yellow-400"
        : "bg-red-400";
  return (
    <div className="flex items-center gap-4 px-4 py-1 text-xs text-gray-400 border-t border-gray-800 bg-[#181825]">
      <span className="flex items-center gap-1">
        <span className={`inline-block w-2 h-2 rounded-full ${dotColor}`} />
        LLM: {status.llm.provider}
        {status.llm.model ? ` (${status.llm.model})` : ""} · {status.llm.status}
      </span>
      <span>记忆: {status.memory.total}</span>
      <span>
        审计: {status.governance.audit.totalEvents} 条
        {status.governance.audit.blockedCount > 0
          ? ` (拦截 ${status.governance.audit.blockedCount})`
          : ""}
      </span>
      <span>工具: {status.tools.join(", ")}</span>
    </div>
  );
};
