import type React from "react";
import type { StatusResponse } from "../../shared/ipc-protocol.js";

/** Re-export 给 App.tsx 与上游消费方使用。 */
export type StatusSnapshot = StatusResponse;

export const StatusBar: React.FC<{ status: StatusSnapshot | undefined }> = ({ status }) => {
  if (!status) {
    return (
      <div className="px-4 py-1 text-xs text-gray-500 border-t border-gray-800">加载状态中...</div>
    );
  }
  const runtimeLlm = status.modelRuntime.llm;
  const runtimeEmbed = status.modelRuntime.embed;
  const dotColor =
    runtimeLlm.status === "connected" && !runtimeLlm.fallbackFrom
      ? "bg-green-400"
      : status.llm.status === "degraded"
        ? "bg-yellow-400"
        : "bg-red-400";
  return (
    <div className="flex items-center gap-4 px-4 py-1 text-xs text-gray-400 border-t border-gray-800 bg-[#181825]">
      <span className="flex items-center gap-1">
        <span className={`inline-block w-2 h-2 rounded-full ${dotColor}`} />
        LLM: {runtimeLlm.provider} ({runtimeLlm.model}) · {runtimeLlm.mode}
        {runtimeLlm.fallbackFrom ? ` ← ${runtimeLlm.fallbackFrom}` : ""}
      </span>
      <span className={runtimeEmbed.fallbackFrom ? "text-yellow-300" : "text-gray-400"}>
        Embed: {runtimeEmbed.provider} ({runtimeEmbed.model}, {runtimeEmbed.dimension}d) ·{" "}
        {runtimeEmbed.mode}
        {runtimeEmbed.fallbackFrom ? ` ← ${runtimeEmbed.fallbackFrom}` : ""}
      </span>
      <span>记忆: {status.memory.total}</span>
      <span>
        审计: {status.governance.audit.totalEvents} 条
        {status.governance.audit.blockedCount > 0
          ? ` (拦截 ${status.governance.audit.blockedCount})`
          : ""}
      </span>
      {status.retrievalMode ? (
        <span className="text-gray-500">
          检索: <span className="text-gray-300">{status.retrievalMode}</span>
        </span>
      ) : null}
      {status.persistence ? (
        <span className="text-gray-500">
          盘: <span className="text-gray-300">{status.persistence.mode}</span>
        </span>
      ) : null}
      <span className="text-gray-500">
        分类器:{" "}
        <span className={status.classifier.enabled ? "text-gray-300" : "text-gray-600"}>
          {status.classifier.enabled
            ? status.classifier.impliedByTaskPool
              ? "on (TaskPool)"
              : "on"
            : "off"}
        </span>
      </span>
      {status.taskPool ? (
        <span className={status.taskPool.active ? "text-green-300" : "text-gray-500"}>
          TaskPool: {status.taskPool.active ? "active" : status.taskPool.reason}
          {status.taskPool.persistence === "sqlite" ? ` · ${status.taskPool.recovery}` : ""}
        </span>
      ) : null}
      {status.dormant ? (
        <span className="text-gray-500">
          Dormant:{" "}
          <span
            className={status.dormant.pendingProposals > 0 ? "text-yellow-300" : "text-gray-300"}
          >
            {status.dormant.passiveSize} ev / {status.dormant.pendingProposals} 待审
          </span>
        </span>
      ) : null}
      <span className="ml-auto">工具: {status.tools.join(", ")}</span>
    </div>
  );
};
