import type React from "react";
import type { StatusResponse } from "../../shared/ipc-protocol.js";

/** Re-export 给 App.tsx 与上游消费方使用。 */
export type StatusSnapshot = StatusResponse;

export const StatusBar: React.FC<{ status: StatusSnapshot | undefined }> = ({ status }) => {
  if (!status) {
    return (
      <div className="shrink-0 px-4 py-1 text-xs text-muted border-t border-line">
        加载状态中...
      </div>
    );
  }
  const runtimeLlm = status.modelRuntime.llm;
  const runtimeEmbed = status.modelRuntime.embed;
  const dotColor =
    runtimeLlm.status === "connected" && !runtimeLlm.fallbackFrom
      ? "bg-[#62a36d]"
      : status.llm.status === "degraded"
        ? "bg-amberdot"
        : "bg-signal";
  return (
    <div className="shrink-0 flex items-center gap-4 overflow-x-auto whitespace-nowrap border-t border-line bg-panel px-4 py-1 font-mono text-[11px] text-muted [&>span]:shrink-0">
      <span className="flex items-center gap-1">
        <span className={`inline-block w-2 h-2 rounded-full ${dotColor}`} />
        LLM: {runtimeLlm.provider} ({runtimeLlm.model}) · {runtimeLlm.mode}
        {runtimeLlm.fallbackFrom ? ` ← ${runtimeLlm.fallbackFrom}` : ""}
      </span>
      <span className={runtimeEmbed.fallbackFrom ? "text-ambertext" : "text-muted"}>
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
        <span className="text-muted">
          检索: <span className="text-muted">{status.retrievalMode}</span>
        </span>
      ) : null}
      {status.persistence ? (
        <span className="text-muted">
          盘: <span className="text-muted">{status.persistence.mode}</span>
        </span>
      ) : null}
      <span className="text-muted">
        分类器:{" "}
        <span className={status.classifier.enabled ? "text-muted" : "text-muted"}>
          {status.classifier.enabled
            ? status.classifier.impliedByTaskPool
              ? "on (TaskPool)"
              : "on"
            : "off"}
        </span>
      </span>
      {status.taskPool ? (
        <span className={status.taskPool.active ? "text-moss" : "text-muted"}>
          TaskPool: {status.taskPool.active ? "active" : status.taskPool.reason}
          {status.taskPool.persistence === "sqlite" ? ` · ${status.taskPool.recovery}` : ""}
        </span>
      ) : null}
      {status.dormant ? (
        <span className="text-muted">
          Dormant:{" "}
          <span
            className={status.dormant.pendingProposals > 0 ? "text-ambertext" : "text-muted"}
          >
            {status.dormant.passiveSize} ev / {status.dormant.pendingProposals} 待审
          </span>
        </span>
      ) : null}
      {status.skills ? (
        <span className={status.skills.enabled ? "text-muted" : "text-muted"}>
          技能:{" "}
          <span
            className={
              status.skills.pendingProposals > 0
                ? "text-ambertext"
                : status.skills.enabled
                  ? "text-muted"
                  : "text-muted"
            }
          >
            {status.skills.enabled
              ? `${status.skills.catalogSize} 包${status.skills.learning ? " · 自学习" : ""}`
              : "off"}
            {status.skills.pendingProposals > 0 ? ` / ${status.skills.pendingProposals} 待审` : ""}
          </span>
        </span>
      ) : null}
      <span className="ml-auto">工具: {status.tools.join(", ")}</span>
    </div>
  );
};
