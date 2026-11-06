import type React from "react";
import type { ApprovalRequestView } from "../../shared/ipc-protocol.js";

/**
 * RFC-013: pending tool approvals. The turn is parked in the governance gate
 * until the user decides; "批准并授权" also mints a 1h scoped grant so the
 * same path/tool does not prompt again.
 */
export const ApprovalBanner: React.FC<{
  pending: ApprovalRequestView[];
  onDecide: (req: ApprovalRequestView, decision: "approved" | "denied", withGrant: boolean) => void;
}> = ({ pending, onDecide }) => {
  if (pending.length === 0) return null;
  const req = pending[0]!;
  const path = typeof req.params["path"] === "string" ? (req.params["path"] as string) : undefined;
  const command =
    typeof req.params["command"] === "string" ? (req.params["command"] as string) : undefined;
  return (
    <div className="shrink-0 border-b border-line bg-panel px-3 py-3" data-testid="approval-banner">
      <div className="rounded-xl border border-signalborder bg-signalwash p-3">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-signaltint text-xs font-bold text-signaldeep">
            !
          </span>
          <span className="min-w-0 flex-1 text-xs font-semibold text-ink">
            需要审批：{req.tool}
          </span>
          <span className="shrink-0 rounded-full bg-signaltint px-2 py-0.5 text-[10px] font-semibold text-signaldeep">
            {req.risk}
          </span>
          {pending.length > 1 ? (
            <span className="shrink-0 font-mono text-[10px] text-muted">
              还有 {pending.length - 1} 个待审批
            </span>
          ) : null}
        </div>
        <div className="mt-2 ml-8 break-all font-mono text-[10px] leading-4 text-muted">
          {path ?? command ?? JSON.stringify(req.params).slice(0, 200)}
        </div>
        <div className="mt-2.5 ml-8 flex items-center gap-2">
          <button
            type="button"
            className="rounded-lg bg-ink px-2.5 py-1.5 text-xs font-medium text-white hover:bg-inksoft"
            onClick={() => onDecide(req, "approved", false)}
          >
            批准一次
          </button>
          <button
            type="button"
            className="rounded-lg border border-line bg-white px-2.5 py-1.5 text-xs font-medium text-ink hover:border-linedark"
            onClick={() => onDecide(req, "approved", true)}
          >
            批准并授权（1 小时）
          </button>
          <button
            type="button"
            className="ml-auto rounded-lg px-2.5 py-1.5 text-xs font-medium text-signaldeep hover:bg-signaltint"
            onClick={() => onDecide(req, "denied", false)}
          >
            拒绝
          </button>
        </div>
      </div>
    </div>
  );
};
