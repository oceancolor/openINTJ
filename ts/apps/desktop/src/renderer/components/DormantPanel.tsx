/**
 * Dormant 审批面板（Phase 3.5 #9.B）。
 *
 * 工作流程：
 *  1. 用户点击 [mine] 触发分析 → 产出待审批 proposals
 *  2. 列表里每条 proposal 显示 pattern 描述 / 目标字段 / 置信度 / 频次
 *  3. pending → 用户点 ✓ 应用（写入 PersonaConfig）/ ✗ 拒绝
 *     applied → 用户点「撤销」把该字段从 PersonaConfig 删除（可回退，RFC-003 §3.6 #4）
 *  4. 顶部 status filter 切换 pending / applied / rejected / revoked / all
 *  5. 底部"当前 Persona"折叠区显示 snapshot
 *
 * Dormant 未启用时（agent.dormant === undefined）：
 *  - status.dormant 是 undefined → 显示"未启用"提示 + 启用方法
 */
import React from "react";
import type { DormantPersonaResponse, DormantProposalDto } from "../../shared/ipc-protocol.js";

type StatusFilter = "pending" | "applied" | "rejected" | "revoked" | "all";

const STATUS_TABS: Array<{ key: StatusFilter; label: string }> = [
  { key: "pending", label: "待审批" },
  { key: "applied", label: "已应用" },
  { key: "rejected", label: "已拒绝" },
  { key: "revoked", label: "已撤销" },
  { key: "all", label: "全部" },
];

const STATUS_BADGE: Record<DormantProposalDto["status"], string> = {
  pending: "bg-ambertint text-amberdeep",
  applied: "bg-mossmist text-moss",
  rejected: "bg-[#f8e9e4] text-signal",
  approved: "bg-mosstint text-mossdeep",
  revoked: "bg-zinc-700 text-zinc-300",
};

const formatTime = (ts: number): string => {
  const d = new Date(ts);
  const pad = (n: number): string => n.toString().padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

const formatValue = (v: unknown): string => {
  if (typeof v === "string") return v;
  if (v === null || v === undefined) return "(空)";
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
};

const isError = (r: unknown): r is { error: string } => {
  return typeof r === "object" && r !== null && "error" in r;
};

export interface DormantPanelProps {
  /** 来自 status.dormant；undefined 表示子系统未启用。 */
  enabled: boolean;
}

export const DormantPanel: React.FC<DormantPanelProps> = ({ enabled }) => {
  const [filter, setFilter] = React.useState<StatusFilter>("pending");
  const [proposals, setProposals] = React.useState<DormantProposalDto[]>([]);
  const [persona, setPersona] = React.useState<DormantPersonaResponse | undefined>();
  const [showPersona, setShowPersona] = React.useState(false);
  const [busy, setBusy] = React.useState<string | undefined>();
  const [error, setError] = React.useState<string | undefined>();
  const [lastMineSummary, setLastMineSummary] = React.useState<string | undefined>();

  const refreshList = React.useCallback(
    async (nextFilter: StatusFilter = filter): Promise<void> => {
      const api = window.openintj;
      if (!api || !enabled) return;
      try {
        const req = nextFilter === "all" ? {} : { status: nextFilter };
        const r = await api.dormantList(req);
        if (isError(r)) {
          setError(r.error);
          setProposals([]);
          return;
        }
        setProposals(r.proposals);
        setError(undefined);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [enabled, filter],
  );

  const refreshPersona = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api || !enabled) return;
    try {
      const r = await api.dormantPersona();
      if (isError(r)) return;
      setPersona(r);
    } catch {
      // ignore
    }
  }, [enabled]);

  React.useEffect(() => {
    if (!enabled) return;
    void refreshList(filter);
  }, [enabled, filter, refreshList]);

  React.useEffect(() => {
    if (!enabled || !showPersona) return;
    void refreshPersona();
  }, [enabled, showPersona, refreshPersona]);

  const handleMine = async (): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    setBusy("mine");
    setError(undefined);
    try {
      const r = await api.dormantMine();
      if (isError(r)) {
        setError(r.error);
        return;
      }
      setLastMineSummary(
        `扫描 ${r.scannedEvents} 条事件 · 产出 ${r.patterns.length} 个 pattern · ${r.proposals.length} 个新 proposal`,
      );
      await refreshList(filter);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(undefined);
    }
  };

  const handleDecision = async (
    proposalId: string,
    decision: "approve" | "reject" | "revoke",
  ): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    setBusy(`${decision}:${proposalId}`);
    setError(undefined);
    try {
      const r =
        decision === "approve"
          ? await api.dormantApprove({ proposalId })
          : decision === "reject"
            ? await api.dormantReject({ proposalId })
            : await api.dormantRevoke({ proposalId });
      if (isError(r)) {
        setError(r.error);
        return;
      }
      await refreshList(filter);
      if ((decision === "approve" || decision === "revoke") && showPersona) {
        await refreshPersona();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(undefined);
    }
  };

  if (!enabled) {
    return (
      <div className="h-full flex items-center justify-center p-6 text-center">
        <div className="text-sm text-muted space-y-3">
          <div className="text-base text-muted">Dormant 子系统未启用</div>
          <div>
            启动桌面端时设置环境变量
            <code className="mx-1 px-1.5 py-0.5 bg-inset rounded text-xs">
              OPENINTJ_DORMANT=1
            </code>
            或在装配时传入
            <code className="mx-1 px-1.5 py-0.5 bg-inset rounded text-xs">
              enableDormant: true
            </code>
            。
          </div>
          <div className="text-xs text-muted">详见 RFC-003 方向 3 / Phase 3.4 文档。</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 flex items-center justify-end px-3 py-2 border-b border-line">
        <button
          type="button"
          onClick={() => void handleMine()}
          disabled={busy === "mine"}
          className="px-2 py-1 text-xs rounded bg-ink hover:bg-inksoft disabled:bg-rail text-white"
        >
          {busy === "mine" ? "分析中…" : "Mine"}
        </button>
      </div>

      <div className="shrink-0 flex items-center gap-1 px-3 py-1 border-b border-line text-xs">
        {STATUS_TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setFilter(t.key)}
            className={
              filter === t.key
                ? "px-2 py-0.5 rounded bg-rail text-ink"
                : "px-2 py-0.5 rounded text-muted hover:text-ink"
            }
          >
            {t.label}
          </button>
        ))}
        <span className="ml-auto text-muted">{proposals.length} 条</span>
      </div>

      {error ? (
        <div className="mx-3 mt-2 px-2 py-1 text-xs bg-signaltint text-signaldeep rounded">
          {error}
        </div>
      ) : null}
      {lastMineSummary ? (
        <div className="mx-3 mt-2 px-2 py-1 text-xs text-muted bg-inset/40 rounded">
          {lastMineSummary}
        </div>
      ) : null}

      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2 text-xs">
        {proposals.length === 0 ? (
          <div className="text-muted">
            {filter === "pending" ? "暂无待审批 proposal — 点击 Mine 触发分析" : "暂无数据"}
          </div>
        ) : (
          proposals.map((p) => (
            <div
              key={p.proposalId}
              className="border border-line rounded p-2 space-y-1 bg-white"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] uppercase ${STATUS_BADGE[p.status]}`}
                >
                  {p.status}
                </span>
                <span className="text-muted text-[11px]">
                  freq {p.frequency} · conf {(p.confidence * 100).toFixed(0)}%
                </span>
                <span className="ml-auto text-muted text-[11px]">{formatTime(p.ts)}</span>
              </div>
              <div className="text-ink">{p.patternDescription}</div>
              <div className="text-muted text-[11px]">
                <code className="text-moss">{p.targetField}</code>
                <span className="mx-1 text-muted">←</span>
                <span className="text-muted">{formatValue(p.value)}</span>
              </div>
              {p.status === "pending" ? (
                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => void handleDecision(p.proposalId, "approve")}
                    disabled={busy === `approve:${p.proposalId}`}
                    className="px-2 py-0.5 text-xs rounded bg-moss hover:bg-mossdeep disabled:bg-rail text-white"
                  >
                    ✓ 应用
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleDecision(p.proposalId, "reject")}
                    disabled={busy === `reject:${p.proposalId}`}
                    className="px-2 py-0.5 text-xs rounded bg-signal hover:bg-signaldeep disabled:bg-rail text-white"
                  >
                    ✗ 拒绝
                  </button>
                </div>
              ) : p.status === "applied" ? (
                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => void handleDecision(p.proposalId, "revoke")}
                    disabled={busy === `revoke:${p.proposalId}`}
                    className="rounded-lg border border-signalborder bg-white px-2 py-1 text-xs font-medium text-signaldeep hover:bg-signalwash disabled:opacity-50"
                  >
                    撤销
                  </button>
                  {p.decidedAt ? (
                    <span className="text-muted text-[10px]">
                      应用于 {formatTime(p.decidedAt)}
                    </span>
                  ) : null}
                </div>
              ) : p.decidedAt ? (
                <div className="text-muted text-[10px]">决策于 {formatTime(p.decidedAt)}</div>
              ) : null}
            </div>
          ))
        )}
      </div>

      <div className="shrink-0 border-t border-line">
        <button
          type="button"
          onClick={() => setShowPersona((v) => !v)}
          className="w-full px-3 py-1.5 text-xs text-left text-muted hover:bg-rail"
        >
          {showPersona ? "▼" : "▶"} 当前 Persona
          {persona ? (
            <span className="ml-2 text-muted">
              v{persona.meta.version} · {Object.keys(persona.preferences).length} 偏好
            </span>
          ) : null}
        </button>
        {showPersona && persona ? (
          <pre className="max-h-40 overflow-y-auto px-3 pb-2 text-[11px] text-muted whitespace-pre-wrap break-all">
            {JSON.stringify(persona, null, 2)}
          </pre>
        ) : null}
      </div>
    </div>
  );
};
