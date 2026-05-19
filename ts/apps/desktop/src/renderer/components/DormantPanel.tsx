/**
 * Dormant 审批面板（Phase 3.5 #9.B）。
 *
 * 工作流程：
 *  1. 用户点击 [mine] 触发分析 → 产出待审批 proposals
 *  2. 列表里每条 proposal 显示 pattern 描述 / 目标字段 / 置信度 / 频次
 *  3. 用户点击 ✓ / ✗ → IPC 调 approve / reject → 列表自动刷新
 *  4. 顶部 status filter 切换 pending / applied / rejected / all
 *  5. 底部"当前 Persona"折叠区显示 snapshot
 *
 * Dormant 未启用时（agent.dormant === undefined）：
 *  - status.dormant 是 undefined → 显示"未启用"提示 + 启用方法
 */
import React from "react";
import type { DormantPersonaResponse, DormantProposalDto } from "../../shared/ipc-protocol.js";

type StatusFilter = "pending" | "applied" | "rejected" | "all";

const STATUS_TABS: Array<{ key: StatusFilter; label: string }> = [
  { key: "pending", label: "待审批" },
  { key: "applied", label: "已应用" },
  { key: "rejected", label: "已拒绝" },
  { key: "all", label: "全部" },
];

const STATUS_BADGE: Record<DormantProposalDto["status"], string> = {
  pending: "bg-yellow-900 text-yellow-200",
  applied: "bg-green-900 text-green-200",
  rejected: "bg-red-900 text-red-200",
  approved: "bg-blue-900 text-blue-200",
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
    decision: "approve" | "reject",
  ): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    setBusy(`${decision}:${proposalId}`);
    setError(undefined);
    try {
      const r =
        decision === "approve"
          ? await api.dormantApprove({ proposalId })
          : await api.dormantReject({ proposalId });
      if (isError(r)) {
        setError(r.error);
        return;
      }
      await refreshList(filter);
      if (decision === "approve" && showPersona) {
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
        <div className="text-sm text-gray-400 space-y-3">
          <div className="text-base text-gray-300">Dormant 子系统未启用</div>
          <div>
            启动桌面端时设置环境变量
            <code className="mx-1 px-1.5 py-0.5 bg-gray-800 rounded text-xs">
              OPENINTJ_DORMANT=1
            </code>
            或在装配时传入
            <code className="mx-1 px-1.5 py-0.5 bg-gray-800 rounded text-xs">
              enableDormant: true
            </code>
            。
          </div>
          <div className="text-xs text-gray-500">详见 RFC-003 方向 3 / Phase 3.4 文档。</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-end px-3 py-2 border-b border-gray-800">
        <button
          type="button"
          onClick={() => void handleMine()}
          disabled={busy === "mine"}
          className="px-2 py-1 text-xs rounded bg-purple-700 hover:bg-purple-600 disabled:bg-gray-700 text-white"
        >
          {busy === "mine" ? "分析中…" : "Mine"}
        </button>
      </div>

      <div className="flex items-center gap-1 px-3 py-1 border-b border-gray-800 text-xs">
        {STATUS_TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setFilter(t.key)}
            className={
              filter === t.key
                ? "px-2 py-0.5 rounded bg-gray-700 text-gray-100"
                : "px-2 py-0.5 rounded text-gray-400 hover:text-gray-200"
            }
          >
            {t.label}
          </button>
        ))}
        <span className="ml-auto text-gray-500">{proposals.length} 条</span>
      </div>

      {error ? (
        <div className="mx-3 mt-2 px-2 py-1 text-xs bg-red-900/50 text-red-200 rounded">
          {error}
        </div>
      ) : null}
      {lastMineSummary ? (
        <div className="mx-3 mt-2 px-2 py-1 text-xs text-gray-400 bg-gray-800/40 rounded">
          {lastMineSummary}
        </div>
      ) : null}

      <div className="flex-1 overflow-y-auto p-3 space-y-2 text-xs">
        {proposals.length === 0 ? (
          <div className="text-gray-500">
            {filter === "pending" ? "暂无待审批 proposal — 点击 Mine 触发分析" : "暂无数据"}
          </div>
        ) : (
          proposals.map((p) => (
            <div
              key={p.proposalId}
              className="border border-gray-800 rounded p-2 space-y-1 bg-[#181825]"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] uppercase ${STATUS_BADGE[p.status]}`}
                >
                  {p.status}
                </span>
                <span className="text-gray-400 text-[11px]">
                  freq {p.frequency} · conf {(p.confidence * 100).toFixed(0)}%
                </span>
                <span className="ml-auto text-gray-500 text-[11px]">{formatTime(p.ts)}</span>
              </div>
              <div className="text-gray-200">{p.patternDescription}</div>
              <div className="text-gray-400 text-[11px]">
                <code className="text-cyan-400">{p.targetField}</code>
                <span className="mx-1 text-gray-600">←</span>
                <span className="text-gray-300">{formatValue(p.value)}</span>
              </div>
              {p.status === "pending" ? (
                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => void handleDecision(p.proposalId, "approve")}
                    disabled={busy === `approve:${p.proposalId}`}
                    className="px-2 py-0.5 text-xs rounded bg-green-700 hover:bg-green-600 disabled:bg-gray-700 text-white"
                  >
                    ✓ 应用
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleDecision(p.proposalId, "reject")}
                    disabled={busy === `reject:${p.proposalId}`}
                    className="px-2 py-0.5 text-xs rounded bg-red-700 hover:bg-red-600 disabled:bg-gray-700 text-white"
                  >
                    ✗ 拒绝
                  </button>
                </div>
              ) : p.decidedAt ? (
                <div className="text-gray-500 text-[10px]">决策于 {formatTime(p.decidedAt)}</div>
              ) : null}
            </div>
          ))
        )}
      </div>

      <div className="border-t border-gray-800">
        <button
          type="button"
          onClick={() => setShowPersona((v) => !v)}
          className="w-full px-3 py-1.5 text-xs text-left text-gray-300 hover:bg-gray-800"
        >
          {showPersona ? "▼" : "▶"} 当前 Persona
          {persona ? (
            <span className="ml-2 text-gray-500">
              v{persona.meta.version} · {Object.keys(persona.preferences).length} 偏好
            </span>
          ) : null}
        </button>
        {showPersona && persona ? (
          <pre className="max-h-40 overflow-y-auto px-3 pb-2 text-[11px] text-gray-400 whitespace-pre-wrap break-all">
            {JSON.stringify(persona, null, 2)}
          </pre>
        ) : null}
      </div>
    </div>
  );
};
