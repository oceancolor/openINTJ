/**
 * 技能审批面板（技能系统 Phase 2）。
 *
 * 工作流程：
 *  1. 用户点击 [蒸馏] 触发一次轨迹蒸馏 → 成功轨迹聚成候选技能 pending 提案
 *  2. 列表里每条 proposal 显示技能名 / 描述 / 证据（命中 query 数 / taskType）
 *  3. pending → 用户点 ✓ 批准（写生效技能并热重载注册表）/ ✗ 拒绝
 *     approved → 用户点「撤销」把它从生效集移除
 *  4. 顶部 status filter 切换 pending / approved / rejected / revoked / all
 *  5. 底部「生效技能」折叠区显示当前学习技能 + 权重
 *
 * 技能自学习未启用时（status.skills === undefined）：
 *  - 显示「未启用」提示 + 启用方法
 */
import React from "react";
import type {
  SkillActiveDto,
  SkillLearningError,
  SkillProposalDto,
} from "../../shared/ipc-protocol.js";

type StatusFilter = "pending" | "approved" | "rejected" | "revoked" | "all";

const STATUS_TABS: Array<{ key: StatusFilter; label: string }> = [
  { key: "pending", label: "待审批" },
  { key: "approved", label: "已批准" },
  { key: "rejected", label: "已拒绝" },
  { key: "revoked", label: "已撤销" },
  { key: "all", label: "全部" },
];

const STATUS_BADGE: Record<SkillProposalDto["status"], string> = {
  pending: "bg-yellow-900 text-yellow-200",
  approved: "bg-green-900 text-green-200",
  rejected: "bg-red-900 text-red-200",
  revoked: "bg-zinc-700 text-zinc-300",
};

const formatTime = (ts: number): string => {
  const d = new Date(ts);
  const pad = (n: number): string => n.toString().padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

const isError = (r: unknown): r is SkillLearningError => {
  return typeof r === "object" && r !== null && "error" in r;
};

export interface SkillPanelProps {
  /** 来自 status.skills；false 表示子系统未启用。 */
  enabled: boolean;
}

export const SkillPanel: React.FC<SkillPanelProps> = ({ enabled }) => {
  const [filter, setFilter] = React.useState<StatusFilter>("pending");
  const [proposals, setProposals] = React.useState<SkillProposalDto[]>([]);
  const [active, setActive] = React.useState<SkillActiveDto[]>([]);
  const [showActive, setShowActive] = React.useState(false);
  const [busy, setBusy] = React.useState<string | undefined>();
  const [error, setError] = React.useState<string | undefined>();
  const [lastDistillSummary, setLastDistillSummary] = React.useState<string | undefined>();

  const refreshList = React.useCallback(
    async (nextFilter: StatusFilter = filter): Promise<void> => {
      const api = window.openintj;
      if (!api || !enabled) return;
      try {
        const req = nextFilter === "all" ? {} : { status: nextFilter };
        const r = await api.skillsList(req);
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

  const refreshActive = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api || !enabled) return;
    try {
      const r = await api.skillsActive();
      if (isError(r)) return;
      setActive(r.skills);
    } catch {
      // ignore
    }
  }, [enabled]);

  React.useEffect(() => {
    if (!enabled) return;
    void refreshList(filter);
  }, [enabled, filter, refreshList]);

  React.useEffect(() => {
    if (!enabled || !showActive) return;
    void refreshActive();
  }, [enabled, showActive, refreshActive]);

  const handleDistill = async (): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    setBusy("distill");
    setError(undefined);
    try {
      const r = await api.skillsDistill();
      if (isError(r)) {
        setError(r.error);
        return;
      }
      setLastDistillSummary(`蒸馏产出 ${r.produced} 个新候选技能提案`);
      await refreshList("pending");
      setFilter("pending");
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
          ? await api.skillsApprove({ proposalId })
          : decision === "reject"
            ? await api.skillsReject({ proposalId })
            : await api.skillsRevoke({ proposalId });
      if (isError(r)) {
        setError(r.error);
        return;
      }
      await refreshList(filter);
      if ((decision === "approve" || decision === "revoke") && showActive) {
        await refreshActive();
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
          <div className="text-base text-gray-300">技能自学习未启用</div>
          <div>
            启动桌面端时设置环境变量
            <code className="mx-1 px-1.5 py-0.5 bg-gray-800 rounded text-xs">
              OPENINTJ_SKILLS_LEARN=1
            </code>
            或在装配时传入
            <code className="mx-1 px-1.5 py-0.5 bg-gray-800 rounded text-xs">
              enableSkillLearning: true
            </code>
            。
          </div>
          <div className="text-xs text-gray-500">详见技能系统 Phase 2 文档。</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="shrink-0 flex items-center justify-end px-3 py-2 border-b border-gray-800">
        <button
          type="button"
          onClick={() => void handleDistill()}
          disabled={busy === "distill"}
          className="px-2 py-1 text-xs rounded bg-purple-700 hover:bg-purple-600 disabled:bg-gray-700 text-white"
        >
          {busy === "distill" ? "蒸馏中…" : "蒸馏"}
        </button>
      </div>

      <div className="shrink-0 flex items-center gap-1 px-3 py-1 border-b border-gray-800 text-xs">
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
      {lastDistillSummary ? (
        <div className="mx-3 mt-2 px-2 py-1 text-xs text-gray-400 bg-gray-800/40 rounded">
          {lastDistillSummary}
        </div>
      ) : null}

      <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2 text-xs">
        {proposals.length === 0 ? (
          <div className="text-gray-500">
            {filter === "pending" ? "暂无待审批提案 — 点击蒸馏从成功轨迹提炼候选技能" : "暂无数据"}
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
                <span className="text-gray-200 font-medium">{p.name}</span>
                <span className="ml-auto text-gray-500 text-[11px]">{formatTime(p.ts)}</span>
              </div>
              <div className="text-gray-300">{p.description}</div>
              <div className="text-gray-400 text-[11px]">
                <code className="text-cyan-400">{p.skillId}</code>
                <span className="mx-1 text-gray-600">·</span>
                <span>命中 {p.evidence.count} 次</span>
                {p.evidence.taskType ? (
                  <>
                    <span className="mx-1 text-gray-600">·</span>
                    <span>{p.evidence.taskType}</span>
                  </>
                ) : null}
              </div>
              {p.evidence.queries.length > 0 ? (
                <div className="text-gray-500 text-[10px] truncate">
                  例：{p.evidence.queries.slice(0, 3).join(" / ")}
                </div>
              ) : null}
              {p.status === "pending" ? (
                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => void handleDecision(p.proposalId, "approve")}
                    disabled={busy === `approve:${p.proposalId}`}
                    className="px-2 py-0.5 text-xs rounded bg-green-700 hover:bg-green-600 disabled:bg-gray-700 text-white"
                  >
                    ✓ 批准
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
              ) : p.status === "approved" ? (
                <div className="flex items-center gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => void handleDecision(p.proposalId, "revoke")}
                    disabled={busy === `revoke:${p.proposalId}`}
                    className="px-2 py-0.5 text-xs rounded bg-orange-800 hover:bg-orange-700 disabled:bg-gray-700 text-white"
                  >
                    撤销
                  </button>
                  {p.decidedAt ? (
                    <span className="text-gray-500 text-[10px]">
                      批准于 {formatTime(p.decidedAt)}
                    </span>
                  ) : null}
                </div>
              ) : p.decidedAt ? (
                <div className="text-gray-500 text-[10px]">决策于 {formatTime(p.decidedAt)}</div>
              ) : null}
            </div>
          ))
        )}
      </div>

      <div className="shrink-0 border-t border-gray-800">
        <button
          type="button"
          onClick={() => setShowActive((v) => !v)}
          className="w-full px-3 py-1.5 text-xs text-left text-gray-300 hover:bg-gray-800"
        >
          {showActive ? "▼" : "▶"} 生效技能
          {active.length > 0 ? (
            <span className="ml-2 text-gray-500">{active.length} 个</span>
          ) : null}
        </button>
        {showActive ? (
          <div className="max-h-40 overflow-y-auto px-3 pb-2 space-y-1 text-[11px]">
            {active.length === 0 ? (
              <div className="text-gray-500">暂无生效的学习技能</div>
            ) : (
              active.map((s) => (
                <div key={s.id} className="flex items-center gap-2 text-gray-400">
                  <span className="text-gray-200">{s.name}</span>
                  {s.source ? <span className="text-gray-600">[{s.source}]</span> : null}
                  <span className="ml-auto text-cyan-400">w={s.weight.toFixed(2)}</span>
                </div>
              ))
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
};
