/**
 * 技能面板：已装能力包目录 + 技能自学习审批。
 *
 *  1. 顶部开关经 updateConfig 热重装 agent（自学习隐含开启技能系统）
 *  2. 「目录」列出作者 SKILL.md 与已批准学习技能
 *  3. 「自学习」蒸馏成功轨迹 → pending 提案 → 批准 / 拒绝 / 撤销
 *  4. 底部显示本轮命中（event.SKILL_SELECTED）
 */
import React from "react";
import type {
  SkillActiveDto,
  SkillCatalogDto,
  SkillLearningError,
  SkillProposalDto,
} from "../../shared/ipc-protocol.js";

type StatusFilter = "pending" | "approved" | "rejected" | "revoked" | "all";
type PanelView = "catalog" | "learning";

const STATUS_TABS: Array<{ key: StatusFilter; label: string }> = [
  { key: "pending", label: "待审批" },
  { key: "approved", label: "已批准" },
  { key: "rejected", label: "已拒绝" },
  { key: "revoked", label: "已撤销" },
  { key: "all", label: "全部" },
];

const STATUS_BADGE: Record<SkillProposalDto["status"], string> = {
  pending: "bg-ambertint text-amberdeep",
  approved: "bg-mossmist text-moss",
  rejected: "bg-[#f8e9e4] text-signal",
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
  /** 技能注入是否已装配。 */
  skillsEnabled: boolean;
  /** 自学习闭环是否已装配。 */
  learningEnabled: boolean;
}

export const SkillPanel: React.FC<SkillPanelProps> = ({ skillsEnabled, learningEnabled }) => {
  const [view, setView] = React.useState<PanelView>(learningEnabled ? "learning" : "catalog");
  const [filter, setFilter] = React.useState<StatusFilter>("pending");
  const [proposals, setProposals] = React.useState<SkillProposalDto[]>([]);
  const [catalog, setCatalog] = React.useState<SkillCatalogDto[]>([]);
  const [active, setActive] = React.useState<SkillActiveDto[]>([]);
  const [showActive, setShowActive] = React.useState(false);
  const [busy, setBusy] = React.useState<string | undefined>();
  const [error, setError] = React.useState<string | undefined>();
  const [lastDistillSummary, setLastDistillSummary] = React.useState<string | undefined>();
  const [recentHits, setRecentHits] = React.useState<Array<{ id: string; score: number }>>([]);

  React.useEffect(() => {
    setView(learningEnabled ? "learning" : "catalog");
  }, [learningEnabled]);

  const refreshCatalog = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.skillsCatalog) return;
    try {
      const r = await api.skillsCatalog();
      setCatalog(r.skills);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const refreshList = React.useCallback(
    async (nextFilter: StatusFilter = filter): Promise<void> => {
      const api = window.openintj;
      if (!api || !learningEnabled) return;
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
    [learningEnabled, filter],
  );

  const refreshActive = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api || !learningEnabled) return;
    try {
      const r = await api.skillsActive();
      if (isError(r)) return;
      setActive(r.skills);
    } catch {
      // ignore
    }
  }, [learningEnabled]);

  React.useEffect(() => {
    if (!skillsEnabled) {
      setCatalog([]);
      return;
    }
    void refreshCatalog();
  }, [skillsEnabled, refreshCatalog]);

  React.useEffect(() => {
    if (!learningEnabled) {
      setProposals([]);
      return;
    }
    void refreshList(filter);
  }, [learningEnabled, filter, refreshList]);

  React.useEffect(() => {
    if (!learningEnabled || !showActive) return;
    void refreshActive();
  }, [learningEnabled, showActive, refreshActive]);

  React.useEffect(() => {
    const api = window.openintj;
    if (!api?.onSkillEvent) return;
    return api.onSkillEvent((p) => {
      const payload = (p ?? {}) as {
        kind?: string;
        skills?: Array<{ id: string; score: number }>;
      };
      if (payload.kind === "selected" && payload.skills) {
        setRecentHits(payload.skills);
      }
      if (payload.kind === "proposed" && learningEnabled) {
        void refreshList("pending");
        setFilter("pending");
        setView("learning");
      }
    });
  }, [learningEnabled, refreshList]);

  const handleEnable = async (learning: boolean): Promise<void> => {
    const api = window.openintj;
    if (!api) return;
    setBusy("enable");
    setError(undefined);
    try {
      const r = await api.updateConfig(
        learning ? { enableSkills: true, enableSkillLearning: true } : { enableSkills: true },
      );
      if (isError(r) || (typeof r === "object" && r !== null && "error" in r)) {
        const err = r as { error: string; message?: string };
        setError(err.message ? `${err.error}: ${err.message}` : err.error);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(undefined);
    }
  };

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
      setView("learning");
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
      await refreshCatalog();
      if ((decision === "approve" || decision === "revoke") && showActive) {
        await refreshActive();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(undefined);
    }
  };

  const enableControls = (
    <div className="flex flex-wrap items-center gap-2">
      {!skillsEnabled ? (
        <button
          type="button"
          onClick={() => void handleEnable(false)}
          disabled={busy === "enable"}
          className="px-2 py-1 text-xs rounded bg-ink hover:bg-inksoft disabled:bg-rail text-white"
        >
          {busy === "enable" ? "启用中…" : "启用技能系统"}
        </button>
      ) : (
        <span className="text-[11px] text-moss">技能系统已开</span>
      )}
      {!learningEnabled ? (
        <button
          type="button"
          onClick={() => void handleEnable(true)}
          disabled={busy === "enable"}
          className="px-2 py-1 text-xs rounded bg-ink hover:bg-inksoft disabled:bg-rail text-white"
        >
          {busy === "enable" ? "启用中…" : "开启自学习"}
        </button>
      ) : (
        <span className="text-[11px] text-moss">自学习已开</span>
      )}
    </div>
  );

  if (!skillsEnabled && !learningEnabled) {
    return (
      <div
        className="h-full flex items-center justify-center p-6 text-center"
        data-testid="skill-panel"
      >
        <div className="text-sm text-muted space-y-3">
          <div className="text-base text-muted">技能系统未启用</div>
          <div>开启后把作者能力包按需注入对话；自学习会从成功轨迹蒸馏候选技能，批准后才生效。</div>
          {enableControls}
          <div className="text-xs text-muted">也可在「设置」里开关技能系统 / 技能自学习。</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full min-h-0" data-testid="skill-panel">
      <div className="shrink-0 flex items-center gap-2 px-3 py-2 border-b border-line">
        <div className="flex items-center gap-1 text-xs">
          <button
            type="button"
            onClick={() => setView("catalog")}
            className={
              view === "catalog"
                ? "px-2 py-0.5 rounded bg-rail text-ink"
                : "px-2 py-0.5 rounded text-muted hover:text-ink"
            }
          >
            目录
            {catalog.length > 0 ? (
              <span className="ml-1 text-muted">{catalog.length}</span>
            ) : null}
          </button>
          <button
            type="button"
            onClick={() => setView("learning")}
            className={
              view === "learning"
                ? "px-2 py-0.5 rounded bg-rail text-ink"
                : "px-2 py-0.5 rounded text-muted hover:text-ink"
            }
          >
            自学习
          </button>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {enableControls}
          {learningEnabled ? (
            <button
              type="button"
              onClick={() => void handleDistill()}
              disabled={busy === "distill"}
              className="px-2 py-1 text-xs rounded bg-ink hover:bg-inksoft disabled:bg-rail text-white"
            >
              {busy === "distill" ? "蒸馏中…" : "蒸馏"}
            </button>
          ) : null}
        </div>
      </div>

      {error ? (
        <div className="mx-3 mt-2 px-2 py-1 text-xs bg-signaltint text-signaldeep rounded">
          {error}
        </div>
      ) : null}

      {view === "catalog" ? (
        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2 text-xs">
          {catalog.length === 0 ? (
            <div className="text-muted">暂无已装技能</div>
          ) : (
            catalog.map((s) => (
              <div key={s.id} className="border border-line rounded p-2 space-y-1 bg-white">
                <div className="flex items-center gap-2">
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] ${
                      s.origin === "learned"
                        ? "bg-mosstint text-mossdeep"
                        : "bg-mosstint text-mossdeep"
                    }`}
                  >
                    {s.origin === "learned" ? "学习" : "作者"}
                  </span>
                  <span className="text-ink font-medium">{s.name}</span>
                  <span className="ml-auto text-muted text-[10px]">v{s.version}</span>
                </div>
                <div className="text-muted">{s.description}</div>
                <div className="text-muted text-[11px]">
                  <code className="text-moss">{s.id}</code>
                  {s.tools.length > 0 ? (
                    <>
                      <span className="mx-1 text-muted">·</span>
                      <span>工具 {s.tools.join(", ")}</span>
                    </>
                  ) : null}
                </div>
                {s.triggers.length > 0 ? (
                  <div className="text-muted text-[10px]">触发：{s.triggers.join(" / ")}</div>
                ) : null}
              </div>
            ))
          )}
        </div>
      ) : !learningEnabled ? (
        <div className="flex-1 flex items-center justify-center p-6 text-center text-sm text-muted">
          <div className="space-y-3">
            <div>自学习未启用。蒸馏候选技能需要人工批准后才会注入。</div>
            {enableControls}
          </div>
        </div>
      ) : (
        <>
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
          {lastDistillSummary ? (
            <div className="mx-3 mt-2 px-2 py-1 text-xs text-muted bg-inset/40 rounded">
              {lastDistillSummary}
            </div>
          ) : null}
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2 text-xs">
            {proposals.length === 0 ? (
              <div className="text-muted">
                {filter === "pending"
                  ? "暂无待审批提案 — 点击蒸馏从成功轨迹提炼候选技能"
                  : "暂无数据"}
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
                    <span className="text-ink font-medium">{p.name}</span>
                    <span className="ml-auto text-muted text-[11px]">{formatTime(p.ts)}</span>
                  </div>
                  <div className="text-muted">{p.description}</div>
                  <div className="text-muted text-[11px]">
                    <code className="text-moss">{p.skillId}</code>
                    <span className="mx-1 text-muted">·</span>
                    <span>命中 {p.evidence.count} 次</span>
                    {p.evidence.taskType ? (
                      <>
                        <span className="mx-1 text-muted">·</span>
                        <span>{p.evidence.taskType}</span>
                      </>
                    ) : null}
                  </div>
                  {p.evidence.queries.length > 0 ? (
                    <div className="text-muted text-[10px] truncate">
                      例：{p.evidence.queries.slice(0, 3).join(" / ")}
                    </div>
                  ) : null}
                  {p.status === "pending" ? (
                    <div className="flex items-center gap-2 pt-1">
                      <button
                        type="button"
                        onClick={() => void handleDecision(p.proposalId, "approve")}
                        disabled={busy === `approve:${p.proposalId}`}
                        className="px-2 py-0.5 text-xs rounded bg-moss hover:bg-mossdeep disabled:bg-rail text-white"
                      >
                        ✓ 批准
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
                  ) : p.status === "approved" ? (
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
                          批准于 {formatTime(p.decidedAt)}
                        </span>
                      ) : null}
                    </div>
                  ) : p.decidedAt ? (
                    <div className="text-muted text-[10px]">
                      决策于 {formatTime(p.decidedAt)}
                    </div>
                  ) : null}
                </div>
              ))
            )}
          </div>
          <div className="shrink-0 border-t border-line">
            <button
              type="button"
              onClick={() => setShowActive((v) => !v)}
              className="w-full px-3 py-1.5 text-xs text-left text-muted hover:bg-rail"
            >
              {showActive ? "▼" : "▶"} 生效学习技能
              {active.length > 0 ? (
                <span className="ml-2 text-muted">{active.length} 个</span>
              ) : null}
            </button>
            {showActive ? (
              <div className="max-h-40 overflow-y-auto px-3 pb-2 space-y-1 text-[11px]">
                {active.length === 0 ? (
                  <div className="text-muted">暂无生效的学习技能</div>
                ) : (
                  active.map((s) => (
                    <div key={s.id} className="flex items-center gap-2 text-muted">
                      <span className="text-ink">{s.name}</span>
                      {s.source ? <span className="text-muted">[{s.source}]</span> : null}
                      <span className="ml-auto text-moss">w={s.weight.toFixed(2)}</span>
                    </div>
                  ))
                )}
              </div>
            ) : null}
          </div>
        </>
      )}

      {recentHits.length > 0 ? (
        <div className="shrink-0 border-t border-line px-3 py-1.5 text-[11px] text-muted">
          本轮命中：{recentHits.map((h) => `${h.id} (${h.score.toFixed(2)})`).join(" · ")}
        </div>
      ) : null}
    </div>
  );
};
