import React from "react";

interface MemoryHit {
  fragmentId: string;
  content: string;
  score?: number;
  memoryType: string;
  taskTags: string[];
}

interface MemoryStatusView {
  canonicalEnabled: boolean;
  activeCount: number;
  tombstonedCount: number;
  quarantinedCount: number;
  supersededCount: number;
  suppressedCount: number;
  memoryEpoch: number;
  index: { pending: number; failed: number; generation: number; dimensionCompatible?: boolean };
  writerStats?: Array<{
    policyVersion: string;
    processedTurns: number;
    emptyTurns: number;
    deadLetters: number;
    candidatesNominated: number;
    startedAt: number;
    updatedAt: number;
  }>;
}

interface CandidateView {
  candidateId: string;
  receiveKey: string;
  status: string;
  reasons: string[];
  candidateJson: string;
  updatedAt: number;
}

interface CandidateBody {
  operation?: string;
  kind?: string;
  content?: string;
  rationale?: string;
  sourceRefs?: string[];
}

const parseCandidate = (json: string): CandidateBody => {
  try {
    return JSON.parse(json) as CandidateBody;
  } catch {
    return {};
  }
};

const STATUS_LABEL: Record<string, string> = {
  shadow_only: "待审核",
  applied: "已通过生效",
  dismissed: "已驳回",
  rejected: "校验拒绝",
  conflicted: "版本冲突",
};

/**
 * 记忆面板（G5.2 扩展为审核台）：
 * - 顶部 F14 状态面：canonical/索引/epoch 一次读清（memoryStatus）
 * - 中部影子候选审核：仅 shadow_only 可操作；通过由主进程 host 执行
 *   mutation 生效，驳回记 dismissed；其余状态只读展示原因
 * - 下部保留原语义检索（memoryQuery）
 */
export const MemoryPanel: React.FC = () => {
  const [query, setQuery] = React.useState("");
  const [hits, setHits] = React.useState<MemoryHit[] | undefined>();
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | undefined>();

  const [status, setStatus] = React.useState<MemoryStatusView | undefined>();
  const [candidates, setCandidates] = React.useState<CandidateView[]>([]);
  const [reviewError, setReviewError] = React.useState<string | undefined>();
  const [busyId, setBusyId] = React.useState<string | undefined>();

  const refreshReview = React.useCallback(async (): Promise<void> => {
    const api = window.openintj;
    if (!api?.memoryStatus || !api?.memoryCandidates) return;
    try {
      const [s, c] = await Promise.all([api.memoryStatus(), api.memoryCandidates()]);
      setStatus(s as MemoryStatusView);
      setCandidates(c as CandidateView[]);
    } catch (e) {
      setReviewError((e as Error).message);
    }
  }, []);

  React.useEffect(() => {
    void refreshReview();
  }, [refreshReview]);

  const decide = React.useCallback(
    async (candidateId: string, decision: "approve" | "reject"): Promise<void> => {
      const api = window.openintj;
      if (!api) return;
      setBusyId(candidateId);
      setReviewError(undefined);
      try {
        const res =
          decision === "approve"
            ? await api.memoryCandidateApprove(candidateId)
            : await api.memoryCandidateReject(candidateId);
        if ("error" in res) setReviewError(res.error);
        await refreshReview();
      } catch (e) {
        setReviewError((e as Error).message);
      } finally {
        setBusyId(undefined);
      }
    },
    [refreshReview],
  );

  const search = React.useCallback(async (q: string): Promise<void> => {
    const api = window.openintj;
    if (!api?.memoryQuery) return;
    setLoading(true);
    setError(undefined);
    try {
      const res = await api.memoryQuery(q.trim() ? { query: q.trim(), topK: 10 } : { topK: 20 });
      setHits(res as MemoryHit[]);
    } catch (e) {
      setError((e as Error).message);
      setHits([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const pending = candidates.filter((c) => c.status === "shadow_only");
  const decided = candidates.filter((c) => c.status !== "shadow_only");

  return (
    <div className="flex flex-col h-full min-h-0 text-xs">
      <div className="shrink-0 p-2 border-b border-line space-y-2">
        <div className="flex items-center gap-2">
          <span className="font-medium text-ink">记忆状态</span>
          <button
            type="button"
            onClick={() => void refreshReview()}
            className="px-2 py-0.5 rounded border border-linedark text-muted hover:text-ink"
          >
            刷新
          </button>
          {status ? (
            <span className={status.canonicalEnabled ? "text-moss" : "text-signal"}>
              canonical {status.canonicalEnabled ? "已启用" : "未启用（降级）"}
            </span>
          ) : null}
        </div>
        {status ? (
          <div className="text-muted leading-relaxed">
            有效 {status.activeCount} · 已遗忘 {status.tombstonedCount} · 隔离 {status.quarantinedCount} ·
            已取代 {status.supersededCount} · 抑制 {status.suppressedCount} · epoch {status.memoryEpoch} ·
            索引待投影 {status.index.pending}/失败 {status.index.failed}（第 {status.index.generation} 代）
          </div>
        ) : (
          <div className="text-muted">状态加载中…</div>
        )}
        {status?.writerStats && status.writerStats.length > 0
          ? status.writerStats.map((w) => (
              <div key={w.policyVersion} className="text-muted leading-relaxed">
                影子 writer[{w.policyVersion}]：已处理 {w.processedTurns} 轮 · 空转 {w.emptyTurns} · 死信{" "}
                {w.deadLetters} · 提名 {w.candidatesNominated} 条（自{" "}
                {new Date(w.startedAt).toLocaleDateString()}）
              </div>
            ))
          : null}
        {reviewError ? <div className="text-signal">[审核错误] {reviewError}</div> : null}
      </div>

      <div className="shrink-0 max-h-64 overflow-y-auto p-2 border-b border-line space-y-2">
        <div className="text-muted">待审候选（{pending.length}）——通过后才由系统写入正式记忆</div>
        {pending.length === 0 ? (
          <div className="text-muted">暂无待审候选。对话中说「记住：…」或「我偏好…」会产生影子候选。</div>
        ) : (
          pending.map((c) => {
            const body = parseCandidate(c.candidateJson);
            return (
              <div key={c.candidateId} className="border-l border-linedark pl-2">
                <div className="flex items-center gap-2 text-[10px] text-muted">
                  <span className="px-1 rounded bg-inset text-muted">{body.kind ?? "?"}</span>
                  <span>{body.operation ?? "add"}</span>
                  {body.sourceRefs ? <span>证据 {body.sourceRefs.length} 条</span> : null}
                </div>
                <div className="text-ink whitespace-pre-wrap break-words mt-0.5">{body.content ?? ""}</div>
                {body.rationale ? <div className="text-muted mt-0.5">理由：{body.rationale}</div> : null}
                <div className="flex gap-2 mt-1">
                  <button
                    type="button"
                    disabled={busyId === c.candidateId}
                    onClick={() => void decide(c.candidateId, "approve")}
                    className="px-2 py-0.5 rounded bg-ink hover:bg-inksoft disabled:opacity-50 text-white"
                  >
                    通过并写入
                  </button>
                  <button
                    type="button"
                    disabled={busyId === c.candidateId}
                    onClick={() => void decide(c.candidateId, "reject")}
                    className="px-2 py-0.5 rounded border border-linedark text-muted hover:text-ink disabled:opacity-50"
                  >
                    驳回
                  </button>
                </div>
              </div>
            );
          })
        )}
        {decided.length > 0 ? (
          <div className="space-y-1 pt-1">
            {decided.map((c) => {
              const body = parseCandidate(c.candidateJson);
              return (
                <div key={c.candidateId} className="text-muted">
                  <span className="px-1 rounded bg-inset">{STATUS_LABEL[c.status] ?? c.status}</span>{" "}
                  {body.content ?? ""}
                  {c.reasons.length > 0 ? <span>（{c.reasons.join(", ")}）</span> : null}
                </div>
              );
            })}
          </div>
        ) : null}
      </div>

      <div className="shrink-0 p-2 border-b border-line flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void search(query);
          }}
          placeholder="检索记忆（留空=最近片段）"
          className="flex-1 bg-white border border-linedark rounded px-2 py-1 text-ink focus:outline-none focus:border-moss"
        />
        <button
          type="button"
          onClick={() => void search(query)}
          disabled={loading}
          className="px-3 py-1 rounded bg-ink hover:bg-inksoft disabled:opacity-50 text-white"
        >
          {loading ? "…" : "检索"}
        </button>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2">
        {error ? <div className="text-signal">[错误] {error}</div> : null}
        {hits === undefined ? (
          <div className="text-muted">输入关键词检索，或留空点「检索」查看最近记忆。</div>
        ) : hits.length === 0 ? (
          <div className="text-muted">没有命中的记忆。</div>
        ) : (
          hits.map((h) => (
            <div key={h.fragmentId} className="border-l border-linedark pl-2">
              <div className="flex items-center gap-2 text-[10px] text-muted">
                <span className="px-1 rounded bg-inset text-muted">{h.memoryType}</span>
                {typeof h.score === "number" ? <span>score={h.score.toFixed(3)}</span> : null}
                {h.taskTags.length > 0 ? <span>tags: {h.taskTags.join(", ")}</span> : null}
              </div>
              <div className="text-muted whitespace-pre-wrap break-words mt-0.5">
                {h.content?.trim() ? (
                  h.content
                ) : (
                  <span className="text-muted">（仅元数据，无正文）</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
