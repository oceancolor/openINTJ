import type React from "react";

/**
 * Ollama 未运行时的显式提示：embedding 已降级为程序哈希向量
 * （simple-sha256，无语义相似度），记忆语义检索基本失效。
 * 用户三选一：启动并自检 / 重新自检 / 明确接受降级（持久化，不再打扰）。
 * 聊天走云端模型，不受此横幅影响。
 */
export const OllamaBanner: React.FC<{
  embedLabel: string;
  busy: boolean;
  note?: string;
  onLaunch: () => void;
  onRecheck: () => void;
  onAcceptDegraded: () => void;
}> = ({ embedLabel, busy, note, onLaunch, onRecheck, onAcceptDegraded }) => (
  <div className="shrink-0 border-b border-line bg-panel px-3 py-3" data-testid="ollama-banner">
    <div className="rounded-xl border border-signalborder bg-signalwash p-3">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-signaltint text-xs font-bold text-signaldeep">
          !
        </span>
        <span className="min-w-0 flex-1 text-xs font-semibold text-ink">
          Ollama 未运行，记忆语义检索已降级
        </span>
        <span className="shrink-0 rounded-full bg-signaltint px-2 py-0.5 font-mono text-[10px] font-semibold text-signaldeep">
          {embedLabel}
        </span>
      </div>
      <div className="mt-2 ml-8 text-[11px] leading-4 text-muted">
        当前 embedding 是程序哈希向量（不是模型）：相同文字才命中，语义相近的记忆召不回来。
        聊天（云端模型）不受影响。启动 Ollama 后会自检并自动切回真实向量；也可以明确选择降级继续用。
      </div>
      {note ? (
        <div className="mt-2 ml-8 break-all font-mono text-[10px] leading-4 text-signaldeep">
          {note}
        </div>
      ) : null}
      <div className="mt-2.5 ml-8 flex items-center gap-2">
        <button
          type="button"
          disabled={busy}
          className="rounded-lg bg-ink px-2.5 py-1.5 text-xs font-medium text-white hover:bg-inksoft disabled:opacity-50"
          onClick={onLaunch}
        >
          {busy ? "自检中…" : "启动 Ollama 并自检"}
        </button>
        <button
          type="button"
          disabled={busy}
          className="rounded-lg border border-line bg-white px-2.5 py-1.5 text-xs font-medium text-ink hover:border-linedark disabled:opacity-50"
          onClick={onRecheck}
        >
          重新自检
        </button>
        <button
          type="button"
          disabled={busy}
          className="ml-auto rounded-lg px-2.5 py-1.5 text-xs font-medium text-signaldeep hover:bg-signaltint disabled:opacity-50"
          onClick={onAcceptDegraded}
        >
          降级使用，不再提醒
        </button>
      </div>
    </div>
  </div>
);
