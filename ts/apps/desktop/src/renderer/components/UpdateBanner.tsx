import React from "react";

interface UpdateState {
  status:
    | "checking"
    | "available"
    | "not-available"
    | "downloading"
    | "downloaded"
    | "error"
    | "disabled";
  version?: string;
  percent?: number;
  message?: string;
}

/**
 * 自动更新状态条（#6）。
 * 仅在「有可用更新 / 下载中 / 已下载 / 出错」时显示；下载完成后提供「重启安装」。
 */
export const UpdateBanner: React.FC = () => {
  const [state, setState] = React.useState<UpdateState | undefined>();

  React.useEffect(() => {
    const api = window.openintj;
    if (!api?.onUpdateEvent) return;
    return api.onUpdateEvent((p) => setState((p ?? {}) as UpdateState));
  }, []);

  if (!state) return null;
  const { status } = state;
  if (status === "checking" || status === "not-available" || status === "disabled") return null;

  const install = (): void => {
    void window.openintj?.updateInstall?.();
  };

  let text = "";
  if (status === "available") text = `发现新版本 ${state.version ?? ""}，正在后台下载…`;
  else if (status === "downloading") text = `下载更新中… ${state.percent ?? 0}%`;
  else if (status === "downloaded") text = `新版本 ${state.version ?? ""} 已就绪`;
  else if (status === "error") text = `更新检查失败：${state.message ?? "未知错误"}`;

  const tone =
    status === "error"
      ? "bg-red-900/60 text-red-200 border-red-700"
      : status === "downloaded"
        ? "bg-emerald-900/60 text-emerald-100 border-emerald-700"
        : "bg-blue-900/50 text-blue-100 border-blue-700";

  return (
    <div className={`shrink-0 px-4 py-1.5 text-xs flex items-center gap-3 border-b ${tone}`}>
      <span className="flex-1">{text}</span>
      {status === "downloaded" ? (
        <button
          type="button"
          onClick={install}
          className="px-2 py-0.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-[11px]"
        >
          重启安装
        </button>
      ) : null}
    </div>
  );
};
