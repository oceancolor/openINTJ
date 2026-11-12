import React from "react";

/**
 * 应用内确认对话框，替代 window.confirm（与 window.prompt 一样属
 * 原生对话框，在 Electron 里行为不可靠，统一收口到应用内实现）。
 */
export const ConfirmDialog: React.FC<{
  title: string;
  message: string;
  confirmLabel?: string | undefined;
  danger?: boolean | undefined;
  onConfirm: () => void;
  onCancel: () => void;
}> = (props) => (
  <div className="fixed inset-0 z-[70] flex items-center justify-center bg-ink/30" onClick={props.onCancel}>
    <div
      className="w-[24rem] rounded-2xl border border-line bg-white p-4 shadow-pop"
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => {
        if (event.key === "Escape") props.onCancel();
      }}
    >
      <div className="text-sm font-semibold text-ink">{props.title}</div>
      <p className="mt-1.5 text-xs leading-5 text-muted">{props.message}</p>
      <div className="mt-4 flex justify-end gap-2 text-xs">
        <button
          type="button"
          className="rounded-lg border border-line px-3 py-1.5 text-muted hover:text-ink"
          onClick={props.onCancel}
        >
          取消
        </button>
        <button
          type="button"
          autoFocus
          className={
            props.danger
              ? "rounded-lg bg-signal px-3 py-1.5 font-medium text-white hover:bg-signaldeep"
              : "rounded-lg bg-ink px-3 py-1.5 font-medium text-white hover:bg-inksoft"
          }
          onClick={props.onConfirm}
        >
          {props.confirmLabel ?? "确定"}
        </button>
      </div>
    </div>
  </div>
);
