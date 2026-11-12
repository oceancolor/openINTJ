import React from "react";

/**
 * 应用内命名对话框，替代 Electron 不支持的 window.prompt（会静默
 * 返回 null，按钮点了没反应）。Enter 确认、Esc/取消关闭。
 */
export const NameDialog: React.FC<{
  title: string;
  label?: string | undefined;
  initialValue: string;
  confirmLabel?: string | undefined;
  onSubmit: (value: string) => void;
  onCancel: () => void;
}> = (props) => {
  const [value, setValue] = React.useState(props.initialValue);
  const inputRef = React.useRef<HTMLInputElement | null>(null);

  React.useEffect(() => {
    setValue(props.initialValue);
    const id = window.setTimeout(() => {
      inputRef.current?.focus();
      inputRef.current?.select();
    }, 0);
    return () => window.clearTimeout(id);
  }, [props.initialValue, props.title]);

  const submit = (): void => {
    const trimmed = value.trim();
    if (trimmed) props.onSubmit(trimmed);
  };

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-ink/30"
      onClick={props.onCancel}
      onKeyDown={(event) => {
        if (event.key === "Escape") props.onCancel();
      }}
    >
      <div
        className="w-[22rem] rounded-2xl border border-line bg-white p-4 shadow-pop"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="text-sm font-semibold text-ink">{props.title}</div>
        {props.label ? <div className="mt-1 text-xs text-muted">{props.label}</div> : null}
        <input
          ref={inputRef}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") submit();
            if (event.key === "Escape") props.onCancel();
          }}
          className="mt-3 w-full rounded-lg border border-linedark px-3 py-2 text-sm text-ink focus:border-moss focus:outline-none"
        />
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
            disabled={!value.trim()}
            className="rounded-lg bg-ink px-3 py-1.5 font-medium text-white hover:bg-inksoft disabled:opacity-40"
            onClick={submit}
          >
            {props.confirmLabel ?? "确定"}
          </button>
        </div>
      </div>
    </div>
  );
};
