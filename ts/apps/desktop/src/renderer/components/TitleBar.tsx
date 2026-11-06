import React from "react";
import { TITLE_BAR_HEIGHT } from "../../shared/ipc-protocol.js";
import appIcon from "../assets/app-icon.png";

type MenuId = "file" | "edit" | "view" | "window";

interface MenuItem {
  label: string;
  onClick: () => void;
}

const runEdit = (command: string): void => {
  document.execCommand(command);
};

export const TitleBar: React.FC = () => {
  const [open, setOpen] = React.useState<MenuId | null>(null);
  const rootRef = React.useRef<HTMLElement | null>(null);
  const platform = window.openintj.hostPlatform;
  const showWindowButtons = platform !== "win32" && platform !== "darwin";

  const menus: Record<MenuId, { label: string; items: MenuItem[] }> = {
    file: {
      label: "File",
      items: [
        { label: "重启", onClick: () => void window.openintj.restartApp() },
        { label: "退出", onClick: () => void window.openintj.windowChrome("quit") },
      ],
    },
    edit: {
      label: "Edit",
      items: [
        { label: "撤销", onClick: () => runEdit("undo") },
        { label: "重做", onClick: () => runEdit("redo") },
        { label: "剪切", onClick: () => runEdit("cut") },
        { label: "复制", onClick: () => runEdit("copy") },
        { label: "粘贴", onClick: () => runEdit("paste") },
        { label: "全选", onClick: () => runEdit("selectAll") },
      ],
    },
    view: {
      label: "View",
      items: [
        { label: "重新加载", onClick: () => void window.openintj.windowChrome("reload") },
        {
          label: "开发者工具",
          onClick: () => void window.openintj.windowChrome("toggleDevTools"),
        },
        { label: "全屏", onClick: () => void window.openintj.windowChrome("toggleFullScreen") },
      ],
    },
    window: {
      label: "Window",
      items: [
        { label: "最小化", onClick: () => void window.openintj.windowChrome("minimize") },
        { label: "最大化", onClick: () => void window.openintj.windowChrome("maximize") },
        { label: "关闭", onClick: () => void window.openintj.windowChrome("close") },
      ],
    },
  };

  const toggle = (id: MenuId): void => {
    setOpen((current) => (current === id ? null : id));
  };

  React.useEffect(() => {
    if (!open) {
      window.dispatchEvent(new Event("resize"));
      return;
    }
    const onPointer = (event: MouseEvent): void => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) setOpen(null);
    };
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === "Escape") setOpen(null);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <header
      ref={rootRef}
      data-testid="title-bar"
      className="titlebar-drag shrink-0 w-full flex items-center text-xs text-white/60 select-none bg-ink"
      style={{ height: `env(titlebar-area-height, ${TITLE_BAR_HEIGHT}px)` }}
    >
      <div
        className="h-full flex items-center gap-1 min-w-0"
        style={{
          width: "env(titlebar-area-width, 100%)",
          paddingLeft: `env(titlebar-area-x, ${platform === "darwin" ? "78px" : "8px"})`,
        }}
      >
        <img
          src={appIcon}
          alt=""
          data-testid="title-bar-logo"
          className="titlebar-no-drag h-6 w-6 rounded-[6px] ml-[2px] mr-1 shrink-0 ring-1 ring-white/20"
        />
        {(Object.keys(menus) as MenuId[]).map((id) => {
          const menu = menus[id];
          return (
            <div key={id} className="titlebar-no-drag relative">
              <button
                type="button"
                className={
                  open === id
                    ? "px-2 py-0.5 rounded bg-white/10 text-white"
                    : "px-2 py-0.5 rounded text-white/60 hover:bg-white/10 hover:text-white"
                }
                onClick={() => toggle(id)}
              >
                {menu.label}
              </button>
              {open === id ? (
                <div className="titlebar-no-drag absolute left-0 top-full z-50 min-w-[9rem] py-1 rounded-lg border border-white/10 bg-ink shadow-pop">
                  {menu.items.map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      className="block w-full px-3 py-1.5 text-left text-white/75 hover:bg-white/10 hover:text-white"
                      onClick={() => {
                        item.onClick();
                        setOpen(null);
                      }}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          );
        })}
        {showWindowButtons ? (
          <div className="titlebar-no-drag ml-auto flex">
            <button
              type="button"
              className="px-3 py-0.5 text-white/60 hover:bg-white/10 hover:text-white"
              onClick={() => void window.openintj.windowChrome("minimize")}
            >
              –
            </button>
            <button
              type="button"
              className="px-3 py-0.5 text-white/60 hover:bg-white/10 hover:text-white"
              onClick={() => void window.openintj.windowChrome("maximize")}
            >
              □
            </button>
            <button
              type="button"
              className="px-3 py-0.5 text-white/60 hover:bg-signal hover:text-white"
              onClick={() => void window.openintj.windowChrome("close")}
            >
              ×
            </button>
          </div>
        ) : null}
      </div>
    </header>
  );
};
