import React from "react";
import { TITLE_BAR_HEIGHT } from "../../shared/ipc-protocol.js";
import avatarLogo from "../assets/app-icon.png";

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
      className="titlebar-drag shrink-0 w-full flex items-center text-xs text-muted select-none bg-panel border-b border-line"
      style={{ height: `env(titlebar-area-height, ${TITLE_BAR_HEIGHT}px)` }}
    >
      <div
        className="h-full flex items-center gap-1 min-w-0"
        style={{
          width: "env(titlebar-area-width, 100%)",
          // 左栏「工作区」标题为 px-3（12px）；logo 左缘与之对齐。
          paddingLeft:
            platform === "darwin"
              ? `env(titlebar-area-x, 78px)`
              : `calc(env(titlebar-area-x, 0px) + 12px)`,
        }}
      >
        {/* INTJ 紫老头头像：紫发黑框眼镜；小尺寸下放大面部保证可辨。 */}
        <div
          data-testid="title-bar-logo"
          className="titlebar-no-drag mr-1 h-[26px] w-[26px] shrink-0 overflow-hidden rounded-[7px] bg-[#e4def8] ring-1 ring-[#c9bdf0]"
        >
          <img
            src={avatarLogo}
            alt="openINTJ"
            draggable={false}
            className="h-full w-full object-cover"
          />
        </div>
        {(Object.keys(menus) as MenuId[]).map((id) => {
          const menu = menus[id];
          return (
            <div key={id} className="titlebar-no-drag relative">
              <button
                type="button"
                className={
                  open === id
                    ? "px-2 py-0.5 rounded bg-ink/5 text-ink"
                    : "px-2 py-0.5 rounded text-muted hover:bg-ink/5 hover:text-ink"
                }
                onClick={() => toggle(id)}
              >
                {menu.label}
              </button>
              {open === id ? (
                <div className="titlebar-no-drag absolute left-0 top-full z-50 min-w-[9rem] py-1 rounded-lg border border-line bg-white shadow-pop">
                  {menu.items.map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      className="block w-full px-3 py-1.5 text-left text-muted hover:bg-rail hover:text-ink"
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
              className="px-3 py-0.5 text-muted hover:bg-ink/5 hover:text-ink"
              onClick={() => void window.openintj.windowChrome("minimize")}
            >
              –
            </button>
            <button
              type="button"
              className="px-3 py-0.5 text-muted hover:bg-ink/5 hover:text-ink"
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
