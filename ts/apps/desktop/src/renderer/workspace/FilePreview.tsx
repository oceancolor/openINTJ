import React from "react";
import {
  guessedRasterFromSvgPath,
  isMarkdownPath,
  mimeForPath,
  previewKindForPath,
  rasterSiblingFromSvg,
} from "../../shared/workspace-view.js";
import { MarkdownView } from "./MarkdownView.js";
import { type WorkspaceClient, isWorkspaceError } from "./types.js";

export const FilePreview: React.FC<{
  client: WorkspaceClient;
  path: string;
  onClose?: () => void;
  onReveal?: (path: string) => void;
  showHeader?: boolean;
}> = ({ client, path, onClose, onReveal, showHeader = true }) => {
  const [state, setState] = React.useState<
    | { status: "loading" }
    | { status: "image"; src: string }
    | { status: "text"; content: string }
    | { status: "file"; bytes?: number }
    | { status: "error"; message: string }
  >({ status: "loading" });
  // Markdown rendered/source toggle + image fit/original-size zoom.
  const [mdMode, setMdMode] = React.useState<"rendered" | "source">("rendered");
  const [zoom, setZoom] = React.useState<"fit" | "actual">("fit");
  React.useEffect(() => {
    setMdMode("rendered");
    setZoom("fit");
  }, [path]);

  React.useEffect(() => {
    let cancelled = false;
    const load = async (): Promise<void> => {
      setState({ status: "loading" });
      const kind = previewKindForPath(path);
      try {
        if (kind === "image") {
          let target = path;
          let svgText: string | undefined;
          if (path.toLowerCase().endsWith(".svg")) {
            const text = await client.read(path);
            if (cancelled) return;
            if (!isWorkspaceError(text)) {
              svgText = text.content;
              target =
                rasterSiblingFromSvg(text.content, path) ?? guessedRasterFromSvgPath(path) ?? path;
            } else {
              target = guessedRasterFromSvgPath(path) ?? path;
            }
          }
          if (target !== path) {
            const bytes = await client.readBytes(target);
            if (cancelled) return;
            if (!isWorkspaceError(bytes)) {
              setState({
                status: "image",
                src: `data:${bytes.mimeType || mimeForPath(target)};base64,${bytes.base64}`,
              });
              return;
            }
          }
          if (svgText) {
            setState({
              status: "image",
              src: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`,
            });
            return;
          }
          const bytes = await client.readBytes(path);
          if (cancelled) return;
          if (isWorkspaceError(bytes)) {
            setState({ status: "error", message: bytes.message });
            return;
          }
          setState({
            status: "image",
            src: `data:${bytes.mimeType || mimeForPath(path)};base64,${bytes.base64}`,
          });
          return;
        }
        if (kind === "text") {
          const text = await client.read(path);
          if (cancelled) return;
          if (isWorkspaceError(text)) {
            setState({ status: "error", message: text.message });
            return;
          }
          setState({ status: "text", content: text.content });
          return;
        }
        setState({ status: "file" });
      } catch (error) {
        if (!cancelled) {
          setState({
            status: "error",
            message: error instanceof Error ? error.message : String(error),
          });
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [client, path]);

  return (
    <section className="h-full min-h-0 flex flex-col bg-panel" data-testid="workspace-preview">
      {showHeader ? (
        <div className="shrink-0 h-9 px-3 border-b border-line flex items-center gap-2 text-xs">
          <span className="text-[10px] font-bold uppercase tracking-[0.13em] text-muted">预览</span>
          <span className="truncate font-mono text-[11px] text-ink" title={path}>
            {path}
          </span>
          <div className="ml-auto flex items-center gap-2">
            {state.status === "text" && isMarkdownPath(path) ? (
              <>
                <button
                  type="button"
                  className={
                    mdMode === "rendered"
                      ? "font-semibold text-moss"
                      : "text-muted hover:text-ink"
                  }
                  onClick={() => setMdMode("rendered")}
                >
                  渲染
                </button>
                <button
                  type="button"
                  className={
                    mdMode === "source" ? "font-semibold text-moss" : "text-muted hover:text-ink"
                  }
                  onClick={() => setMdMode("source")}
                >
                  源码
                </button>
              </>
            ) : null}
            {state.status === "image" ? (
              <button
                type="button"
                className="text-muted hover:text-ink"
                onClick={() => setZoom((z) => (z === "fit" ? "actual" : "fit"))}
              >
                {zoom === "fit" ? "1:1" : "适应"}
              </button>
            ) : null}
            {onReveal ? (
              <button
                type="button"
                className="text-muted hover:text-ink"
                onClick={() => onReveal(path)}
              >
                打开所在目录
              </button>
            ) : null}
            {onClose ? (
              <button type="button" className="text-muted hover:text-ink" onClick={onClose}>
                关闭
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
      <div
        className={
          state.status === "image" && zoom === "fit"
            ? "flex-1 min-h-0 overflow-hidden p-3 flex items-center justify-center"
            : "flex-1 min-h-0 overflow-auto p-3"
        }
      >
        {state.status === "loading" ? (
          <div className="text-xs text-muted">加载中…</div>
        ) : state.status === "error" ? (
          <div className="text-xs text-signal">{state.message}</div>
        ) : state.status === "image" ? (
          <img
            src={state.src}
            alt={path}
            className={
              zoom === "fit" ? "max-w-full max-h-full w-auto h-auto object-contain" : "max-w-none"
            }
          />
        ) : state.status === "text" ? (
          isMarkdownPath(path) && mdMode === "rendered" ? (
            <MarkdownView text={state.content} />
          ) : (
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-5 text-ink">
              {state.content}
            </pre>
          )
        ) : (
          <div className="text-xs text-muted">此类型暂不预览，请用「打开所在目录」查看。</div>
        )}
      </div>
    </section>
  );
};
