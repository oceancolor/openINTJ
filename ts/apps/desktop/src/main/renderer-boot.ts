/**
 * Renderer boot helpers for Electron main.
 *
 * electron-vite dev injects ELECTRON_RENDERER_URL. `app.relaunch()` starts a
 * bare Electron process; the Vite server usually dies with the old tree, so
 * the new window loads about:blank (package name title, white body).
 */

export type RendererLoadTarget = { kind: "url"; url: string } | { kind: "file"; file: string };

export const keepDevWindowOnRestart = (opts: {
  isPackaged: boolean;
  rendererUrl: string | undefined;
}): boolean => !opts.isPackaged && Boolean(opts.rendererUrl?.trim());

export const resolveRendererLoad = (opts: {
  rendererUrl: string | undefined;
  rendererFile: string;
}): RendererLoadTarget => {
  const url = opts.rendererUrl?.trim();
  if (url) return { kind: "url", url };
  return { kind: "file", file: opts.rendererFile };
};

/** Chromium net::ERR_ABORTED — expected on reload/navigation replace. */
const ERR_ABORTED = -3;

export const shouldFallbackToRendererFile = (opts: {
  isMainFrame: boolean;
  errorCode: number;
  failedUrl: string;
}): boolean => {
  if (!opts.isMainFrame) return false;
  if (opts.errorCode === ERR_ABORTED) return false;
  return /^https?:\/\//i.test(opts.failedUrl);
};
