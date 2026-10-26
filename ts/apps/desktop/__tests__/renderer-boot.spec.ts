import { describe, expect, it } from "vitest";
import {
  keepDevWindowOnRestart,
  resolveRendererLoad,
  shouldFallbackToRendererFile,
} from "../src/main/renderer-boot.js";

describe("keepDevWindowOnRestart", () => {
  it("keeps the Vite-backed window in unpackaged electron-vite dev", () => {
    expect(
      keepDevWindowOnRestart({
        isPackaged: false,
        rendererUrl: "http://localhost:5173/",
      }),
    ).toBe(true);
  });

  it("relaunches the process when packaged", () => {
    expect(
      keepDevWindowOnRestart({
        isPackaged: true,
        rendererUrl: "http://localhost:5173/",
      }),
    ).toBe(false);
  });

  it("relaunches preview/out builds that load a file URL", () => {
    expect(keepDevWindowOnRestart({ isPackaged: false, rendererUrl: undefined })).toBe(false);
    expect(keepDevWindowOnRestart({ isPackaged: false, rendererUrl: "  " })).toBe(false);
  });
});

describe("resolveRendererLoad", () => {
  it("prefers ELECTRON_RENDERER_URL when set", () => {
    expect(
      resolveRendererLoad({
        rendererUrl: "http://localhost:5173/",
        rendererFile: "/tmp/index.html",
      }),
    ).toEqual({ kind: "url", url: "http://localhost:5173/" });
  });

  it("falls back to the built renderer file", () => {
    expect(
      resolveRendererLoad({
        rendererUrl: undefined,
        rendererFile: "/tmp/index.html",
      }),
    ).toEqual({ kind: "file", file: "/tmp/index.html" });
  });
});

describe("shouldFallbackToRendererFile", () => {
  it("recovers from a dead Vite URL on the main frame", () => {
    expect(
      shouldFallbackToRendererFile({
        isMainFrame: true,
        errorCode: -102,
        failedUrl: "http://localhost:5173/",
      }),
    ).toBe(true);
  });

  it("does not loop on aborted navigations or file:// failures", () => {
    expect(
      shouldFallbackToRendererFile({
        isMainFrame: true,
        errorCode: -3,
        failedUrl: "http://localhost:5173/",
      }),
    ).toBe(false);
    expect(
      shouldFallbackToRendererFile({
        isMainFrame: true,
        errorCode: -6,
        failedUrl: "file:///tmp/index.html",
      }),
    ).toBe(false);
    expect(
      shouldFallbackToRendererFile({
        isMainFrame: false,
        errorCode: -102,
        failedUrl: "http://localhost:5173/assets/x.js",
      }),
    ).toBe(false);
  });
});
