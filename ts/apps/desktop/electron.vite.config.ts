import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, externalizeDepsPlugin } from "electron-vite";

const appVersion = (
  JSON.parse(readFileSync(resolve(__dirname, "package.json"), "utf8")) as { version: string }
).version;

const workspacePackages = [
  "@openintj/channels",
  "@openintj/classifier",
  "@openintj/concurrency",
  "@openintj/connectors",
  "@openintj/core",
  "@openintj/dormant",
  "@openintj/identity",
  "@openintj/llm-hunyuan",
  "@openintj/llm-ollama",
  "@openintj/model-runtime",
  "@openintj/plane-control",
  "@openintj/plane-execution",
  "@openintj/plane-governance",
  "@openintj/plane-memory",
  "@openintj/profile",
  "@openintj/runtime",
  "@openintj/scheduler",
  "@openintj/shared",
  "@openintj/skills",
  "@openintj/storage-lance",
  "@openintj/storage-sqlite",
  "@openintj/sync",
  "@openintj/taskpool",
  "@openintj/telemetry-otel",
  "@openintj/voice",
];

export default defineConfig({
  main: {
    // pnpm links workspace packages outside this app. Bundle those packages so
    // electron-builder never follows their symlinks outside appDir.
    plugins: [externalizeDepsPlugin({ exclude: workspacePackages })],
    build: {
      lib: { entry: resolve(__dirname, "src/main/index.ts") },
      outDir: "out/main",
    },
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      lib: { entry: resolve(__dirname, "src/preload/index.ts") },
      outDir: "out/preload",
    },
  },
  renderer: {
    root: resolve(__dirname, "src/renderer"),
    plugins: [react()],
    define: {
      // Single source of truth for the window/document title version.
      __OPENINTJ_APP_VERSION__: JSON.stringify(appVersion),
    },
    build: {
      outDir: "out/renderer",
      rollupOptions: {
        input: { index: resolve(__dirname, "src/renderer/index.html") },
      },
    },
    resolve: {
      alias: {
        "@": resolve(__dirname, "src/renderer"),
      },
    },
    server: {
      fs: {
        allow: [resolve(__dirname)],
      },
    },
  },
});
