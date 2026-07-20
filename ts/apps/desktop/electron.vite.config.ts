import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, externalizeDepsPlugin } from "electron-vite";

const workspacePackages = [
  "@openintj/classifier",
  "@openintj/concurrency",
  "@openintj/core",
  "@openintj/dormant",
  "@openintj/llm-hunyuan",
  "@openintj/llm-ollama",
  "@openintj/model-runtime",
  "@openintj/plane-control",
  "@openintj/plane-execution",
  "@openintj/plane-governance",
  "@openintj/plane-memory",
  "@openintj/shared",
  "@openintj/skills",
  "@openintj/storage-lance",
  "@openintj/storage-sqlite",
  "@openintj/taskpool",
  "@openintj/telemetry-otel",
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
  },
});
