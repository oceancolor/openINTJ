import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: [
      "__tests__/artifact-fallback.spec.ts",
      "__tests__/image-task.spec.ts",
      "__tests__/workspace-view.spec.ts",
      "__tests__/editor-buffers.spec.ts",
    ],
  },
});
