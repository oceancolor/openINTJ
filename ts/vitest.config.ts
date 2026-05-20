import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  test: {
    globals: false,
    environment: "node",
    globalSetup: [resolve(__dirname, "vitest.global-setup.ts")],
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      exclude: [
        "**/node_modules/**",
        "**/dist/**",
        "**/out/**",
        "**/release/**",
        "**/__tests__/**",
        "**/*.config.ts",
        "**/index.ts",
      ],
    },
    include: ["**/__tests__/**/*.spec.ts", "**/*.test.ts"],
  },
});
