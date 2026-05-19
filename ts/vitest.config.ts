import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: false,
    environment: "node",
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
