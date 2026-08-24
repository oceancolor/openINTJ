/**
 * Live Tavily/Brave search harness (RUN_SEARCH_LIVE=1, not in normal CI).
 *
 *   RUN_SEARCH_LIVE=1 pnpm --filter @openintj/plane-execution test web-search.live
 *
 * Requires TAVILY_API_KEY or BRAVE_API_KEY (or OPENINTJ_SEARCH_PROVIDER + OPENINTJ_SEARCH_API_KEY).
 */
import { describe, expect, it } from "vitest";
import { createWebSearchTool, resolveWebSearchConfig } from "../src/web-search-tool.js";

const RUN = process.env["RUN_SEARCH_LIVE"] === "1";

describe("web search live harness (gated)", () => {
  it.runIf(RUN)("real provider returns at least one http(s) result", async () => {
    const cfg = resolveWebSearchConfig();
    expect(cfg, "set TAVILY_API_KEY or BRAVE_API_KEY before RUN_SEARCH_LIVE=1").toBeDefined();
    const tool = createWebSearchTool(cfg!);
    const output = (await tool({ query: "OpenAI GPT-4 release date" })) as {
      ok: boolean;
      provider: string;
      results: Array<{ title?: string; url?: string }>;
      error?: string;
    };
    expect(output.ok, output.error).toBe(true);
    expect(output.results.length).toBeGreaterThan(0);
    expect(output.results.some((item) => /^https?:\/\//.test(item.url ?? ""))).toBe(true);
  });
});
