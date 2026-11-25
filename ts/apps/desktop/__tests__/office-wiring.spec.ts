import { describe, expect, it } from "vitest";
import { assembleDesktopAgent } from "../src/main/agent.js";

describe("P1-3 office tools wiring (desktop)", () => {
  it(
    "has create_docx / create_pptx / create_xlsx",
    async () => {
      const agent = await assembleDesktopAgent({ llmProvider: "mock" });
      try {
        expect(agent.execution.toolHub.has("create_docx")).toBe(true);
        expect(agent.execution.toolHub.has("create_pptx")).toBe(true);
        expect(agent.execution.toolHub.has("create_xlsx")).toBe(true);
      } finally {
        await agent.close();
      }
    },
    20_000,
  );
});
