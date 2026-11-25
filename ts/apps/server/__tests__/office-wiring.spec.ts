import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type ServerAgent, assembleServerAgent } from "../src/agent.js";

describe("P1-3 office tools wiring (server)", () => {
  let agent: ServerAgent;
  beforeEach(async () => {
    agent = await assembleServerAgent({ llmProvider: "mock" });
  });
  afterEach(async () => {
    await agent.close();
  });

  it("has create_docx / create_pptx / create_xlsx", () => {
    expect(agent.execution.toolHub.has("create_docx")).toBe(true);
    expect(agent.execution.toolHub.has("create_pptx")).toBe(true);
    expect(agent.execution.toolHub.has("create_xlsx")).toBe(true);
  });
});
