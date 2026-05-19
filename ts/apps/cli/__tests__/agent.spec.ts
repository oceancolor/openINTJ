import { describe, expect, it } from "vitest";
import { assembleAgent } from "../src/agent.js";

describe("assembleAgent E2E (mock LLM)", () => {
  it("answers a greeting via mock fallback in single tao iter", async () => {
    const agent = assembleAgent({ llmProvider: "mock", maxTaoIterations: 1 });
    const r = await agent.run("你好");
    expect(r.status).toBe("completed");
    expect(r.finalAnswer).toContain("OpenINTJ");
    expect(r.iterations).toBe(1);
    // memory 至少记录了 user input + assistant output
    expect(agent.memory.getStats().total).toBeGreaterThanOrEqual(2);
  });

  it("registers 4 builtin tools via toolHub", () => {
    const agent = assembleAgent({ llmProvider: "mock" });
    const names = agent.execution.toolHub.list().map((t) => t.name);
    expect(names).toEqual(
      expect.arrayContaining(["read_file", "write_file", "execute_command", "search"]),
    );
  });

  it("governance + memory + execution wired into hooks", async () => {
    const agent = assembleAgent({ llmProvider: "mock" });
    let memHits = 0;
    agent.hooks.on("event.MEMORY_LOADED", () => memHits++);
    // 先 retrieve 触发 memory event
    agent.memory.store.addShortTerm("a memory");
    await agent.memory.retrieve("a memory");
    expect(memHits).toBe(1);
  });

  it("getStatus returns mock mode by default with empty key", () => {
    const agent = assembleAgent({ llmProvider: "mock" });
    const s = agent.llm.getStatus();
    expect(s.mode).toBe("mock");
    expect(s.status).toBe("missing_api_key");
  });
});
