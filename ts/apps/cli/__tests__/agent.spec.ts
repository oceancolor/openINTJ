import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { assembleAgent } from "../src/agent.js";

const tmpDirs: string[] = [];
afterAll(() => {
  for (const d of tmpDirs) {
    try {
      rmSync(d, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
});

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

  it("真实工作区工具：write_file → read_file 往返，越界被拒", async () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-cli-ws-"));
    tmpDirs.push(dir);
    const agent = assembleAgent({ llmProvider: "mock", workspaceDir: dir });
    const w = await agent.execution.toolHub.call("write_file", {
      path: "sub/x.txt",
      content: "hi there",
    });
    expect(w.success).toBe(true);
    const r = await agent.execution.toolHub.call("read_file", { path: "sub/x.txt" });
    expect(r.success).toBe(true);
    expect((r.output as { content: string }).content).toBe("hi there");
    // 越界路径必须被沙箱拒绝（success=false，非崩溃）
    const blocked = await agent.execution.toolHub.call("read_file", { path: "../../escape.txt" });
    expect(blocked.success).toBe(false);
    expect(blocked.error).toMatch(/越界/);
    // 命令默认禁用
    const cmd = await agent.execution.toolHub.call("execute_command", { command: "echo hi" });
    expect(cmd.success).toBe(false);
    expect(cmd.error).toMatch(/未启用/);
  });

  it("自一致性：samples>1 时用 forkJoin 并行多采样并选出答案", async () => {
    const agent = assembleAgent({
      llmProvider: "mock",
      maxTaoIterations: 1,
      selfConsistency: { samples: 3, strategy: "majority" },
    });
    let joinPayload: { fulfilled?: number; total?: number } | undefined;
    agent.hooks.on("forkjoin.afterJoin", (ctx) => {
      joinPayload = ctx.payload as { fulfilled?: number; total?: number };
    });
    const r = await agent.run("你好");
    expect(r.status).toBe("completed");
    expect(r.finalAnswer.length).toBeGreaterThan(0);
    // forkJoin 发了 afterJoin，且并行了 3 路采样
    expect(joinPayload?.total).toBe(3);
    expect(joinPayload?.fulfilled).toBe(3);
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
