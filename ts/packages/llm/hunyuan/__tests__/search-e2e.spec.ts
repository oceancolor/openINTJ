/**
 * search 工具「真盘」冒烟测试（联网，B）。
 *
 * 默认 skip。需要同时满足：
 *   - 环境变量 OPENINTJ_E2E=1（与仓库其它真盘 e2e 一致的总开关）
 *   - 提供真实 HUNYUAN_API_KEY（否则 client 退化 mock，无法验证联网链路）
 *
 * 运行（PowerShell）：
 *   $env:OPENINTJ_E2E=1; $env:HUNYUAN_API_KEY="sk-..."; pnpm --filter @openintj/llm-hunyuan test
 *
 * 断言取「不脆」策略：联网命中与否依赖实时结果，这里只验证链路 live 且产出非空回答；
 * 命中来源数仅打印，不做强断言（避免因当日检索结果波动而 flaky）。
 */
import { describe, expect, it } from "vitest";
import { HunyuanClient, createHunyuanSearchTool, loadHunyuanConfigFromEnv } from "../src/index.js";

const RUN_E2E =
  process.env["OPENINTJ_E2E"] === "1" && Boolean(process.env["HUNYUAN_API_KEY"]?.trim());

const describeE2E = RUN_E2E ? describe : describe.skip;

describeE2E("Hunyuan search 真盘冒烟（OPENINTJ_E2E=1 + HUNYUAN_API_KEY）", () => {
  it("client.webSearch 走 live 链路并返回非空回答", async () => {
    const client = new HunyuanClient(loadHunyuanConfigFromEnv(process.env));
    expect(client.isMockMode).toBe(false);

    const result = await client.webSearch("2026 年值得关注的 AI 进展有哪些？");
    expect(result.mode).toBe("live");
    expect(typeof result.answer).toBe("string");
    expect(result.answer.trim().length).toBeGreaterThan(0);
    // 来源数随实时检索波动，仅记录不强断言。
    console.log(
      `[search e2e] webSearch sources=${result.sources.length} answer.len=${result.answer.length}`,
    );
  }, 60_000);

  it("createHunyuanSearchTool 作为 ToolHandler 返回 ok=true", async () => {
    const client = new HunyuanClient(loadHunyuanConfigFromEnv(process.env));
    const tool = createHunyuanSearchTool(client);
    const out = (await tool({ query: "OpenINTJ 是什么类型的项目？给出最新公开信息" })) as {
      ok?: boolean;
      mode?: string;
      answer?: string;
      sources?: unknown[];
    };
    expect(out.ok).toBe(true);
    expect(out.mode).toBe("live");
    expect((out.answer ?? "").trim().length).toBeGreaterThan(0);
    console.log(`[search e2e] tool sources=${(out.sources ?? []).length}`);
  }, 60_000);
});
