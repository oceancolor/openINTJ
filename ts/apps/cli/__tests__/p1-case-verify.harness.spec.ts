/**
 * P1 case 实测验证（mock LLM / 本地工具，不接真实 hy4）。
 * 对照规划 §9.2 验收标准逐条跑通。
 */
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ControlPlane, GoalParser, Planner } from "@openintj/plane-control";
import { enforceProductBehaviorAnswer } from "@openintj/shared";
import { afterAll, describe, expect, it } from "vitest";
import { assembleAgent } from "../src/agent.js";
import {
  mimeForPath,
  previewKindForPath,
} from "../../desktop/src/shared/workspace-view.js";

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

const makeWs = (): string => {
  const d = mkdtempSync(join(tmpdir(), "openintj-p1-case-"));
  tmpDirs.push(d);
  return d;
};

const ZIP_MAGIC = Buffer.from([0x50, 0x4b, 0x03, 0x04]);

describe("P1 case 实测验证", () => {
  it("CASE P1-1: contentBase64 往返 + ZIP 读回 base64", async () => {
    const dir = makeWs();
    const agent = assembleAgent({ llmProvider: "mock", workspaceDir: dir });
    const src = Buffer.concat([ZIP_MAGIC, Buffer.from("case-payload")]);
    const w = await agent.execution.toolHub.call("write_file", {
      path: "bin/a.zip",
      contentBase64: src.toString("base64"),
    });
    expect(w.success).toBe(true);
    expect(Buffer.compare(readFileSync(join(dir, "bin", "a.zip")), src)).toBe(0);

    const r = await agent.execution.toolHub.call("read_file", { path: "bin/a.zip" });
    expect(r.success).toBe(true);
    const out = r.output as { encoding?: string; contentBase64?: string; content?: string };
    expect(out.encoding).toBe("base64");
    expect(out.contentBase64).toBe(src.toString("base64"));
    expect(out.content).toBeUndefined();
  });

  it("CASE P1-2/P1-3: create_docx/pptx/xlsx 魔数 + 三端装配 has()", async () => {
    const dir = makeWs();
    const agent = assembleAgent({ llmProvider: "mock", workspaceDir: dir });
    expect(agent.execution.toolHub.has("create_docx")).toBe(true);
    expect(agent.execution.toolHub.has("create_pptx")).toBe(true);
    expect(agent.execution.toolHub.has("create_xlsx")).toBe(true);

    const doc = await agent.execution.toolHub.call("create_docx", {
      path: "out/report",
      title: "实测报告",
      body: ["第一段", "第二段"],
    });
    expect(doc.success).toBe(true);
    const docPath = join(dir, "out", "report.docx");
    expect(Buffer.compare(readFileSync(docPath).subarray(0, 4), ZIP_MAGIC)).toBe(0);

    const ppt = await agent.execution.toolHub.call("create_pptx", {
      path: "out/deck.pptx",
      slides: [{ title: "封面", bullets: ["A", "B"] }],
    });
    expect(ppt.success).toBe(true);
    expect(
      Buffer.compare(readFileSync(join(dir, "out", "deck.pptx")).subarray(0, 4), ZIP_MAGIC),
    ).toBe(0);

    const xls = await agent.execution.toolHub.call("create_xlsx", {
      path: "out/sales.xlsx",
      sheets: [
        {
          name: "Sales",
          rows: [
            ["Item", "Qty", "Price", "Total"],
            ["Widget", 2, 10, { formula: "B2*C2", result: 20 }],
          ],
        },
      ],
    });
    expect(xls.success).toBe(true);
    expect(
      Buffer.compare(readFileSync(join(dir, "out", "sales.xlsx")).subarray(0, 4), ZIP_MAGIC),
    ).toBe(0);

    // utf8 写 .docx 仍降级；base64/Office 工具不降级
    const fake = await agent.execution.toolHub.call("write_file", {
      path: "fake.docx",
      content: "# 假文档",
    });
    expect(fake.success).toBe(true);
    expect((fake.output as { path: string }).path).toBe("fake.md");
  });

  it("CASE P1-4: 规划扇出 ready≥2 + dispatchPlanStep→ToolHub.call", async () => {
    const dir = makeWs();
    writeFileSync(join(dir, "README.md"), "# hello\n");
    const agent = assembleAgent({ llmProvider: "mock", workspaceDir: dir });
    const planner = new Planner();
    const goal = new GoalParser().parse("查询项目状态");
    const plan = planner.createPlan(goal);
    const ready = plan.getReadySteps();
    expect(ready.length).toBeGreaterThanOrEqual(2);

    const cp = new ControlPlane();
    const searchStep = ready.find((s) => s.action === "search")!;
    const dispatched = await cp.dispatchPlanStep(searchStep, async (name, params) => {
      const r = await agent.execution.toolHub.call(name, params);
      expect(r.success).toBe(true);
      return r.output;
    });
    expect(dispatched?.command.target).toBe("search");

    const createPlan = planner.createPlan(
      new GoalParser().parse('请创建 "alpha.ts" 和 "beta.ts"'),
    );
    expect(createPlan.getReadySteps().length).toBeGreaterThanOrEqual(2);
    for (const step of createPlan.getReadySteps()) {
      const r = await agent.execution.toolHub.call(step.action, step.params ?? {});
      expect(r.success).toBe(true);
    }
    expect(readFileSync(join(dir, "alpha.ts"), "utf8")).toContain("alpha.ts");
    expect(readFileSync(join(dir, "beta.ts"), "utf8")).toContain("beta.ts");
  });

  it("CASE P1-5: 链式拒绝 + http_request + 写→跑→错→改→再跑", async () => {
    const dir = makeWs();
    const agent = assembleAgent({
      llmProvider: "mock",
      workspaceDir: dir,
      enableCommands: true,
      allowedCommands: ["node", "npm"],
    });

    const chained = await agent.execution.toolHub.call("execute_command", {
      command: "npm install && curl http://evil",
    });
    expect(chained.success).toBe(false);
    expect(chained.error).toMatch(/链式|元字符/);

    const http = await agent.execution.toolHub.call("http_request", {
      url: "https://example.com",
      method: "GET",
    });
    expect(http.success).toBe(true);
    expect((http.output as { status: number }).status).toBe(200);

    // 写一段会失败的脚本 → 跑 → 改 → 再跑
    await agent.execution.toolHub.call("write_file", {
      path: "fail.js",
      content: "process.exit(3)\n",
    });
    const failRun = await agent.execution.toolHub.call("execute_command", {
      command: "node fail.js",
    });
    expect(failRun.success).toBe(false);
    expect((failRun.output as { exitCode: number }).exitCode).toBe(3);

    await agent.execution.toolHub.call("write_file", {
      path: "fail.js",
      content: "process.stdout.write('ok')\n",
    });
    const okRun = await agent.execution.toolHub.call("execute_command", {
      command: "node fail.js",
    });
    expect(okRun.success).toBe(true);
    expect((okRun.output as { stdout: string }).stdout).toContain("ok");
  });

  it("CASE P1-6: 单 run 内连续 25 次 write 不被配额拦截", async () => {
    const dir = makeWs();
    const agent = assembleAgent({ llmProvider: "mock", workspaceDir: dir });
    await agent.run("warmup"); // beginRun
    for (let i = 0; i < 25; i++) {
      const r = await agent.execution.toolHub.call("write_file", {
        path: `n${i}.txt`,
        content: `x${i}`,
      });
      expect(r.success).toBe(true);
    }
    expect(agent.governance.getStats().quota.toolCallsThisRun).toBeGreaterThanOrEqual(25);
  });

  it("CASE P1-8: 生图意图短路落盘（mock 无生图 → SVG fallback）", async () => {
    const dir = makeWs();
    const agent = assembleAgent({ llmProvider: "mock", workspaceDir: dir });
    const r = await agent.run("请生成一张以「大厂病」为主题的评论文章封面图", {
      contextTags: ["task:case-img"],
    });
    expect(r.status).toBe("completed");
    expect(r.iterations).toBe(0);
    expect(r.finalAnswer).toMatch(/cover\.svg|写入/);
    expect(r.metrics.imageGenerationFallback ?? r.metrics.imageGenerated).toBeTruthy();
    const svg = readFileSync(join(dir, "artifacts", "case-img", "cover.svg"), "utf8");
    expect(svg).toContain("<svg");
  });

  it("CASE P1-9: OOXML MIME + html/office preview kind", () => {
    expect(previewKindForPath("a/report.docx")).toBe("office");
    expect(previewKindForPath("index.html")).toBe("html");
    expect(mimeForPath("a.docx")).not.toBe("application/octet-stream");
    expect(mimeForPath("a.docx")).toContain("wordprocessingml");
    expect(mimeForPath("a.html")).toBe("text/html");
  });

  it("CASE P1-10: tcamp 类审查无搜索 → 短拒，不交付长假报告", async () => {
    const longFake =
      "# tcamp.qq.com 审查报告\n\n".repeat(50) +
      "查证状态：联网检索工具未配置……【待核验】\n通用基线……";
    const blocked = await enforceProductBehaviorAnswer({
      query: "请审查评测 https://tcamp.qq.com 线上页面并给出完整报告",
      draft: longFake,
      searchEvidence: "unavailable",
    });
    expect(blocked.answer).toContain("无法对线上页面做可核验审查");
    expect(blocked.answer.length).toBeLessThan(longFake.length / 2);
    expect(blocked.guards.some((g) => g.startsWith("site-audit-no-search"))).toBe(true);

    const allowed = await enforceProductBehaviorAnswer({
      query: "请审查评测 https://tcamp.qq.com 线上页面并给出完整报告",
      draft: longFake,
      searchEvidence: "reliable",
    });
    expect(allowed.answer).toBe(longFake.trim());
  });
});
