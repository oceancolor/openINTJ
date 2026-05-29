import { describe, expect, it } from "vitest";
import {
  DEFAULT_AGENT_SYSTEM_PROMPT,
  appendSourcesFooter,
  collectSearchSources,
  formatSourcesFooter,
} from "../src/agent-prompt.js";

const obs = (toolName: string, success: boolean, output: unknown) => ({
  state: { type: "observation", toolResult: { toolName, success, output } },
});

describe("agent-prompt", () => {
  it("DEFAULT_AGENT_SYSTEM_PROMPT 引导使用 search 工具", () => {
    expect(DEFAULT_AGENT_SYSTEM_PROMPT).toContain("search");
    expect(DEFAULT_AGENT_SYSTEM_PROMPT).toContain("联网");
  });

  it("collectSearchSources 抽取 search 命中来源并按 url 去重", () => {
    const traj = [
      { state: { type: "thought", content: "x" } },
      obs("search", true, {
        sources: [
          { title: "A", url: "https://a" },
          { title: "A-dup", url: "https://a" },
          { title: "B", url: "https://b" },
        ],
      }),
    ];
    const s = collectSearchSources(traj);
    expect(s).toHaveLength(2);
    expect(s[0]?.url).toBe("https://a");
    expect(s[1]?.url).toBe("https://b");
  });

  it("collectSearchSources 忽略非 search / 失败 / 无来源", () => {
    expect(collectSearchSources([obs("read_file", true, { sources: [{ url: "x" }] })])).toEqual([]);
    expect(collectSearchSources([obs("search", false, { sources: [{ url: "x" }] })])).toEqual([]);
    expect(collectSearchSources([obs("search", true, { sources: [] })])).toEqual([]);
    expect(collectSearchSources([])).toEqual([]);
  });

  it("formatSourcesFooter 生成参考来源脚注（截断到 max）", () => {
    const footer = formatSourcesFooter(
      [
        { title: "T1", url: "https://1" },
        { title: "T2", url: "https://2" },
        { title: "T3", url: "https://3" },
      ],
      2,
    );
    expect(footer).toContain("参考来源");
    expect(footer).toContain("1. T1 — https://1");
    expect(footer).toContain("2. T2 — https://2");
    expect(footer).not.toContain("T3");
  });

  it("appendSourcesFooter 在有来源时追加，无来源时原样返回", () => {
    const traj = [obs("search", true, { sources: [{ title: "X", url: "https://x" }] })];
    const withFooter = appendSourcesFooter("答案", traj);
    expect(withFooter).toContain("答案");
    expect(withFooter).toContain("参考来源");
    expect(withFooter).toContain("https://x");

    expect(appendSourcesFooter("答案", [])).toBe("答案");
  });

  it("appendSourcesFooter 不重复追加已有参考来源", () => {
    const traj = [obs("search", true, { sources: [{ url: "https://x" }] })];
    const already = "答案\n\n参考来源：\n1. 旧";
    expect(appendSourcesFooter(already, traj)).toBe(already);
  });
});
