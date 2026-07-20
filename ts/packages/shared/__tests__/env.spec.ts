import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { loadOpenintjEnv, summarizeLlmEnv } from "../src/env.js";

const SAVED_KEYS = [
  "FOOBAR_LOCAL_ONLY",
  "FOOBAR_SHARED",
  "FOOBAR_OVERRIDE",
  "HUNYUAN_API_KEY",
  "LLM_PROVIDER",
  "OLLAMA_MODEL",
];

describe("@openintj/shared env loader", () => {
  let tmp: string;
  let snapshot: Record<string, string | undefined>;

  beforeEach(() => {
    tmp = mkdtempSync(path.join(os.tmpdir(), "openintj-env-"));
    // 伪造一个仓库根
    writeFileSync(path.join(tmp, "pnpm-workspace.yaml"), "packages:\n");
    snapshot = Object.fromEntries(SAVED_KEYS.map((k) => [k, process.env[k]]));
    for (const k of SAVED_KEYS) delete process.env[k];
  });

  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
    for (const [k, v] of Object.entries(snapshot)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  });

  it("loads .env by walking up from startDir", () => {
    writeFileSync(path.join(tmp, ".env"), "FOOBAR_SHARED=from_env\nHUNYUAN_API_KEY=test_key\n");
    mkdirSync(path.join(tmp, ".git"));
    const nested = path.join(tmp, "ts", "apps", "desktop");
    mkdirSync(nested, { recursive: true });

    const r = loadOpenintjEnv({ startDir: nested, log: "silent" });

    expect(r.loaded.length).toBe(1);
    expect(path.basename(r.loaded[0]!)).toBe(".env");
    expect(process.env["FOOBAR_SHARED"]).toBe("from_env");
    expect(process.env["HUNYUAN_API_KEY"]).toBe("test_key");
  });

  it(".env.local 优先于 .env，且都加载（同层）", () => {
    writeFileSync(path.join(tmp, ".env"), "FOOBAR_OVERRIDE=shared\nFOOBAR_SHARED=shared\n");
    writeFileSync(path.join(tmp, ".env.local"), "FOOBAR_OVERRIDE=local\nFOOBAR_LOCAL_ONLY=only\n");

    const r = loadOpenintjEnv({ stopAt: tmp, startDir: tmp, log: "silent" });

    expect(r.loaded.length).toBe(2);
    expect(path.basename(r.loaded[0]!)).toBe(".env.local");
    expect(process.env["FOOBAR_OVERRIDE"]).toBe("local");
    expect(process.env["FOOBAR_SHARED"]).toBe("shared");
    expect(process.env["FOOBAR_LOCAL_ONLY"]).toBe("only");
  });

  it("仓库混合布局：内层 pnpm-workspace + 外层 .git/.env，都能找到", () => {
    // 模拟 openintj 实际布局：F:\openINTJ\.env + F:\openINTJ\ts\pnpm-workspace.yaml
    mkdirSync(path.join(tmp, ".git"));
    writeFileSync(path.join(tmp, ".env"), "FOOBAR_SHARED=outer\n");
    const ts = path.join(tmp, "ts");
    mkdirSync(ts);
    writeFileSync(path.join(ts, "pnpm-workspace.yaml"), "packages:\n");

    const r = loadOpenintjEnv({ startDir: ts, log: "silent" });

    expect(r.loaded.length).toBe(1);
    expect(r.loaded[0]).toBe(path.join(tmp, ".env"));
    expect(process.env["FOOBAR_SHARED"]).toBe("outer");
  });

  it("已存在的 process.env 优先级最高（不被 .env 覆盖）", () => {
    process.env["FOOBAR_OVERRIDE"] = "from_shell";
    writeFileSync(path.join(tmp, ".env"), "FOOBAR_OVERRIDE=from_env\n");

    loadOpenintjEnv({ startDir: tmp, stopAt: tmp, log: "silent" });

    expect(process.env["FOOBAR_OVERRIDE"]).toBe("from_shell");
  });

  it("没有任何 .env 文件时安静返回", () => {
    mkdirSync(path.join(tmp, ".git"));
    const r = loadOpenintjEnv({ startDir: tmp, log: "silent" });
    expect(r.loaded).toEqual([]);
    expect(r.skipped).toEqual([]);
  });

  it("summarizeLlmEnv 在没有 key 时给 MISSING 信号", () => {
    const s = summarizeLlmEnv({ LLM_PROVIDER: "hunyuan" } as NodeJS.ProcessEnv);
    expect(s.provider).toBe("hunyuan");
    expect(s.hunyuan.hasKey).toBe(false);
    expect(s.summary).toContain("MISSING");
  });

  it("summarizeLlmEnv 不会泄漏 key 本体", () => {
    const s = summarizeLlmEnv({
      LLM_PROVIDER: "hunyuan",
      HUNYUAN_API_KEY: "sk-secret-deadbeef",
    } as NodeJS.ProcessEnv);
    expect(s.hunyuan.hasKey).toBe(true);
    expect(s.summary).not.toContain("sk-secret-deadbeef");
    expect(s.summary).toContain("set");
  });

  it("summarizeLlmEnv 反映联网搜索开关", () => {
    const off = summarizeLlmEnv({
      LLM_PROVIDER: "hunyuan",
      HUNYUAN_API_KEY: "k",
    } as NodeJS.ProcessEnv);
    expect(off.hunyuan.search).toBe(false);
    expect(off.summary).toContain("search=off");

    const on = summarizeLlmEnv({
      LLM_PROVIDER: "hunyuan",
      HUNYUAN_API_KEY: "k",
      HUNYUAN_ENABLE_SEARCH: "1",
    } as NodeJS.ProcessEnv);
    expect(on.hunyuan.search).toBe(true);
    expect(on.summary).toContain("search=on");
  });

  it("summarizeLlmEnv ollama 分支", () => {
    const s = summarizeLlmEnv({
      LLM_PROVIDER: "ollama",
      OLLAMA_MODEL: "qwen2:14b",
      OLLAMA_EMBED_MODEL: "nomic-custom",
    } as NodeJS.ProcessEnv);
    expect(s.provider).toBe("ollama");
    expect(s.summary).toContain("qwen2:14b");
    expect(s.ollama.embedModel).toBe("nomic-custom");
    expect(s.summary).toContain("embedModel=nomic-custom");
  });

  it.each([
    ["kimi", "KIMI_API_KEY", "kimi-k2.5"],
    ["minimax", "MINIMAX_API_KEY", "MiniMax-M2.1"],
    ["glm", "GLM_API_KEY", "glm-4.7"],
  ] as const)("summarizeLlmEnv supports %s without leaking its key", (provider, key, model) => {
    const s = summarizeLlmEnv({
      LLM_PROVIDER: provider,
      [key]: "provider-secret",
    } as NodeJS.ProcessEnv);
    expect(s.summary).toContain(`${provider}ApiKey=set`);
    expect(s.summary).toContain(`model=${model}`);
    expect(s.summary).not.toContain("provider-secret");
  });

  it("summarizeLlmEnv mock 分支不暴露任何 url", () => {
    const s = summarizeLlmEnv({ LLM_PROVIDER: "mock" } as NodeJS.ProcessEnv);
    expect(s.summary).toContain("mock");
    expect(s.summary).not.toContain("http");
  });
});
