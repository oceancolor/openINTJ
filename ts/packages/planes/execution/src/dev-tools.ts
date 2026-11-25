import { spawn } from "node:child_process";
import { resolve } from "node:path";
import type { ToolHandler } from "@openintj/core";
import { resolveInRoot } from "./workspace-tools.js";

export interface DevToolsOpts {
  root: string;
  /** 子进程超时（毫秒，默认 120s）。 */
  timeoutMs?: number;
  /** stdout/stderr 截断（默认 12_000）。 */
  outputMaxChars?: number;
  /** http_request 响应体截断（默认 50_000）。 */
  httpBodyMaxChars?: number;
}

export interface DevTools {
  runTests: ToolHandler;
  typecheck: ToolHandler;
  httpRequest: ToolHandler;
  startServer: ToolHandler;
}

const DEFAULT_TIMEOUT = 120_000;
const DEFAULT_OUTPUT_MAX = 12_000;
const DEFAULT_HTTP_MAX = 50_000;

const truncate = (s: string, max: number): string =>
  s.length > max ? `${s.slice(0, max)}\n…[已截断 ${s.length - max} 字符]` : s;

const runArgv = (
  cwd: string,
  argv: string[],
  opts: { timeoutMs: number; outputMaxChars: number },
): Promise<{
  command: string;
  stdout: string;
  stderr: string;
  exitCode: number;
  ok: boolean;
}> =>
  new Promise((resolvePromise) => {
    const [bin, ...args] = argv;
    if (!bin) {
      resolvePromise({
        command: "",
        stdout: "",
        stderr: "空命令",
        exitCode: 1,
        ok: false,
      });
      return;
    }
    const child = spawn(bin, args, {
      cwd,
      windowsHide: true,
      shell: false,
      env: process.env,
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill("SIGTERM");
      resolvePromise({
        command: argv.join(" "),
        stdout: truncate(stdout, opts.outputMaxChars),
        stderr: truncate(`${stderr}\n[timeout ${opts.timeoutMs}ms]`, opts.outputMaxChars),
        exitCode: 124,
        ok: false,
      });
    }, opts.timeoutMs);

    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (err) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolvePromise({
        command: argv.join(" "),
        stdout: truncate(stdout, opts.outputMaxChars),
        stderr: truncate(err.message, opts.outputMaxChars),
        exitCode: 1,
        ok: false,
      });
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      const exitCode = typeof code === "number" ? code : 1;
      resolvePromise({
        command: argv.join(" "),
        stdout: truncate(stdout, opts.outputMaxChars),
        stderr: truncate(stderr, opts.outputMaxChars),
        exitCode,
        ok: exitCode === 0,
      });
    });
  });

/**
 * P1-5：写→跑→错→改闭环辅助工具（无 shell 链式；argv spawn）。
 */
export const createDevTools = (opts: DevToolsOpts): DevTools => {
  const root = resolve(opts.root);
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT;
  const outputMaxChars = opts.outputMaxChars ?? DEFAULT_OUTPUT_MAX;
  const httpBodyMaxChars = opts.httpBodyMaxChars ?? DEFAULT_HTTP_MAX;
  const spawnOpts = { timeoutMs, outputMaxChars };

  const runTests: ToolHandler = async (params) => {
    const filter = typeof params["filter"] === "string" ? params["filter"] : "";
    const argv = filter
      ? ["npx", "vitest", "run", filter]
      : ["npx", "vitest", "run"];
    return runArgv(root, argv, spawnOpts);
  };

  const typecheck: ToolHandler = async (params) => {
    const project =
      typeof params["project"] === "string" && params["project"]
        ? params["project"]
        : ".";
    // 相对工作区；越界拒绝
    resolveInRoot(root, project === "." ? "package.json" : project);
    return runArgv(root, ["npx", "tsc", "--noEmit", "-p", project], spawnOpts);
  };

  const httpRequest: ToolHandler = async (params) => {
    const url = typeof params["url"] === "string" ? params["url"] : "";
    if (!url) throw new Error("http_request 需要非空 url");
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      throw new Error(`http_request：非法 url '${url}'`);
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error("http_request 仅允许 http/https");
    }
    const method =
      typeof params["method"] === "string" && params["method"]
        ? params["method"].toUpperCase()
        : "GET";
    const body = typeof params["body"] === "string" ? params["body"] : undefined;
    const headers =
      params["headers"] !== null &&
      typeof params["headers"] === "object" &&
      !Array.isArray(params["headers"])
        ? (params["headers"] as Record<string, string>)
        : undefined;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), Math.min(timeoutMs, 60_000));
    try {
      const res = await fetch(url, {
        method,
        ...(headers ? { headers } : {}),
        ...(body !== undefined ? { body } : {}),
        signal: controller.signal,
      });
      const text = truncate(await res.text(), httpBodyMaxChars);
      return {
        ok: res.ok,
        status: res.status,
        url,
        method,
        body: text,
        exitCode: res.ok ? 0 : 1,
      };
    } finally {
      clearTimeout(timer);
    }
  };

  const startServer: ToolHandler = async (params) => {
    const command = typeof params["command"] === "string" ? params["command"] : "";
    if (!command) throw new Error("start_server 需要非空 command（argv 空格分隔，无 shell）");
    const argv = command.trim().split(/\s+/).filter(Boolean);
    const [bin, ...args] = argv;
    if (!bin) throw new Error("start_server：空命令");
    // 仅允许常见本地开发启动器，避免任意二进制常驻
    const allowed = new Set(["node", "npx", "npm", "pnpm", "yarn", "python", "python3"]);
    if (!allowed.has(bin)) {
      throw new Error(
        `start_server 拒绝 '${bin}'（允许：${[...allowed].join(", ")}）`,
      );
    }
    const child = spawn(bin, args, {
      cwd: root,
      windowsHide: true,
      shell: false,
      detached: true,
      stdio: "ignore",
      env: process.env,
    });
    child.unref();
    return {
      ok: true,
      pid: child.pid ?? null,
      command: argv.join(" "),
      note: "已后台启动；请用 http_request 探活，自行 stop。",
      exitCode: 0,
    };
  };

  return { runTests, typecheck, httpRequest, startServer };
};
