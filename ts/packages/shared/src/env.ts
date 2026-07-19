/**
 * 跨入口共用的 `.env` 加载器。
 *
 * - 走 Node 21.7+ / 22+ 原生 `process.loadEnvFile`，不引入 dotenv 依赖。
 * - 文件名顺序：`.env.local`（本地覆盖，git ignored）→ `.env`（仓库共享模板）。
 *   后加载的不会覆盖已存在的 `process.env`，所以越靠前的优先级越高。
 * - 目录遍历策略：从入口的 `startDir` **逐级向上**找，每层都尝试 `.env.local` / `.env`。
 *   先命中的（更靠近 startDir 的）优先级最高。
 *   本仓库的实际布局是：`F:\openINTJ\.env.local`（外层 git 根）+
 *   `F:\openINTJ\ts\pnpm-workspace.yaml`（内层 pnpm workspace 根），所以单点查找会漏。
 * - 默认到包含 `.git` 的目录就停（无 `.git` 时走到文件系统根）。
 */

import { statSync } from "node:fs";
import path from "node:path";

export interface LoadEnvOptions {
  /** 显式指定起点目录（向上遍历）。缺省 `process.cwd()`。 */
  startDir?: string;
  /** 显式停止目录（找到此目录后再扫一层就停）。缺省自动检测 `.git`。 */
  stopAt?: string;
  /** 额外的候选文件名（按顺序尝试）。缺省 `[".env.local", ".env"]`。 */
  filenames?: readonly string[];
  /** 加载后是否回写一行 stderr 日志（'silent' / 'short' / 'verbose'）。缺省 `short`。 */
  log?: "silent" | "short" | "verbose";
  /** 注入 stderr 的前缀（cli/server/desktop 各自标注）。缺省 `[env]`。 */
  logPrefix?: string;
  /** 最多向上扫几层（安全护栏）。缺省 10。 */
  maxDepth?: number;
}

export interface LoadEnvResult {
  /** 实际加载的文件（绝对路径）。先加载的在前。 */
  loaded: string[];
  /** 找到但当时 Node 不支持 loadEnvFile 跳过的文件。 */
  skipped: string[];
  /** 实际扫到的目录序列（startDir 在第 0 位）。 */
  scannedDirs: string[];
}

const DEFAULT_FILENAMES = [".env.local", ".env"] as const;
const DEFAULT_MAX_DEPTH = 10;

const isFile = (p: string): boolean => {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
};

const isDir = (p: string): boolean => {
  try {
    return statSync(p).isDirectory();
  } catch {
    return false;
  }
};

export const loadOpenintjEnv = (opts: LoadEnvOptions = {}): LoadEnvResult => {
  const startDir = path.resolve(opts.startDir ?? process.cwd());
  const filenames = opts.filenames ?? DEFAULT_FILENAMES;
  const maxDepth = opts.maxDepth ?? DEFAULT_MAX_DEPTH;
  const result: LoadEnvResult = { loaded: [], skipped: [], scannedDirs: [] };

  // Node 21.7+ / 22+：process.loadEnvFile 同步把 KEY=VALUE 注入 process.env，但不覆盖已存在键。
  const loader = (process as unknown as { loadEnvFile?: (p?: string) => void }).loadEnvFile;

  const fsRoot = path.parse(startDir).root;
  let dir = startDir;
  let depth = 0;
  let stopAfterCurrent = false;

  while (true) {
    result.scannedDirs.push(dir);
    for (const name of filenames) {
      const p = path.join(dir, name);
      if (!isFile(p)) continue;
      if (typeof loader !== "function") {
        result.skipped.push(p);
        continue;
      }
      try {
        loader.call(process, p);
        result.loaded.push(p);
      } catch {
        // 文件格式错误 / 权限问题；不要让 .env 解析挂掉整个 app
        result.skipped.push(p);
      }
    }

    if (stopAfterCurrent) break;

    // 用户显式 stopAt 后再扫一层即可
    if (opts.stopAt && path.resolve(opts.stopAt) === dir) {
      stopAfterCurrent = true;
    } else if (isDir(path.join(dir, ".git"))) {
      // git 根：该层也是常见 .env 位置；扫完就停
      stopAfterCurrent = true;
    }

    depth += 1;
    if (depth >= maxDepth) break;
    if (dir === fsRoot) break;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  const log = opts.log ?? "short";
  if (log !== "silent" && (result.loaded.length > 0 || result.skipped.length > 0)) {
    const prefix = opts.logPrefix ?? "[env]";
    if (log === "verbose") {
      for (const p of result.loaded) process.stderr.write(`${prefix} loaded ${p}\n`);
      for (const p of result.skipped) process.stderr.write(`${prefix} skipped ${p}\n`);
    } else {
      const parts: string[] = [];
      if (result.loaded.length > 0) {
        parts.push(`loaded ${result.loaded.map((p) => path.basename(p)).join(", ")}`);
      }
      if (result.skipped.length > 0) {
        parts.push(`skipped ${result.skipped.length}`);
      }
      process.stderr.write(`${prefix} ${parts.join(" · ")}\n`);
    }
  }
  return result;
};

/**
 * 把当前 LLM 相关 env 浓缩成单行摘要（不打印 API key 本身）。
 * 用于启动日志，方便确认到底读没读到 key。
 */
export const summarizeLlmEnv = (
  env: NodeJS.ProcessEnv = process.env,
): {
  provider: string;
  hunyuan: { hasKey: boolean; baseUrl: string; model: string; search: boolean };
  ollama: { baseUrl: string; model: string };
  embedProvider: string;
  summary: string;
} => {
  const provider = env["LLM_PROVIDER"] ?? "auto";
  const embedProvider = env["EMBEDDING_PROVIDER"] ?? env["EMBED_PROVIDER"] ?? "auto";
  const isTruthy = (raw: string | undefined): boolean => {
    if (!raw) return false;
    const v = raw.trim().toLowerCase();
    return v === "1" || v === "true" || v === "yes" || v === "on";
  };
  const hunyuan = {
    hasKey: Boolean(env["HUNYUAN_API_KEY"]?.trim()),
    baseUrl: env["HUNYUAN_BASE_URL"] ?? "https://api.hunyuan.cloud.tencent.com/v1",
    model: env["HUNYUAN_MODEL"] ?? "hunyuan-turbos-latest",
    search: isTruthy(env["HUNYUAN_ENABLE_SEARCH"]) || isTruthy(env["HUNYUAN_FORCE_SEARCH"]),
  };
  const ollama = {
    baseUrl: env["OLLAMA_BASE_URL"] ?? "http://127.0.0.1:11434",
    model: env["OLLAMA_MODEL"] ?? "qwen2.5:7b",
  };
  const tail =
    provider === "hunyuan"
      ? `hunyuanApiKey=${hunyuan.hasKey ? "set" : "MISSING"} model=${hunyuan.model} search=${hunyuan.search ? "on" : "off"}`
      : provider === "ollama"
        ? `ollamaModel=${ollama.model} baseUrl=${ollama.baseUrl}`
        : provider === "auto"
          ? `auto(local-first) ollama=${ollama.baseUrl} hunyuanKey=${hunyuan.hasKey ? "set" : "MISSING"}`
          : "(explicit mock)";
  return {
    provider,
    hunyuan,
    ollama,
    embedProvider,
    summary: `provider=${provider} embed=${embedProvider} ${tail}`,
  };
};
