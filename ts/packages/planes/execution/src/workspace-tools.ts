import { exec } from "node:child_process";
import { promises as fs } from "node:fs";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import { promisify } from "node:util";
import type { ToolHandler } from "@openintj/core";

const execAsync = promisify(exec);

/**
 * 工作区工具集配置。
 *
 * 设计目标（对齐 RFC-004 §8 "系统能力的治理边界"）：
 *  - **路径沙箱**：readFile/writeFile 一律解析到 `root` 内；任何 `..` / 绝对路径越界一律拒绝。
 *  - **大小上限**：防止把超大文件灌进 prompt 或写爆磁盘。
 *  - **命令默认禁用**：executeCommand 是高危能力，必须显式 `enableCommands` + 白名单可执行文件名。
 */
export interface WorkspaceToolsOpts {
  /** 工作区根目录；所有 fs 操作被限制在此目录内。 */
  root: string;
  /** 单文件读取最大字节（默认 1 MiB）。 */
  maxReadBytes?: number;
  /** 单文件写入最大字节（默认 1 MiB）。 */
  maxWriteBytes?: number;
  /**
   * 是否允许 executeCommand（默认 false）。
   * 命令执行高危，本地优先也要显式开启，避免被诱导跑任意命令。
   */
  enableCommands?: boolean;
  /**
   * 命令白名单（按可执行文件名匹配第一个 token；仅 enableCommands=true 时生效）。
   * 空数组 = 拒绝全部（即使 enableCommands=true）。
   */
  allowedCommands?: string[];
  /** 命令执行超时（毫秒，默认 30s）。 */
  commandTimeoutMs?: number;
  /** 命令 stdout/stderr 截断字符数（默认 10000）。 */
  commandOutputMaxChars?: number;
}

export interface WorkspaceTools {
  readFile: ToolHandler;
  writeFile: ToolHandler;
  executeCommand: ToolHandler;
}

const DEFAULT_MAX_BYTES = 1024 * 1024;
const DEFAULT_CMD_TIMEOUT = 30_000;
const DEFAULT_CMD_OUTPUT_MAX = 10_000;

/**
 * 把相对路径解析到 root 内，越界则抛错。
 * 同时拒绝绝对路径输入（必须相对工作区）。
 */
export const resolveInRoot = (root: string, rel: string): string => {
  const normalizedRoot = resolve(root);
  if (isAbsolute(rel)) {
    throw new Error(`路径越界：不接受绝对路径 '${rel}'，请使用相对工作区根目录的路径`);
  }
  const abs = resolve(normalizedRoot, rel);
  const relToRoot = relative(normalizedRoot, abs);
  if (relToRoot.startsWith("..") || isAbsolute(relToRoot)) {
    throw new Error(`路径越界：'${rel}' 超出工作区根目录`);
  }
  return abs;
};

const truncate = (s: string, max: number): string =>
  s.length > max ? `${s.slice(0, max)}\n…[已截断 ${s.length - max} 字符]` : s;

const firstToken = (command: string): string => command.trim().split(/\s+/)[0] ?? "";

/**
 * 构造一组被沙箱约束的工作区工具，供 `ToolHub.registerBuiltinTools` 使用。
 */
export const createWorkspaceTools = (opts: WorkspaceToolsOpts): WorkspaceTools => {
  const root = resolve(opts.root);
  const maxReadBytes = opts.maxReadBytes ?? DEFAULT_MAX_BYTES;
  const maxWriteBytes = opts.maxWriteBytes ?? DEFAULT_MAX_BYTES;
  const enableCommands = opts.enableCommands ?? false;
  const allowedCommands = opts.allowedCommands ?? [];
  const commandTimeoutMs = opts.commandTimeoutMs ?? DEFAULT_CMD_TIMEOUT;
  const commandOutputMax = opts.commandOutputMaxChars ?? DEFAULT_CMD_OUTPUT_MAX;

  const readFile: ToolHandler = async (params) => {
    const path = typeof params["path"] === "string" ? params["path"] : "";
    if (!path) throw new Error("read_file 需要非空 path 参数");
    const abs = resolveInRoot(root, path);
    const stat = await fs.stat(abs);
    if (!stat.isFile()) throw new Error(`不是文件：${path}`);
    if (stat.size > maxReadBytes) {
      throw new Error(`文件过大：${stat.size} 字节 > 上限 ${maxReadBytes}`);
    }
    const content = await fs.readFile(abs, "utf8");
    return { path, content, bytes: stat.size };
  };

  const writeFile: ToolHandler = async (params) => {
    const path = typeof params["path"] === "string" ? params["path"] : "";
    const content = typeof params["content"] === "string" ? params["content"] : "";
    if (!path) throw new Error("write_file 需要非空 path 参数");
    const bytes = Buffer.byteLength(content, "utf8");
    if (bytes > maxWriteBytes) {
      throw new Error(`写入内容过大：${bytes} 字节 > 上限 ${maxWriteBytes}`);
    }
    const abs = resolveInRoot(root, path);
    await fs.mkdir(dirname(abs), { recursive: true });
    await fs.writeFile(abs, content, "utf8");
    return { path, bytesWritten: bytes };
  };

  const executeCommand: ToolHandler = async (params) => {
    if (!enableCommands) {
      throw new Error("命令执行未启用（enableCommands=false）");
    }
    const command = typeof params["command"] === "string" ? params["command"] : "";
    if (!command) throw new Error("execute_command 需要非空 command 参数");
    const exe = firstToken(command);
    if (allowedCommands.length === 0 || !allowedCommands.includes(exe)) {
      throw new Error(`命令 '${exe}' 不在白名单内（allowedCommands=[${allowedCommands.join(", ")}]）`);
    }
    try {
      const { stdout, stderr } = await execAsync(command, {
        cwd: root,
        timeout: commandTimeoutMs,
        maxBuffer: 4 * 1024 * 1024,
        windowsHide: true,
      });
      return {
        command,
        stdout: truncate(stdout ?? "", commandOutputMax),
        stderr: truncate(stderr ?? "", commandOutputMax),
        exitCode: 0,
      };
    } catch (e) {
      const err = e as { stdout?: string; stderr?: string; code?: number; message?: string };
      // 命令非 0 退出在 ReAct 里也是有用的"观察"，整理成结构化结果而非直接抛异常吞掉输出。
      return {
        command,
        stdout: truncate(err.stdout ?? "", commandOutputMax),
        stderr: truncate(err.stderr ?? err.message ?? "", commandOutputMax),
        exitCode: typeof err.code === "number" ? err.code : 1,
      };
    }
  };

  return { readFile, writeFile, executeCommand };
};
