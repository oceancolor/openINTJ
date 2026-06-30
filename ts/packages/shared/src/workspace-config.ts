/**
 * 跨入口（cli / server / desktop）共用的工作区工具配置解析。
 *
 * 解析后交给 `@openintj/plane-execution` 的 `createWorkspaceTools`。
 * 这里只做纯粹的"opts > env > 默认值"解析，不 import 执行平面，避免 shared 反向依赖。
 */
export interface WorkspaceConfigInput {
  /** 工作区根目录；缺省走 env OPENINTJ_WORKSPACE_DIR，再退到 fallbackRoot。 */
  workspaceDir?: string;
  /** 是否允许执行命令；缺省走 env OPENINTJ_ENABLE_COMMANDS=1。命令执行高危，默认关。 */
  enableCommands?: boolean;
  /** 命令白名单；缺省走 env OPENINTJ_ALLOWED_COMMANDS（逗号分隔）。 */
  allowedCommands?: string[];
}

export interface ResolvedWorkspaceConfig {
  root: string;
  enableCommands: boolean;
  allowedCommands: string[];
}

const parseCommandList = (raw: string | undefined): string[] =>
  (raw ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

export const resolveWorkspaceConfig = (
  opts: WorkspaceConfigInput,
  fallbackRoot: string,
): ResolvedWorkspaceConfig => {
  const root = opts.workspaceDir ?? process.env["OPENINTJ_WORKSPACE_DIR"] ?? fallbackRoot;
  const enableCommands =
    opts.enableCommands ?? process.env["OPENINTJ_ENABLE_COMMANDS"] === "1";
  const allowedCommands =
    opts.allowedCommands ?? parseCommandList(process.env["OPENINTJ_ALLOWED_COMMANDS"]);
  return { root, enableCommands, allowedCommands };
};
