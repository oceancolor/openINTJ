/**
 * P1-8：capability × host 装配矩阵（显式清单，供文档与测试共用）。
 *
 * 取值：
 * - available：默认装配且可调用
 * - opt_in：需环境变量/配置开启
 * - absent：该端未装配
 * - partial：有部分能力（如 CLI 可走生图短路但无 GUI 预览）
 */

export type HostId = "cli" | "server" | "desktop";

export type CapabilityStatus = "available" | "opt_in" | "absent" | "partial";

export type CapabilityId =
  | "office_tools"
  | "dev_tools"
  | "image_generation"
  | "shadow_memory"
  | "mcp_connectors"
  | "web_search"
  | "workspace_preview";

export const CAPABILITY_HOST_MATRIX: Record<
  CapabilityId,
  Record<HostId, CapabilityStatus>
> = {
  office_tools: { cli: "available", server: "available", desktop: "available" },
  dev_tools: { cli: "available", server: "available", desktop: "available" },
  image_generation: { cli: "partial", server: "partial", desktop: "available" },
  shadow_memory: { cli: "absent", server: "opt_in", desktop: "opt_in" },
  mcp_connectors: { cli: "opt_in", server: "opt_in", desktop: "opt_in" },
  web_search: { cli: "opt_in", server: "opt_in", desktop: "opt_in" },
  workspace_preview: { cli: "absent", server: "absent", desktop: "available" },
};

export const capabilityStatus = (cap: CapabilityId, host: HostId): CapabilityStatus =>
  CAPABILITY_HOST_MATRIX[cap][host];
