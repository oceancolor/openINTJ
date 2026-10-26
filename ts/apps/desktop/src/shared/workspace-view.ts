/** Shared workspace-view helpers for the desktop workbench and a future IDE shell. */

export type WorkspacePreviewKind = "image" | "text" | "file";

export interface WorkspaceArtifact {
  path: string;
  kind: WorkspacePreviewKind;
}

export interface WorkspaceEntry {
  path: string;
  name: string;
  type: "file" | "dir";
  bytes?: number;
}

export interface WorkspaceTreeNode {
  path: string;
  name: string;
  type: "file" | "dir";
  bytes?: number;
  children?: WorkspaceTreeNode[];
}

const WORKSPACE_PATH_RE =
  /`((?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12})`/g;

const RASTER_EXT = new Set(["png", "jpg", "jpeg", "webp", "gif"]);
const IMAGE_EXT = new Set([...RASTER_EXT, "svg"]);
const TEXT_EXT = new Set([
  "md",
  "txt",
  "json",
  "ts",
  "tsx",
  "js",
  "jsx",
  "css",
  "html",
  "yml",
  "yaml",
  "xml",
  "csv",
  "svg",
  "py",
  "rs",
  "go",
  "toml",
  "sh",
  "bash",
  "ini",
]);

const MONACO_LANGUAGE: Record<string, string> = {
  ts: "typescript",
  tsx: "typescript",
  js: "javascript",
  jsx: "javascript",
  json: "json",
  css: "css",
  html: "html",
  md: "markdown",
  xml: "xml",
  svg: "xml",
};

export const extractWorkspaceArtifactPaths = (text: string): string[] => {
  const found: string[] = [];
  for (const match of text.matchAll(WORKSPACE_PATH_RE)) {
    const path = match[1];
    if (path && !found.includes(path)) found.push(path);
  }
  return found;
};

export const splitWorkspacePathMarkup = (
  text: string,
): Array<{ type: "text" | "path"; value: string }> => {
  const parts: Array<{ type: "text" | "path"; value: string }> = [];
  let last = 0;
  for (const match of text.matchAll(WORKSPACE_PATH_RE)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ type: "text", value: text.slice(last, index) });
    parts.push({ type: "path", value: match[1] ?? "" });
    last = index + match[0].length;
  }
  if (last < text.length) parts.push({ type: "text", value: text.slice(last) });
  return parts.length > 0 ? parts : [{ type: "text", value: text }];
};

export const extensionOf = (path: string): string =>
  path.split(".").pop()?.toLowerCase() ?? "";

export const previewKindForPath = (path: string): WorkspacePreviewKind => {
  const ext = extensionOf(path);
  if (IMAGE_EXT.has(ext)) return "image";
  if (TEXT_EXT.has(ext)) return "text";
  return "file";
};

/** Raster stays a preview; SVG and other text open in the editor. */
export const editorKindForPath = (path: string): WorkspacePreviewKind => {
  const ext = extensionOf(path);
  if (RASTER_EXT.has(ext)) return "image";
  if (TEXT_EXT.has(ext)) return "text";
  return "file";
};

export const isEditablePath = (path: string): boolean => editorKindForPath(path) === "text";

export const monacoLanguageForPath = (path: string): string =>
  MONACO_LANGUAGE[extensionOf(path)] ?? "plaintext";

export const mimeForPath = (path: string): string => {
  const ext = extensionOf(path);
  if (ext === "png") return "image/png";
  if (ext === "jpg" || ext === "jpeg") return "image/jpeg";
  if (ext === "webp") return "image/webp";
  if (ext === "gif") return "image/gif";
  if (ext === "svg") return "image/svg+xml";
  return "application/octet-stream";
};

export const artifactsFromAnswer = (text: string): WorkspaceArtifact[] =>
  extractWorkspaceArtifactPaths(text).map((path) => ({
    path,
    kind: previewKindForPath(path),
  }));

/** 「预览刚才的图」不应再走生图模型。 */
export const requestsArtifactPreview = (query: string): boolean =>
  /(?:预览|展示|看看|显示一下|打开预览)\s*.{0,32}(?:图|插图|封面|产物|文件|workspace|artifact)/iu.test(
    query,
  );

export const preferredPreviewPath = (paths: readonly string[]): string | undefined => {
  const rasters = paths.filter((path) => RASTER_EXT.has(extensionOf(path)));
  const svg = paths.find((path) => /\.svg$/i.test(path));
  if (svg && rasters.length > 0) {
    const dir = svg.includes("/") ? svg.slice(0, svg.lastIndexOf("/") + 1) : "";
    const sibling = rasters.find((path) => (dir ? path.startsWith(dir) : true));
    if (sibling) return sibling;
  }
  if (rasters[0]) return rasters[0];
  const notStub = paths.find((path) => !/_preview\.svg$/i.test(path));
  return notStub ?? paths[0];
};

export const guessedRasterFromSvgPath = (svgPath: string): string | undefined => {
  if (!/\.svg$/i.test(svgPath)) return undefined;
  if (/_preview\.svg$/i.test(svgPath)) return svgPath.replace(/_preview\.svg$/i, ".png");
  return svgPath.replace(/\.svg$/i, ".png");
};

export const rasterSiblingFromSvg = (svg: string, svgPath: string): string | undefined => {
  const href = svg.match(/(?:xlink:href|href)\s*=\s*["']([^"']+)["']/i)?.[1];
  if (!href || /^(?:data:|https?:|file:)/i.test(href)) return undefined;
  const file = href.replace(/^\.\//, "").split(/[?#]/)[0] ?? "";
  if (!RASTER_EXT.has(extensionOf(file))) return undefined;
  if (file.includes("/") && !file.startsWith("../")) return file.replaceAll("\\", "/");
  const dir = svgPath.includes("/") ? svgPath.slice(0, svgPath.lastIndexOf("/") + 1) : "";
  return `${dir}${file}`.replaceAll("\\", "/");
};

export const buildWorkspaceTree = (entries: readonly WorkspaceEntry[]): WorkspaceTreeNode[] => {
  const dirs = new Map<string, WorkspaceTreeNode>();
  const roots: WorkspaceTreeNode[] = [];
  const ensureDir = (path: string): WorkspaceTreeNode => {
    const existing = dirs.get(path);
    if (existing) return existing;
    const node: WorkspaceTreeNode = {
      path,
      name: path.split("/").pop() ?? path,
      type: "dir",
      children: [],
    };
    dirs.set(path, node);
    const slash = path.lastIndexOf("/");
    if (slash < 0) roots.push(node);
    else ensureDir(path.slice(0, slash)).children?.push(node);
    return node;
  };
  for (const entry of entries) {
    if (entry.type === "dir") {
      ensureDir(entry.path);
      continue;
    }
    const node: WorkspaceTreeNode = {
      path: entry.path,
      name: entry.name,
      type: "file",
      ...(entry.bytes !== undefined ? { bytes: entry.bytes } : {}),
    };
    const slash = entry.path.lastIndexOf("/");
    if (slash < 0) roots.push(node);
    else ensureDir(entry.path.slice(0, slash)).children?.push(node);
  }
  return roots;
};
