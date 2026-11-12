import { mkdir, readFile, readdir, rename, stat } from "node:fs/promises";
import { join } from "node:path";
import { resolveInRoot } from "@openintj/plane-execution";
import type { WorkspaceEntry } from "../shared/workspace-view.js";
import { mimeForPath } from "../shared/workspace-view.js";

const SKIP_DIR = new Set(["node_modules", ".git", "dist", "out", "release", ".cache", ".turbo"]);
const MAX_ENTRIES = 400;
const MAX_DEPTH = 6;
const MAX_PREVIEW_BYTES = 8 * 1024 * 1024;

export const listWorkspaceEntries = async (root: string): Promise<WorkspaceEntry[]> => {
  const entries: WorkspaceEntry[] = [];
  const walk = async (rel: string, depth: number): Promise<void> => {
    if (entries.length >= MAX_ENTRIES || depth > MAX_DEPTH) return;
    const abs = resolveInRoot(root, rel || ".");
    const names = await readdir(abs, { withFileTypes: true });
    names.sort((a, b) => a.name.localeCompare(b.name, "en"));
    for (const dirent of names) {
      if (entries.length >= MAX_ENTRIES) return;
      if (SKIP_DIR.has(dirent.name)) continue;
      const child = rel ? `${rel}/${dirent.name}` : dirent.name;
      const path = child.replaceAll("\\", "/");
      if (dirent.isDirectory()) {
        entries.push({ path, name: dirent.name, type: "dir" });
        await walk(path, depth + 1);
      } else if (dirent.isFile()) {
        const st = await stat(join(abs, dirent.name));
        entries.push({ path, name: dirent.name, type: "file", bytes: st.size });
      }
    }
  };
  await walk("", 0);
  return entries;
};

export const readWorkspaceBytes = async (
  root: string,
  rel: string,
): Promise<{ path: string; mimeType: string; base64: string; bytes: number }> => {
  const abs = resolveInRoot(root, rel);
  const st = await stat(abs);
  if (!st.isFile()) throw new Error(`不是文件：${rel}`);
  if (st.size > MAX_PREVIEW_BYTES) {
    throw new Error(`文件过大：${st.size} 字节 > 上限 ${MAX_PREVIEW_BYTES}`);
  }
  const buf = await readFile(abs);
  return {
    path: rel.replaceAll("\\", "/"),
    mimeType: mimeForPath(rel),
    base64: buf.toString("base64"),
    bytes: st.size,
  };
};

export const resolveWorkspaceItem = async (
  root: string,
  rel?: string,
): Promise<{ abs: string; kind: "file" | "dir" }> => {
  const abs = resolveInRoot(root, rel?.trim() || ".");
  const st = await stat(abs);
  return { abs, kind: st.isDirectory() ? "dir" : "file" };
};

/** 在工作区内建目录（已存在则 created=false，不报错）。 */
export const mkdirWorkspaceDir = async (
  root: string,
  rel: string,
): Promise<{ path: string; created: boolean }> => {
  const abs = resolveInRoot(root, rel);
  try {
    const st = await stat(abs);
    if (!st.isDirectory()) throw new Error(`同名文件已存在，无法建目录：${rel}`);
    return { path: rel.replaceAll("\\", "/"), created: false };
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }
  await mkdir(abs, { recursive: true });
  return { path: rel.replaceAll("\\", "/"), created: true };
};

/** 工作区内目录改名：源必须是已存在目录，目标不得已存在。 */
export const renameWorkspaceDir = async (
  root: string,
  from: string,
  to: string,
): Promise<{ from: string; to: string }> => {
  const fromAbs = resolveInRoot(root, from);
  const toAbs = resolveInRoot(root, to);
  const st = await stat(fromAbs);
  if (!st.isDirectory()) throw new Error(`不是目录：${from}`);
  try {
    await stat(toAbs);
    throw new Error(`目标目录已存在：${to}`);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }
  await rename(fromAbs, toAbs);
  return { from: from.replaceAll("\\", "/"), to: to.replaceAll("\\", "/") };
};
