import {
  type Dirent,
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import type { Connector, ConnectorTool } from "./types.js";

export interface NotesConnectorOpts {
  rootDir: string;
  id?: string;
  /** File extensions treated as notes (default .md / .txt). */
  extensions?: readonly string[];
  maxDepth?: number;
  maxFiles?: number;
  /** Per-read output truncation (default 20_000 chars). */
  maxReadChars?: number;
  /** RFC-017: expose create/append/delete write tools (default off). */
  writable?: boolean;
  /** Per-write content cap (default 200_000 chars). */
  maxWriteChars?: number;
}

interface NoteFile {
  path: string; // relative, posix-style
  mtime: number;
  size: number;
}

const baseDescriptor = {
  permissions: ["filesystem.read"],
  idempotent: true,
  errorSemantics: "fail_fast" as const,
  timeoutS: 30,
};

const writeDescriptor = {
  permissions: ["filesystem.write"],
  idempotent: false,
  errorSemantics: "fail_fast" as const,
  timeoutS: 30,
};

/** Read-only connector over a local notes directory (Markdown / text). */
export const createNotesConnector = (opts: NotesConnectorOpts): Connector => {
  const root = resolve(opts.rootDir);
  const extensions = new Set((opts.extensions ?? [".md", ".txt"]).map((ext) => ext.toLowerCase()));
  const maxDepth = opts.maxDepth ?? 6;
  const maxFiles = opts.maxFiles ?? 500;
  const maxReadChars = opts.maxReadChars ?? 20_000;
  const id = opts.id ?? "notes";

  const listFiles = (): NoteFile[] => {
    const out: NoteFile[] = [];
    const walk = (dir: string, depth: number): void => {
      if (depth > maxDepth || out.length >= maxFiles) return;
      let entries: Dirent[];
      try {
        entries = readdirSync(dir, { withFileTypes: true });
      } catch {
        return; // unreadable directory = no notes there
      }
      for (const entry of entries) {
        if (out.length >= maxFiles) return;
        if (entry.name.startsWith(".")) continue;
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full, depth + 1);
        } else if (entry.isFile()) {
          const dot = entry.name.lastIndexOf(".");
          if (dot < 0 || !extensions.has(entry.name.slice(dot).toLowerCase())) continue;
          try {
            const stat = statSync(full);
            out.push({
              path: relative(root, full).split(sep).join("/"),
              mtime: stat.mtimeMs,
              size: stat.size,
            });
          } catch {
            // vanished between readdir and stat; skip
          }
        }
      }
    };
    walk(root, 0);
    return out.sort((a, b) => b.mtime - a.mtime);
  };

  /** Resolve a user-supplied relative path, rejecting escapes outside root. */
  const resolveInside = (input: string): string => {
    const full = resolve(root, input);
    const rel = relative(root, full);
    if (rel === "" || rel === ".." || rel.startsWith(`..${sep}`)) {
      throw new Error(`path escapes notes root: ${input}`);
    }
    return full;
  };

  const readNote = (relPath: string): string => {
    const content = readFileSync(resolveInside(relPath), "utf8");
    return content.length > maxReadChars
      ? `${content.slice(0, maxReadChars)}\n…[truncated ${content.length - maxReadChars} chars]`
      : content;
  };

  // RFC-017 write side (only exposed when opts.writable).
  const maxWriteChars = opts.maxWriteChars ?? 200_000;
  const requireContent = (value: unknown): string => {
    if (typeof value !== "string") throw new Error("missing or invalid param: content");
    if (value.length > maxWriteChars) {
      throw new Error(`content exceeds maxWriteChars (${maxWriteChars})`);
    }
    return value;
  };
  const assertNoteExtension = (relPath: string): void => {
    const dot = relPath.lastIndexOf(".");
    if (dot < 0 || !extensions.has(relPath.slice(dot).toLowerCase())) {
      throw new Error(`unsupported note extension: ${relPath}`);
    }
  };

  const tools: ConnectorTool[] = [
    {
      descriptor: {
        name: "notes_list",
        description: "列出笔记目录中的笔记（按最近修改排序）：相对路径、大小、修改时间。",
        inputSchema: { limit: "number?" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const limit = clampLimit(params["limit"], 50, 200);
        return { root: opts.rootDir, notes: listFiles().slice(0, limit) };
      },
    },
    {
      descriptor: {
        name: "notes_read",
        description: "读取一篇笔记的正文（相对路径，如 docs/plan.md）。",
        inputSchema: { path: "string" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const path = requireString(params["path"], "path");
        return { path, content: readNote(path) };
      },
    },
    {
      descriptor: {
        name: "notes_search",
        description: "按关键词搜索笔记正文（不区分大小写子串），返回命中文件与片段。",
        inputSchema: { query: "string", limit: "number?" },
        outputSchema: {},
        ...baseDescriptor,
      },
      handler: (params) => {
        const query = requireString(params["query"], "query").toLowerCase();
        const limit = clampLimit(params["limit"], 10, 50);
        const hits: Array<{ path: string; snippet: string }> = [];
        for (const file of listFiles()) {
          if (hits.length >= limit) break;
          let content: string;
          try {
            content = readFileSync(resolveInside(file.path), "utf8");
          } catch {
            continue;
          }
          const idx = content.toLowerCase().indexOf(query);
          if (idx < 0) continue;
          const start = Math.max(0, idx - 60);
          hits.push({
            path: file.path,
            snippet: content
              .slice(start, idx + query.length + 60)
              .replace(/\s+/g, " ")
              .trim(),
          });
        }
        return { query: params["query"], hits };
      },
    },
  ];

  if (opts.writable) {
    tools.push(
      {
        descriptor: {
          name: "notes_create",
          description:
            "新建一篇笔记（相对路径，须在笔记目录内；已存在则失败，不覆盖）。写操作，需审批。",
          inputSchema: { path: "string", content: "string" },
          outputSchema: {},
          ...writeDescriptor,
        },
        handler: (params) => {
          const path = requireString(params["path"], "path");
          const content = requireContent(params["content"]);
          assertNoteExtension(path);
          const full = resolveInside(path);
          mkdirSync(dirname(full), { recursive: true });
          writeFileSync(full, content, { encoding: "utf8", flag: "wx" });
          return { path, bytes: Buffer.byteLength(content, "utf8") };
        },
      },
      {
        descriptor: {
          name: "notes_append",
          description: "向一篇已存在的笔记末尾追加内容。写操作，需审批。",
          inputSchema: { path: "string", content: "string" },
          outputSchema: {},
          ...writeDescriptor,
        },
        handler: (params) => {
          const path = requireString(params["path"], "path");
          const content = requireContent(params["content"]);
          assertNoteExtension(path);
          const full = resolveInside(path);
          if (!existsSync(full)) throw new Error(`note not found: ${path}`);
          appendFileSync(full, content, "utf8");
          return { path, size: statSync(full).size };
        },
      },
      {
        descriptor: {
          name: "notes_delete",
          description:
            "删除一篇笔记（移入 .trash 回收目录，可人工恢复，不直接抹除）。写操作，需审批。",
          inputSchema: { path: "string" },
          outputSchema: {},
          ...writeDescriptor,
        },
        handler: (params) => {
          const path = requireString(params["path"], "path");
          assertNoteExtension(path);
          const full = resolveInside(path);
          if (!existsSync(full)) throw new Error(`note not found: ${path}`);
          const trashDir = join(root, ".trash");
          mkdirSync(trashDir, { recursive: true });
          const stamp = new Date().toISOString().replace(/[:.]/g, "-");
          const trashedName = `${stamp}-${basename(full)}`;
          renameSync(full, join(trashDir, trashedName));
          return { path, trashedTo: `.trash/${trashedName}` };
        },
      },
    );
  }

  return {
    id,
    displayName: "笔记（本地目录）",
    tools: () => tools,
    status: () => {
      let ok = true;
      let detail: string | undefined;
      try {
        statSync(root);
      } catch {
        ok = false;
        detail = `notes root not found: ${root}`;
      }
      return {
        id,
        displayName: "笔记（本地目录）",
        ok,
        ...(detail ? { detail } : {}),
        toolCount: tools.length,
      };
    },
  };
};

const clampLimit = (value: unknown, fallback: number, max: number): number => {
  const n = typeof value === "number" && Number.isFinite(value) ? Math.floor(value) : fallback;
  return Math.min(Math.max(n, 1), max);
};

const requireString = (value: unknown, field: string): string => {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`missing or invalid param: ${field}`);
  }
  return value;
};
