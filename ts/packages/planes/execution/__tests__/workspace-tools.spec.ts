import { mkdtempSync, rmSync } from "node:fs";
import { readFile as fsReadFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { createWorkspaceTools, resolveInRoot } from "../src/workspace-tools.js";

const roots: string[] = [];
const makeRoot = (): string => {
  const d = mkdtempSync(join(tmpdir(), "openintj-ws-"));
  roots.push(d);
  return d;
};
afterAll(() => {
  for (const d of roots) {
    try {
      rmSync(d, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
});

describe("resolveInRoot", () => {
  it("解析相对路径到 root 内", () => {
    const abs = resolveInRoot("/work", "a/b.txt");
    expect(abs.replace(/\\/g, "/")).toContain("/work/a/b.txt");
  });
  it("拒绝 .. 越界", () => {
    expect(() => resolveInRoot("/work", "../escape.txt")).toThrow(/越界/);
    expect(() => resolveInRoot("/work", "a/../../escape.txt")).toThrow(/越界/);
  });
  it("拒绝绝对路径输入", () => {
    expect(() => resolveInRoot("/work", "/etc/passwd")).toThrow(/绝对路径/);
  });
});

describe("createWorkspaceTools - fs", () => {
  it("写入后能读回，且 bytes 一致", async () => {
    const root = makeRoot();
    const tools = createWorkspaceTools({ root });
    const w = (await tools.writeFile({ path: "notes/a.txt", content: "你好 hello" })) as {
      bytesWritten: number;
    };
    expect(w.bytesWritten).toBe(Buffer.byteLength("你好 hello", "utf8"));
    const r = (await tools.readFile({ path: "notes/a.txt" })) as { content: string; bytes: number };
    expect(r.content).toBe("你好 hello");
    // 确认真落盘到 root 内
    const onDisk = await fsReadFile(join(root, "notes", "a.txt"), "utf8");
    expect(onDisk).toBe("你好 hello");
  });

  it("读写越界路径被拒绝", async () => {
    const tools = createWorkspaceTools({ root: makeRoot() });
    await expect(tools.readFile({ path: "../../etc/passwd" })).rejects.toThrow(/越界/);
    await expect(tools.writeFile({ path: "../evil.txt", content: "x" })).rejects.toThrow(/越界/);
  });

  it("缺 path 参数报错", async () => {
    const tools = createWorkspaceTools({ root: makeRoot() });
    await expect(tools.readFile({})).rejects.toThrow(/path/);
    await expect(tools.writeFile({ content: "x" })).rejects.toThrow(/path/);
  });

  it("超过读上限报错", async () => {
    const root = makeRoot();
    const tools = createWorkspaceTools({ root, maxReadBytes: 4 });
    await tools.writeFile({ path: "big.txt", content: "0123456789" });
    await expect(tools.readFile({ path: "big.txt" })).rejects.toThrow(/过大/);
  });

  it("超过写上限报错", async () => {
    const tools = createWorkspaceTools({ root: makeRoot(), maxWriteBytes: 4 });
    await expect(tools.writeFile({ path: "x.txt", content: "0123456789" })).rejects.toThrow(/过大/);
  });
});

describe("createWorkspaceTools - executeCommand", () => {
  it("默认禁用命令执行", async () => {
    const tools = createWorkspaceTools({ root: makeRoot() });
    await expect(tools.executeCommand({ command: "echo hi" })).rejects.toThrow(/未启用/);
  });

  it("启用但不在白名单 → 拒绝", async () => {
    const tools = createWorkspaceTools({
      root: makeRoot(),
      enableCommands: true,
      allowedCommands: ["node"],
    });
    await expect(tools.executeCommand({ command: "rm -rf /" })).rejects.toThrow(/白名单/);
  });

  it("白名单内命令可执行并返回结构化结果", async () => {
    const tools = createWorkspaceTools({
      root: makeRoot(),
      enableCommands: true,
      allowedCommands: ["node"],
    });
    const r = (await tools.executeCommand({
      command: "node -e \"process.stdout.write('ok')\"",
    })) as {
      stdout: string;
      exitCode: number;
    };
    expect(r.stdout).toContain("ok");
    expect(r.exitCode).toBe(0);
  });

  it("命令非 0 退出 → 返回 exitCode 而非抛异常", async () => {
    const tools = createWorkspaceTools({
      root: makeRoot(),
      enableCommands: true,
      allowedCommands: ["node"],
    });
    const r = (await tools.executeCommand({ command: 'node -e "process.exit(3)"' })) as {
      exitCode: number;
    };
    expect(r.exitCode).toBe(3);
  });
});
