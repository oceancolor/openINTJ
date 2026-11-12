import { mkdtemp, readdir, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { mkdirWorkspaceDir, renameWorkspaceDir } from "../src/main/workspace-fs.js";

const makeRoot = (): Promise<string> => mkdtemp(join(tmpdir(), "openintj-ws-fs-"));

describe("workspace-fs 任务文件夹联动", () => {
  it("mkdir 建目录且幂等", async () => {
    const root = await makeRoot();
    await expect(mkdirWorkspaceDir(root, "贵州自驾")).resolves.toEqual({
      path: "贵州自驾",
      created: true,
    });
    await expect(mkdirWorkspaceDir(root, "贵州自驾")).resolves.toEqual({
      path: "贵州自驾",
      created: false,
    });
    expect((await stat(join(root, "贵州自驾"))).isDirectory()).toBe(true);
  });

  it("mkdir 越界路径被拒绝", async () => {
    const root = await makeRoot();
    await expect(mkdirWorkspaceDir(root, "../evil")).rejects.toThrow();
  });

  it("rename 改名目录，目标已存在时拒绝", async () => {
    const root = await makeRoot();
    await mkdirWorkspaceDir(root, "旧名");
    await writeFile(join(root, "旧名", "a.txt"), "x");
    await expect(renameWorkspaceDir(root, "旧名", "新名")).resolves.toEqual({
      from: "旧名",
      to: "新名",
    });
    expect(await readdir(join(root, "新名"))).toEqual(["a.txt"]);

    await mkdirWorkspaceDir(root, "另一个");
    await expect(renameWorkspaceDir(root, "新名", "另一个")).rejects.toThrow();
    await expect(renameWorkspaceDir(root, "不存在", "x")).rejects.toThrow();
  });
});
