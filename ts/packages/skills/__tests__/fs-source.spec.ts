import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { parseFrontmatter } from "../src/frontmatter.js";
import { FsSkillSource, builtinSkillsDir, resolveSkillDirs } from "../src/fs-source.js";

describe("parseFrontmatter", () => {
  it("解析标量 / 内联数组 / 块状列表 / 引号", () => {
    const raw = [
      "---",
      "id: code-review",
      'name: "Code Review"',
      "description: 审查代码",
      "triggers: [review, 代码审查]",
      "taskTypes:",
      "  - code_generation",
      "  - analysis",
      "priority: 10",
      "# 注释行",
      "---",
      "正文第一行",
      "正文第二行",
    ].join("\n");
    const { data, body } = parseFrontmatter(raw);
    expect(data["id"]).toBe("code-review");
    expect(data["name"]).toBe("Code Review");
    expect(data["triggers"]).toEqual(["review", "代码审查"]);
    expect(data["taskTypes"]).toEqual(["code_generation", "analysis"]);
    expect(data["priority"]).toBe("10");
    expect(body).toBe("正文第一行\n正文第二行");
  });

  it("无 frontmatter 时整体当 body", () => {
    const { data, body } = parseFrontmatter("# 只有正文\n内容");
    expect(Object.keys(data)).toHaveLength(0);
    expect(body).toBe("# 只有正文\n内容");
  });
});

describe("resolveSkillDirs", () => {
  it("内建在前、OPENINTJ_SKILLS_DIR 追加、去重、不切盘符冒号", () => {
    const dirs = resolveSkillDirs("C:\\builtin\\skills", {
      OPENINTJ_SKILLS_DIR: "D:\\a\\skills; C:\\builtin\\skills ,E:\\b",
    } as NodeJS.ProcessEnv);
    expect(dirs).toEqual(["C:\\builtin\\skills", "D:\\a\\skills", "E:\\b"]);
  });

  it("无内建目录 / 无 env 时为空", () => {
    expect(resolveSkillDirs(undefined, {} as NodeJS.ProcessEnv)).toEqual([]);
  });
});

describe("FsSkillSource", () => {
  let root: string;

  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), "openintj-skills-"));
    // 合法技能：目录名做 id 兜底 + 非法 taskType 被过滤。
    await mkdir(join(root, "code-review"), { recursive: true });
    await writeFile(
      join(root, "code-review", "SKILL.md"),
      [
        "---",
        "name: Code Review",
        "description: 审查代码找 bug",
        "triggers: [review, pr]",
        "taskTypes: [code_generation, not_a_real_type]",
        "priority: 5",
        "tools: [readFile, search]",
        "---",
        "审查步骤：读 diff → 找问题 → 给建议。",
      ].join("\n"),
      "utf8",
    );
    // 缺 description → 应被跳过。
    await mkdir(join(root, "broken"), { recursive: true });
    await writeFile(join(root, "broken", "SKILL.md"), "---\nname: X\n---\n只有正文", "utf8");
    // 非 SKILL.md → 忽略。
    await writeFile(join(root, "code-review", "README.md"), "ignore me", "utf8");
  });

  afterAll(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it("递归发现并解析 SKILL.md，id 兜底目录名，非法 taskType 过滤，缺字段跳过", async () => {
    const src = new FsSkillSource({ dirs: [root] });
    const skills = await src.load();
    expect(skills).toHaveLength(1);
    const s = skills[0]!;
    expect(s.id).toBe("code-review");
    expect(s.name).toBe("Code Review");
    expect(s.triggers).toEqual(["review", "pr"]);
    expect(s.taskTypes).toEqual(["code_generation"]);
    expect(s.priority).toBe(5);
    expect(s.tools).toEqual(["read_file", "search"]);
    expect(s.body).toContain("审查步骤");
  });

  it("未声明 tools 的技能 tools 为空数组", async () => {
    const dir = await mkdtemp(join(tmpdir(), "openintj-skills-notools-"));
    await mkdir(join(dir, "plain"), { recursive: true });
    await writeFile(
      join(dir, "plain", "SKILL.md"),
      ["---", "description: 无工具技能", "---", "正文"].join("\n"),
      "utf8",
    );
    const skills = await new FsSkillSource({ dirs: [dir] }).load();
    expect(skills[0]?.tools).toEqual([]);
    await rm(dir, { recursive: true, force: true });
  });

  it("目录不存在时返回空、不抛错", async () => {
    const src = new FsSkillSource({ dirs: [join(root, "nope")] });
    await expect(src.load()).resolves.toEqual([]);
  });

  it("ships RFC-006 planning and clarification skills with deterministic triggers", async () => {
    const skills = await new FsSkillSource({ dirs: [builtinSkillsDir()] }).load();
    const planning = skills.find((s) => s.id === "planning");
    const clarification = skills.find((s) => s.id === "clarification");
    expect(planning?.taskTypes).toEqual(expect.arrayContaining(["planning", "analysis"]));
    expect(planning?.triggers).toContain("规划");
    expect(clarification?.taskTypes).toContain("planning");
    expect(clarification?.triggers).toContain("澄清");
  });
});
