import { readFile, readdir } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { TaskType, type TaskTypeType } from "@openintj/core";
import { parseFrontmatter } from "./frontmatter.js";
import type { Skill, SkillSource } from "./types.js";

const SKILL_FILENAME = "SKILL.md";
const VALID_TASK_TYPES = new Set<string>(Object.values(TaskType));

export interface FsSkillSourceOpts {
  /** 扫描的技能目录（递归查找 SKILL.md）。约定：每个技能一个子目录，内含 SKILL.md。 */
  dirs: readonly string[];
}

const asString = (v: string | string[] | undefined, fallback: string): string =>
  typeof v === "string" ? v : fallback;

const asStringArray = (v: string | string[] | undefined): string[] =>
  Array.isArray(v) ? v : typeof v === "string" && v.length > 0 ? [v] : [];

/** 递归收集目录下所有 SKILL.md 的绝对路径（目录不存在 / 不可读时静默跳过）。 */
const findSkillFiles = async (dir: string): Promise<string[]> => {
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    const out: string[] = [];
    for (const e of entries) {
      const full = join(dir, e.name);
      if (e.isDirectory()) {
        out.push(...(await findSkillFiles(full)));
      } else if (e.isFile() && e.name === SKILL_FILENAME) {
        out.push(full);
      }
    }
    return out;
  } catch {
    return [];
  }
};

const parseSkillFile = (raw: string, filePath: string): Skill | undefined => {
  const { data, body } = parseFrontmatter(raw);
  const name = asString(data["name"], "").trim();
  const description = asString(data["description"], "").trim();
  // 至少要有 description（进目录 + 参与匹配）与 body（注入内容）。
  if (description.length === 0 || body.length === 0) return undefined;
  // id 兜底：frontmatter.id → 所在目录名 → 文件名。
  const dirName = basename(dirname(filePath));
  const id = asString(data["id"], "").trim() || dirName || basename(filePath);
  const taskTypes = asStringArray(data["taskTypes"]).filter((t): t is TaskTypeType =>
    VALID_TASK_TYPES.has(t),
  );
  const priorityRaw = asString(data["priority"], "0").trim();
  const priority = Number.isFinite(Number(priorityRaw)) ? Number(priorityRaw) : 0;
  return {
    id,
    name: name || id,
    description,
    triggers: asStringArray(data["triggers"]).map((s) => s.toLowerCase()),
    taskTypes,
    priority,
    version: asString(data["version"], "0.0.0").trim() || "0.0.0",
    body,
    source: filePath,
  };
};

/**
 * 文件系统技能来源：从若干目录递归发现 `SKILL.md` 并解析。
 * 同 id 冲突时「后加载覆盖先加载」（dirs 顺序靠后的优先，便于用户目录覆盖内建）。
 */
export class FsSkillSource implements SkillSource {
  readonly name: string;
  private readonly dirs: readonly string[];

  constructor(opts: FsSkillSourceOpts) {
    this.dirs = opts.dirs;
    this.name = `fs-skills:${opts.dirs.join(",")}`;
  }

  async load(): Promise<Skill[]> {
    const byId = new Map<string, Skill>();
    for (const dir of this.dirs) {
      const files = await findSkillFiles(dir);
      for (const file of files) {
        try {
          const raw = await readFile(file, "utf8");
          const skill = parseSkillFile(raw, file);
          if (skill) byId.set(skill.id, skill);
        } catch {
          // 单个技能解析失败不影响其余。
        }
      }
    }
    return [...byId.values()];
  }
}

/**
 * 内建种子技能目录：`<@openintj/skills 包根>/skills`。
 * 用包自身的 import.meta.url 解析，src(vitest/tsx) 与 dist(构建后) 都指向同一个 `../skills`，
 * 与调用方 cwd 无关。种子 SKILL.md 通过 package.json `files` 一起发布。
 */
export const builtinSkillsDir = (): string =>
  join(dirname(fileURLToPath(import.meta.url)), "..", "skills");

/**
 * 解析 SKILL.md 环境目录：内建 `builtinDir` + `OPENINTJ_SKILLS_DIR`（分号 / 逗号分隔）追加。
 * 不用冒号分隔，避免误切 Windows 盘符路径（`C:\...`）。
 * 返回去重后的目录列表（顺序：内建在前，用户目录在后 → 用户可覆盖同 id）。
 */
export const resolveSkillDirs = (
  builtinDir: string | undefined,
  env: NodeJS.ProcessEnv = process.env,
): string[] => {
  const dirs: string[] = [];
  if (builtinDir) dirs.push(builtinDir);
  const extra = env["OPENINTJ_SKILLS_DIR"]?.trim();
  if (extra) {
    for (const d of extra.split(/[;,]/).map((s) => s.trim())) {
      if (d.length > 0 && !dirs.includes(d)) dirs.push(d);
    }
  }
  return dirs;
};
