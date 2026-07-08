import { TaskType } from "@openintj/core";
import { describe, expect, it } from "vitest";
import type { TrajectorySample } from "../src/learning-runtime.js";
import { type SkillDistillLlm, createLlmSkillDistiller } from "../src/llm-distill.js";

const sample = (over: Partial<TrajectorySample> = {}): TrajectorySample => ({
  query: "fix the failing test",
  finalAnswer: "done",
  toolsUsed: [],
  ts: 1,
  ...over,
});

const llmOf = (out: string): SkillDistillLlm => ({ generate: async () => out });

describe("createLlmSkillDistiller", () => {
  it("解析裸 JSON 数组，归一 triggers（小写去重）", async () => {
    const distill = createLlmSkillDistiller(
      llmOf(
        JSON.stringify([
          {
            id: "fix-tests",
            name: "Fix Tests",
            description: "systematically fix failing tests",
            triggers: ["Test", "test", "  CI  "],
            body: "Read the failure, reproduce, then patch the root cause.",
          },
        ]),
      ),
    );
    const drafts = await distill([sample()]);
    expect(drafts).toHaveLength(1);
    expect(drafts[0]!.id).toBe("fix-tests");
    expect(drafts[0]!.triggers).toEqual(["test", "ci"]);
  });

  it("容忍 ```json 栅栏 + 前后散文", async () => {
    const distill = createLlmSkillDistiller(
      llmOf(
        'Sure! Here you go:\n```json\n[{"name":"N","body":"a reusable body long enough"}]\n```\nHope that helps.',
      ),
    );
    const drafts = await distill([sample()]);
    expect(drafts[0]!.name).toBe("N");
  });

  it("单对象（非数组）也接受", async () => {
    const distill = createLlmSkillDistiller(
      llmOf('{"name":"Solo","body":"a body that is long enough to pass"}'),
    );
    const drafts = await distill([sample()]);
    expect(drafts).toHaveLength(1);
    expect(drafts[0]!.name).toBe("Solo");
  });

  it("body 过短 → 丢弃该草案；全丢则抛错触发回退", async () => {
    const distill = createLlmSkillDistiller(llmOf('[{"name":"X","body":"ok"}]'), {
      minBodyChars: 16,
    });
    await expect(distill([sample()])).rejects.toThrow(/no usable skill drafts/);
  });

  it("name/body 缺失的项被过滤", async () => {
    const distill = createLlmSkillDistiller(
      llmOf(
        JSON.stringify([
          { name: "NoBody" },
          { body: "no name here but long enough body" },
          { name: "Good", body: "a sufficiently long reusable body here" },
        ]),
      ),
    );
    const drafts = await distill([sample()]);
    expect(drafts.map((d) => d.name)).toEqual(["Good"]);
  });

  it("taskTypes 校验到合法枚举，幻觉类型被过滤", async () => {
    const distill = createLlmSkillDistiller(
      llmOf(
        JSON.stringify([
          {
            name: "T",
            body: "a sufficiently long reusable body here",
            taskTypes: [TaskType.ANALYSIS, "not_a_type", "also_fake"],
          },
        ]),
      ),
    );
    const drafts = await distill([sample()]);
    expect(drafts[0]!.taskTypes).toEqual([TaskType.ANALYSIS]);
  });

  it("全部 taskTypes 非法 → 不带 taskTypes 字段", async () => {
    const distill = createLlmSkillDistiller(
      llmOf('[{"name":"T","body":"a sufficiently long reusable body","taskTypes":["fake"]}]'),
    );
    const drafts = await distill([sample()]);
    expect(drafts[0]!.taskTypes).toBeUndefined();
  });

  it("tools 归一去重并透传", async () => {
    const distill = createLlmSkillDistiller(
      llmOf(
        '[{"name":"T","body":"a sufficiently long reusable body","tools":["readFile","readFile","search"]}]',
      ),
    );
    const drafts = await distill([sample()]);
    expect(drafts[0]!.tools).toEqual(["readFile", "search"]);
  });

  it("name / body / description 超长被截断", async () => {
    const long = "x".repeat(500);
    const distill = createLlmSkillDistiller(
      llmOf(JSON.stringify([{ name: long, description: long, body: long }])),
      { maxNameChars: 80, maxDescriptionChars: 240, maxBodyChars: 100 },
    );
    const drafts = await distill([sample()]);
    expect(drafts[0]!.name.length).toBe(80);
    expect(drafts[0]!.description.length).toBe(240);
    expect(drafts[0]!.body.length).toBe(100);
  });

  it("批内按 id/name 去重，取 maxDrafts", async () => {
    const distill = createLlmSkillDistiller(
      llmOf(
        JSON.stringify([
          { id: "dup", name: "A", body: "a sufficiently long reusable body 1" },
          { id: "dup", name: "B", body: "a sufficiently long reusable body 2" },
          { name: "Third", body: "a sufficiently long reusable body 3" },
        ]),
      ),
      { maxDrafts: 5 },
    );
    const drafts = await distill([sample()]);
    expect(drafts.map((d) => d.id ?? d.name)).toEqual(["dup", "Third"]);
  });

  it("无 JSON → 抛错（触发 runtime 回退启发式）", async () => {
    const distill = createLlmSkillDistiller(llmOf("I could not find a reusable pattern."));
    await expect(distill([sample()])).rejects.toThrow(/no JSON found/);
  });

  it("空数组 [] → 抛错（无候选）", async () => {
    const distill = createLlmSkillDistiller(llmOf("[]"));
    await expect(distill([sample()])).rejects.toThrow(/no usable skill drafts/);
  });
});
