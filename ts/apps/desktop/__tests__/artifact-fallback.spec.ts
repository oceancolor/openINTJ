import { describe, expect, it } from "vitest";
import {
  OFFICE_DOWNGRADE_BANNER,
  coverTitleFromQuery,
  downgradeOfficePathToMarkdown,
  fallbackArtifactPath,
  fallbackCoverSvg,
  requestsFileArtifact,
  requestsImageArtifact,
  requestsImageGeneration,
  requestsSlideOrDeck,
  resolveMissingArtifact,
} from "../src/main/artifact-fallback.js";

describe("artifact fallback", () => {
  it("treats cover-image asks as file artifacts", () => {
    const query = "请生成一张以「大厂病」为主题的评论文章封面图";
    expect(requestsImageArtifact(query)).toBe(true);
    expect(requestsImageGeneration(query)).toBe(true);
    expect(requestsFileArtifact(query)).toBe(true);
    expect(fallbackArtifactPath(query, "task-1")).toBe("artifacts/task-1/cover.svg");
    expect(coverTitleFromQuery(query)).toBe("大厂病");
  });

  it("keeps explicit filenames and markdown fallback for documents", () => {
    expect(fallbackArtifactPath("请生成 report.md 文件", "task-1")).toBe("report.md");
    expect(fallbackArtifactPath("请生成一份项目文档", "task-1")).toBe("artifacts/task-1/result.md");
  });

  it("downgrades explicit Office paths to .md (P0-5)", () => {
    expect(fallbackArtifactPath("请生成 quarterly.docx", "task-1")).toBe("quarterly.md");
    expect(fallbackArtifactPath("写入 deck.pptx", "t")).toBe("deck.md");
    expect(fallbackArtifactPath("保存 data.xlsx", "t")).toBe("data.md");
    expect(downgradeOfficePathToMarkdown("a/b/report.DOCX")).toBe("a/b/report.md");
  });

  it("treats 幻灯片/PPT/表格 as file intents and not image generation", () => {
    const slide = "请做一份带封面的幻灯片";
    expect(requestsSlideOrDeck(slide)).toBe(true);
    expect(requestsFileArtifact(slide)).toBe(true);
    expect(requestsImageGeneration(slide)).toBe(false);
    expect(fallbackArtifactPath(slide, "task-2")).toBe("artifacts/task-2/slides.md");

    const sheet = "生成一份销售表格";
    expect(requestsFileArtifact(sheet)).toBe(true);
    expect(requestsImageGeneration(sheet)).toBe(false);
    expect(fallbackArtifactPath(sheet, "task-3")).toBe("artifacts/task-3/sheet.md");
  });

  it("banners markdown when Office/slide/sheet was requested", () => {
    const artifact = resolveMissingArtifact(
      "请生成季度报告 quarterly.docx",
      "task-1",
      "这是报告正文。",
    );
    expect(artifact.path).toBe("quarterly.md");
    expect(artifact.content.startsWith(OFFICE_DOWNGRADE_BANNER.trim())).toBe(true);
    expect(artifact.content).toContain("这是报告正文。");
  });

  it("materializes a themed SVG when the model omitted write_file", () => {
    const query = "请生成一张以「大厂病」为主题的评论文章封面图";
    const artifact = resolveMissingArtifact(query, "task-9", "先检查 Python 环境。");
    expect(artifact.path).toBe("artifacts/task-9/cover.svg");
    expect(artifact.content).toContain("<svg");
    expect(artifact.content).toContain("大厂病");
    expect(fallbackCoverSvg(query)).toContain("评论文章封面");
  });

  it("reuses an SVG already present in the model answer", () => {
    const artifact = resolveMissingArtifact(
      "生成封面图",
      "t",
      '说明如下\n<svg xmlns="http://www.w3.org/2000/svg"></svg>',
    );
    expect(artifact.content).toBe('<svg xmlns="http://www.w3.org/2000/svg"></svg>');
  });

  it("does not route vision-understanding asks to the image model", () => {
    expect(requestsImageGeneration("请分析这张图片里的表格")).toBe(false);
    expect(requestsImageArtifact("请分析这张图片里的表格")).toBe(true);
  });

  it("does not treat preview/show requests as image generation", () => {
    expect(requestsImageGeneration("预览刚才生图的产物")).toBe(false);
    expect(requestsImageGeneration("在预览区域展示")).toBe(false);
  });
});
