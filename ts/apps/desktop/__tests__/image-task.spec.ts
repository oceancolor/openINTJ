import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import { fulfillImageGenerationTask } from "../src/main/image-task.js";

const roots: string[] = [];
afterAll(() => {
  for (const root of roots) rmSync(root, { recursive: true, force: true });
});

const mockChatLlm = {
  chat: vi.fn(),
  visionChat: vi.fn(),
  getStatus: () => ({
    provider: "mock",
    model: "mock",
    available: true,
    mode: "live" as const,
    status: "connected" as const,
    visionSupported: false,
  }),
};

describe("fulfillImageGenerationTask", () => {
  it("writes a PNG when the client can generate images", async () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-img-"));
    roots.push(root);
    const llm = {
      ...mockChatLlm,
      generateImage: vi.fn(async () => ({
        url: "https://img.example/cover.png",
        bytes: Buffer.from("PNGDATA"),
        mimeType: "image/png",
      })),
    };
    const result = await fulfillImageGenerationTask({
      query: "请生成一张以「大厂病」为主题的评论文章封面图",
      workspaceRoot: root,
      taskId: "task-1",
      llm,
    });
    expect(result.fallback).toBe(false);
    expect(result.path).toBe("artifacts/task-1/cover.png");
    expect(readFileSync(join(root, result.path), "utf8")).toBe("PNGDATA");
    expect(llm.generateImage).toHaveBeenCalledOnce();
    expect(mockChatLlm.chat).not.toHaveBeenCalled();
  });

  it("unwraps StrictLlmWrapper-style inner clients to reach generateImage", async () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-img-"));
    roots.push(root);
    const generateImage = vi.fn(async () => ({
      url: "https://img.example/cover.png",
      bytes: Buffer.from("PNGDATA"),
      mimeType: "image/png",
    }));
    const result = await fulfillImageGenerationTask({
      query: "请生成一张封面图",
      workspaceRoot: root,
      taskId: "task-wrap",
      llm: {
        ...mockChatLlm,
        inner: { generateImage },
      } as typeof mockChatLlm & { inner: { generateImage: typeof generateImage } },
    });
    expect(result.fallback).toBe(false);
    expect(result.path).toBe("artifacts/task-wrap/cover.png");
    expect(generateImage).toHaveBeenCalledOnce();
  });

  it("falls back to SVG when the chat model cannot generate images", async () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-img-"));
    roots.push(root);
    const result = await fulfillImageGenerationTask({
      query: "请生成一张以「大厂病」为主题的评论文章封面图",
      workspaceRoot: root,
      taskId: "task-2",
      llm: mockChatLlm,
    });
    expect(result.fallback).toBe(true);
    expect(result.path).toBe("artifacts/task-2/cover.svg");
    expect(readFileSync(join(root, result.path), "utf8")).toContain("大厂病");
    expect(result.finalAnswer).toContain("没有生图能力");
  });

  it("surfaces the TokenHub error when generateImage fails", async () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-img-"));
    roots.push(root);
    const result = await fulfillImageGenerationTask({
      query: "请生成一张封面图",
      workspaceRoot: root,
      taskId: "task-402",
      llm: {
        ...mockChatLlm,
        generateImage: vi.fn(async () => {
          throw new Error("Hunyuan 生图失败 (402): 未开通 Hy-Image-3.0 后付费且无免费额度");
        }),
      },
    });
    expect(result.fallback).toBe(true);
    expect(result.finalAnswer).toContain("混元生图调用失败");
    expect(result.finalAnswer).toContain("402");
    expect(result.finalAnswer).toMatch(/已写入工作区：`artifacts\/task-402\/cover\.svg`/);
  });
});
