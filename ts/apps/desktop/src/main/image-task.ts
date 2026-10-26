import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { LlmClient } from "@openintj/core";
import { resolveInRoot } from "@openintj/plane-execution";
import {
  coverTitleFromQuery,
  fallbackCoverSvg,
  requestsImageGeneration,
} from "./artifact-fallback.js";

export { requestsImageGeneration };

export interface ImageTaskResult {
  path: string;
  finalAnswer: string;
  fallback: boolean;
}

type GenerateImageFn = (
  prompt: string,
  opts?: { size?: string; signal?: AbortSignal },
) => Promise<{ url: string; bytes: Uint8Array; mimeType: string }>;

const findGenerateImage = (client: LlmClient): GenerateImageFn | undefined => {
  let current: unknown = client;
  const seen = new Set<unknown>();
  while (current && typeof current === "object" && !seen.has(current)) {
    seen.add(current);
    const candidate = current as {
      generateImage?: GenerateImageFn;
      inner?: unknown;
    };
    if (typeof candidate.generateImage === "function") {
      return candidate.generateImage.bind(current);
    }
    current = candidate.inner;
  }
  return undefined;
};

const extensionFor = (mimeType: string): string => {
  if (mimeType === "image/jpeg") return "jpg";
  if (mimeType === "image/webp") return "webp";
  return "png";
};

const imagePrompt = (query: string): string => {
  const title = coverTitleFromQuery(query);
  return `评论文章封面，横版 16:9，信息图风格，主题「${title}」。要求：${query}`;
};

export const fulfillImageGenerationTask = async (opts: {
  query: string;
  workspaceRoot: string;
  taskId: string;
  llm: LlmClient;
  signal?: AbortSignal;
}): Promise<ImageTaskResult> => {
  const svgPath = `artifacts/${opts.taskId}/cover.svg`;
  const writeUtf8 = async (rel: string, content: string): Promise<void> => {
    const abs = resolveInRoot(opts.workspaceRoot, rel);
    await mkdir(dirname(abs), { recursive: true });
    await writeFile(abs, content, "utf8");
  };
  const writeBytes = async (rel: string, bytes: Uint8Array): Promise<void> => {
    const abs = resolveInRoot(opts.workspaceRoot, rel);
    await mkdir(dirname(abs), { recursive: true });
    await writeFile(abs, bytes);
  };

  const generateImage = findGenerateImage(opts.llm);
  if (generateImage) {
    try {
      const image = await generateImage(imagePrompt(opts.query), {
        size: "1280x720",
        ...(opts.signal ? { signal: opts.signal } : {}),
      });
      const path = `artifacts/${opts.taskId}/cover.${extensionFor(image.mimeType)}`;
      await writeBytes(path, image.bytes);
      return {
        path,
        fallback: false,
        finalAnswer: `已用混元生图模型 Hy-Image-3.0 生成封面。已写入工作区：\`${path}\``,
      };
    } catch (error) {
      console.warn("[OpenINTJ desktop] hunyuan image generation failed; falling back to SVG", error);
      await writeUtf8(svgPath, fallbackCoverSvg(opts.query));
      const detail = error instanceof Error ? error.message : String(error);
      return {
        path: svgPath,
        fallback: true,
        finalAnswer:
          `混元生图调用失败，已写入工作区：\`${svgPath}\`。${detail}`,
      };
    }
  }

  await writeUtf8(svgPath, fallbackCoverSvg(opts.query));
  return {
    path: svgPath,
    fallback: true,
    finalAnswer:
      `当前对话配置档没有生图能力。已写入工作区：\`${svgPath}\`。` +
      "请改用混元对话配置档，并在 TokenHub 开通 Hy-Image-3.0（hy-image-v3）。",
  };
};

export const taskIdFromContextTags = (tags: readonly string[] | undefined): string =>
  tags?.find((tag) => tag.startsWith("task:"))?.slice("task:".length) || "inbox";
