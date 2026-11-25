import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { LlmClient } from "@openintj/core";
import { resolveInRoot } from "./workspace-tools.js";

/** 平台无关生图意图检测（P1-8，对齐 desktop artifact-fallback）。 */
export const requestsImageGeneration = (query: string): boolean => {
  if (/(?:分析|识别|解读|看看).{0,12}(?:这张|该|图片|截图|照片)/u.test(query)) return false;
  if (/(?:幻灯片|ppt|pptx|表格|xlsx|spreadsheet|powerpoint)/iu.test(query)) return false;
  return (
    /(?:生成|画|绘制|做|制作|create|generate|draw|make).{0,40}(?:封面|海报|配图|插图|图片|图像|cover|poster|illustration)/iu.test(
      query,
    ) || /(?:封面图|生图|文生图|text[- ]to[- ]image)/iu.test(query)
  );
};

const coverTitleFromQuery = (query: string): string => {
  const m =
    query.match(/[「「]([^」」]{1,40})[」」]/u) ??
    query.match(/主题[是为]?[：:]\s*([^\s，。,]{1,40})/u);
  return m?.[1]?.trim() || "封面";
};

const fallbackCoverSvg = (query: string): string => {
  const title = coverTitleFromQuery(query);
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">
  <rect width="100%" height="100%" fill="#111827"/>
  <text x="64" y="360" fill="#f9fafb" font-size="48" font-family="sans-serif">${title}</text>
  <text x="64" y="420" fill="#9ca3af" font-size="20" font-family="sans-serif">评论文章封面（fallback SVG）</text>
</svg>`;
};

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
    const candidate = current as { generateImage?: GenerateImageFn; inner?: unknown };
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

/** 平台无关生图落盘（CLI/server/desktop 共用）。 */
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
  const prompt = `评论文章封面，横版 16:9，主题「${coverTitleFromQuery(opts.query)}」。要求：${opts.query}`;
  if (generateImage) {
    try {
      const image = await generateImage(prompt, {
        size: "1280x720",
        ...(opts.signal ? { signal: opts.signal } : {}),
      });
      const path = `artifacts/${opts.taskId}/cover.${extensionFor(image.mimeType)}`;
      await writeBytes(path, image.bytes);
      return {
        path,
        fallback: false,
        finalAnswer: `已生成封面并写入工作区：\`${path}\``,
      };
    } catch (error) {
      await writeUtf8(svgPath, fallbackCoverSvg(opts.query));
      const detail = error instanceof Error ? error.message : String(error);
      return {
        path: svgPath,
        fallback: true,
        finalAnswer: `生图调用失败，已写入：\`${svgPath}\`。${detail}`,
      };
    }
  }

  await writeUtf8(svgPath, fallbackCoverSvg(opts.query));
  return {
    path: svgPath,
    fallback: true,
    finalAnswer: `当前配置无生图能力。已写入：\`${svgPath}\`。`,
  };
};

export const taskIdFromContextTags = (tags: readonly string[] | undefined): string =>
  tags?.find((tag) => tag.startsWith("task:"))?.slice("task:".length) || "inbox";
