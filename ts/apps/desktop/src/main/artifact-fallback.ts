/** Detect user asks that should leave a workspace file even if the model skipped write_file. */

import { requestsArtifactPreview } from "../shared/workspace-view.js";

export const requestsImageArtifact = (query: string): boolean =>
  /封面|海报|配图|插图|图片|图像|cover\s*image|poster|\.svg\b|\.png\b|\.jpe?g\b|\.webp\b/iu.test(
    query,
  );

/** 需要调用生图模型，而不是对话/视觉理解模型。 */
export const requestsImageGeneration = (query: string): boolean => {
  if (requestsArtifactPreview(query)) return false;
  return (
    /(?:生成|画|绘制|做|制作|create|generate|draw|make).{0,40}(?:封面|海报|配图|插图|图片|图像|cover|poster|illustration)/iu.test(
      query,
    ) || /(?:封面图|生图|文生图|text[- ]to[- ]image)/iu.test(query)
  );
};

export const requestsFileArtifact = (query: string): boolean =>
  requestsImageArtifact(query) ||
  /(?:创建|生成|写入|保存|输出|落盘|create|generate|write|save).{0,40}(?:文件|文档|代码|脚本|file|document|script|[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)/iu.test(
    query,
  );

export const fallbackArtifactPath = (query: string, taskId: string): string => {
  const explicit = query.match(
    /(?:^|[\s"'`（(])([A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.[A-Za-z0-9]{1,12})(?=$|[\s"'`，。）、)])/u,
  )?.[1];
  if (explicit) return explicit.replaceAll("\\", "/");
  if (requestsImageArtifact(query)) return `artifacts/${taskId}/cover.svg`;
  return `artifacts/${taskId}/result.md`;
};

export const extractSvg = (text: string): string | undefined => {
  const match = text.match(/<svg\b[\s\S]*<\/svg>/i);
  return match?.[0];
};

export const coverTitleFromQuery = (query: string): string => {
  const quoted = query.match(/[「『""]([^」』""]{1,40})[」』""]/u)?.[1];
  if (quoted?.trim()) return quoted.trim();
  const themed = query.match(/以\s*[「『""]?([^」』""，。\s]{1,20})[」』""]?\s*为主题/u)?.[1];
  if (themed?.trim()) return themed.trim();
  return "封面";
};

const escapeXml = (value: string): string =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

export const fallbackCoverSvg = (query: string): string => {
  const title = escapeXml(coverTitleFromQuery(query));
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <rect width="1200" height="630" fill="#1e1e2e"/>
  <rect x="48" y="48" width="1104" height="534" fill="none" stroke="#cba6f7" stroke-width="4"/>
  <text x="600" y="320" text-anchor="middle" fill="#cdd6f4" font-size="72" font-family="serif">${title}</text>
  <text x="600" y="400" text-anchor="middle" fill="#a6adc8" font-size="28" font-family="sans-serif">评论文章封面</text>
</svg>
`;
};

export const resolveMissingArtifact = (
  query: string,
  taskId: string,
  finalAnswer: string,
): { path: string; content: string } => {
  const path = fallbackArtifactPath(query, taskId);
  if (requestsImageArtifact(query)) {
    return { path, content: extractSvg(finalAnswer) ?? fallbackCoverSvg(query) };
  }
  return { path, content: finalAnswer };
};
