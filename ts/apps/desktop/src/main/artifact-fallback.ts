/** Detect user asks that should leave a workspace file even if the model skipped write_file. */

import { requestsArtifactPreview } from "../shared/workspace-view.js";

/** OOXML 扩展名：当前 write_file 只能写 UTF-8，绝不能以这些扩展名落盘假文件。 */
const OFFICE_EXT_RE = /\.(docx|pptx|xlsx)$/iu;

export const isOfficeExtensionPath = (path: string): boolean => OFFICE_EXT_RE.test(path);

/** 把假 Office 路径降为同名 .md（保留目录前缀）。 */
export const downgradeOfficePathToMarkdown = (path: string): string =>
  path.replace(OFFICE_EXT_RE, ".md");

export const OFFICE_DOWNGRADE_BANNER =
  "> 当前无法生成原生 Office 二进制（.docx/.pptx/.xlsx），已降级为 Markdown。用 Word/WPS/Excel 打开前请另存为对应格式。\n";

export const requestsImageArtifact = (query: string): boolean =>
  /封面|海报|配图|插图|图片|图像|cover\s*image|poster|\.svg\b|\.png\b|\.jpe?g\b|\.webp\b/iu.test(
    query,
  );

/** 幻灯片/PPT 意图：优先当文件产物，不要因「封面」误判成生图。 */
export const requestsSlideOrDeck = (query: string): boolean =>
  /幻灯片|演示文稿|PPT|pptx|powerpoint/iu.test(query);

/** 表格/电子表格意图。 */
export const requestsSpreadsheet = (query: string): boolean =>
  /(?:电子)?表格|\.xlsx\b|excel|spreadsheet/iu.test(query);

/** 需要调用生图模型，而不是对话/视觉理解模型。 */
export const requestsImageGeneration = (query: string): boolean => {
  if (requestsArtifactPreview(query)) return false;
  // 「带封面的幻灯片」是文档意图，不是文生图。
  if (requestsSlideOrDeck(query) || requestsSpreadsheet(query)) return false;
  return (
    /(?:生成|画|绘制|做|制作|create|generate|draw|make).{0,40}(?:封面|海报|配图|插图|图片|图像|cover|poster|illustration)/iu.test(
      query,
    ) || /(?:封面图|生图|文生图|text[- ]to[- ]image)/iu.test(query)
  );
};

export const requestsFileArtifact = (query: string): boolean =>
  requestsImageArtifact(query) ||
  requestsSlideOrDeck(query) ||
  requestsSpreadsheet(query) ||
  /(?:创建|生成|写入|保存|输出|落盘|create|generate|write|save).{0,40}(?:文件|文档|代码|脚本|file|document|script|[A-Za-z0-9_.-]+\.[A-Za-z0-9]+)/iu.test(
    query,
  );

export const fallbackArtifactPath = (query: string, taskId: string): string => {
  const explicit = query.match(
    /(?:^|[\s"'`（(])([A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.[A-Za-z0-9]{1,12})(?=$|[\s"'`，。）、)])/u,
  )?.[1];
  if (explicit) {
    const path = explicit.replaceAll("\\", "/");
    // P0-5：无 Office 原生工具时绝不产出 .docx/.pptx/.xlsx 路径。
    return isOfficeExtensionPath(path) ? downgradeOfficePathToMarkdown(path) : path;
  }
  if (requestsImageArtifact(query) && !requestsSlideOrDeck(query) && !requestsSpreadsheet(query)) {
    return `artifacts/${taskId}/cover.svg`;
  }
  if (requestsSlideOrDeck(query)) return `artifacts/${taskId}/slides.md`;
  if (requestsSpreadsheet(query)) return `artifacts/${taskId}/sheet.md`;
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
  if (
    requestsImageArtifact(query) &&
    !requestsSlideOrDeck(query) &&
    !requestsSpreadsheet(query) &&
    path.endsWith(".svg")
  ) {
    return { path, content: extractSvg(finalAnswer) ?? fallbackCoverSvg(query) };
  }
  // Markdown / 降级 Office：首行标明无法生成原生二进制。
  const needsBanner =
    path.endsWith(".md") &&
    (isOfficeExtensionPath(
      query.match(
        /(?:^|[\s"'`（(])([A-Za-z0-9_.-]+(?:[\\/][A-Za-z0-9_.-]+)*\.(?:docx|pptx|xlsx))(?=$|[\s"'`，。）、)])/iu,
      )?.[1] ?? "",
    ) ||
      requestsSlideOrDeck(query) ||
      requestsSpreadsheet(query) ||
      /(?:文档|报告|docx|pptx|xlsx)/iu.test(query));
  const body = finalAnswer.trim().length > 0 ? finalAnswer : "（模型未产出正文，已落盘占位。）";
  return {
    path,
    content: needsBanner ? `${OFFICE_DOWNGRADE_BANNER}\n${body}` : body,
  };
};
