/**
 * 极简 frontmatter 解析（只覆盖 SKILL.md 需要的子集，避免引入 YAML 依赖）。
 *
 * 支持：
 *  - `key: scalar`（可带引号）
 *  - `key: [a, b, c]`（内联数组）
 *  - 块状列表：
 *      key:
 *        - a
 *        - b
 *  - `#` 起头的整行注释
 *
 * 不支持嵌套对象 / 多行标量等；解析失败时对该字段宽松跳过。
 */
export interface Frontmatter {
  data: Record<string, string | string[]>;
  body: string;
}

const stripQuotes = (s: string): string => {
  const t = s.trim();
  if (
    t.length >= 2 &&
    ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")))
  ) {
    return t.slice(1, -1);
  }
  return t;
};

const parseInlineArray = (raw: string): string[] =>
  raw
    .slice(1, -1)
    .split(",")
    .map((x) => stripQuotes(x))
    .filter((x) => x.length > 0);

export const parseFrontmatter = (raw: string): Frontmatter => {
  const text = raw.replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
  // 必须以 --- 开头才视为有 frontmatter；否则整体当 body。
  if (!text.startsWith("---\n")) {
    return { data: {}, body: text.trim() };
  }
  const end = text.indexOf("\n---", 4);
  if (end === -1) {
    return { data: {}, body: text.trim() };
  }
  const fmBlock = text.slice(4, end);
  const afterIdx = text.indexOf("\n", end + 1);
  const body = (afterIdx === -1 ? "" : text.slice(afterIdx + 1)).trim();

  const data: Record<string, string | string[]> = {};
  const lines = fmBlock.split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i] ?? "";
    i++;
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith("#")) continue;
    const colon = line.indexOf(":");
    if (colon === -1) continue;
    const key = line.slice(0, colon).trim();
    if (key.length === 0) continue;
    const rest = line.slice(colon + 1).trim();

    if (rest.length === 0) {
      // 可能是块状列表：收集后续 "  - x" 行。
      const items: string[] = [];
      while (i < lines.length) {
        const next = lines[i] ?? "";
        const m = next.match(/^\s*-\s+(.*)$/);
        if (!m) break;
        items.push(stripQuotes(m[1] ?? ""));
        i++;
      }
      data[key] = items;
      continue;
    }
    if (rest.startsWith("[") && rest.endsWith("]")) {
      data[key] = parseInlineArray(rest);
      continue;
    }
    data[key] = stripQuotes(rest);
  }
  return { data, body };
};
