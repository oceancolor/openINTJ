/**
 * Safe Markdown subset parser for workspace previews. Produces an AST that
 * the renderer maps to React elements — raw HTML in the source is always
 * treated as literal text (never injected), and link hrefs are restricted
 * to http(s)/mailto/anchors/relative paths (`javascript:` etc. degrade to
 * plain text). Supported: ATX headings, paragraphs, fenced code, flat
 * lists, blockquotes, hr, pipe tables, bold/italic/inline code/links/images.
 */

export type MdInline =
  | { t: "text"; text: string }
  | { t: "code"; text: string }
  | { t: "bold"; children: MdInline[] }
  | { t: "italic"; children: MdInline[] }
  | { t: "link"; href: string; children: MdInline[] }
  | { t: "image"; src: string; alt: string };

export type MdBlock =
  | { t: "heading"; level: number; children: MdInline[] }
  | { t: "paragraph"; children: MdInline[] }
  | { t: "code"; lang?: string; text: string }
  | { t: "list"; ordered: boolean; items: MdInline[][] }
  | { t: "quote"; blocks: MdBlock[] }
  | { t: "hr" }
  | { t: "table"; header: MdInline[][]; rows: MdInline[][][] };

const SAFE_HREF = /^(https?:\/\/|mailto:|#|\/|\.\/|\.\.\/)/i;

export const parseInline = (input: string): MdInline[] => {
  const out: MdInline[] = [];
  let rest = input;
  const pushText = (text: string): void => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last?.t === "text") last.text += text;
    else out.push({ t: "text", text });
  };
  while (rest.length > 0) {
    const code = /^`([^`]+)`/.exec(rest);
    if (code) {
      out.push({ t: "code", text: code[1]! });
      rest = rest.slice(code[0].length);
      continue;
    }
    const image = /^!\[([^\]]*)\]\(([^)\s]+)\)/.exec(rest);
    if (image) {
      out.push({ t: "image", src: image[2]!, alt: image[1]! });
      rest = rest.slice(image[0].length);
      continue;
    }
    const link = /^\[([^\]]+)\]\(([^)\s]+)\)/.exec(rest);
    if (link) {
      const href = link[2]!;
      if (SAFE_HREF.test(href)) {
        out.push({ t: "link", href, children: parseInline(link[1]!) });
      } else {
        pushText(link[1]!);
      }
      rest = rest.slice(link[0].length);
      continue;
    }
    const bold = /^\*\*([^*]+)\*\*/.exec(rest) ?? /^__([^_]+)__/.exec(rest);
    if (bold) {
      out.push({ t: "bold", children: parseInline(bold[1]!) });
      rest = rest.slice(bold[0].length);
      continue;
    }
    const italic = /^\*([^*]+)\*/.exec(rest) ?? /^_([^_]+)_/.exec(rest);
    if (italic) {
      out.push({ t: "italic", children: parseInline(italic[1]!) });
      rest = rest.slice(italic[0].length);
      continue;
    }
    pushText(rest[0]!);
    rest = rest.slice(1);
  }
  return out;
};

const splitRow = (line: string): string[] =>
  line
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((cell) => cell.trim());

const isTableSeparator = (line: string): boolean =>
  /^\|?[\s:|-]+\|?$/.test(line.trim()) && line.includes("-");

export const parseMarkdown = (source: string): MdBlock[] => {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: MdBlock[] = [];
  let i = 0;
  const isBlank = (line: string | undefined): boolean => !line || line.trim() === "";

  while (i < lines.length) {
    const line = lines[i]!;
    if (isBlank(line)) {
      i += 1;
      continue;
    }
    const fence = /^```(\w*)\s*$/.exec(line.trim());
    if (fence) {
      const buf: string[] = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i]!.trim())) {
        buf.push(lines[i]!);
        i += 1;
      }
      i += 1; // closing fence
      blocks.push({ t: "code", ...(fence[1] ? { lang: fence[1] } : {}), text: buf.join("\n") });
      continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line.trim());
    if (heading) {
      blocks.push({
        t: "heading",
        level: heading[1]!.length,
        children: parseInline(heading[2]!.trim()),
      });
      i += 1;
      continue;
    }
    if (/^\s*(---+|\*\*\*+|___+)\s*$/.test(line)) {
      blocks.push({ t: "hr" });
      i += 1;
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1]!)) {
      const header = splitRow(line).map(parseInline);
      i += 2;
      const rows: MdInline[][][] = [];
      while (i < lines.length && !isBlank(lines[i]) && lines[i]!.includes("|")) {
        rows.push(splitRow(lines[i]!).map(parseInline));
        i += 1;
      }
      blocks.push({ t: "table", header, rows });
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i]!)) {
        buf.push(lines[i]!.replace(/^\s*>\s?/, ""));
        i += 1;
      }
      blocks.push({ t: "quote", blocks: parseMarkdown(buf.join("\n")) });
      continue;
    }
    const listItem = /^(\s*)([-*+]|\d+[.)])\s+(.+)$/.exec(line);
    if (listItem) {
      const ordered = /\d/.test(listItem[2]!);
      const items: MdInline[][] = [];
      while (i < lines.length) {
        const m = /^(\s*)([-*+]|\d+[.)])\s+(.+)$/.exec(lines[i]!);
        if (!m) break;
        items.push(parseInline(m[3]!.trim()));
        i += 1;
      }
      blocks.push({ t: "list", ordered, items });
      continue;
    }
    const buf: string[] = [];
    while (i < lines.length && !isBlank(lines[i])) {
      const next = lines[i]!;
      if (
        buf.length > 0 &&
        (/^```/.test(next.trim()) ||
          /^#{1,6}\s/.test(next.trim()) ||
          /^\s*>\s?/.test(next) ||
          /^(\s*)([-*+]|\d+[.)])\s+/.test(next) ||
          (next.includes("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1]!)))
      ) {
        break;
      }
      buf.push(next.trim());
      i += 1;
    }
    blocks.push({ t: "paragraph", children: parseInline(buf.join(" ")) });
  }
  return blocks;
};
