import { describe, expect, it } from "vitest";
import { parseInline, parseMarkdown } from "../src/renderer/workspace/markdown.js";

describe("parseMarkdown blocks", () => {
  it("parses headings, paragraphs, lists, code fences, quotes, tables", () => {
    const blocks = parseMarkdown(
      [
        "# Title",
        "",
        "hello **world** and `code`",
        "",
        "- one",
        "- two",
        "",
        "```ts",
        "const a = 1;",
        "```",
        "",
        "> quoted *text*",
        "",
        "| a | b |",
        "|---|---|",
        "| 1 | 2 |",
      ].join("\n"),
    );
    expect(blocks.map((b) => b.t)).toEqual([
      "heading",
      "paragraph",
      "list",
      "code",
      "quote",
      "table",
    ]);
    const heading = blocks[0]!;
    expect(heading.t === "heading" && heading.level).toBe(1);
    const code = blocks[3]!;
    expect(code.t === "code" && code.text).toBe("const a = 1;");
    const table = blocks[5]!;
    expect(table.t === "table" && table.rows).toHaveLength(1);
  });

  it("treats raw HTML as literal text (no injection surface)", () => {
    const blocks = parseMarkdown("<script>alert(1)</script>");
    expect(blocks).toHaveLength(1);
    const para = blocks[0]!;
    expect(para.t).toBe("paragraph");
    if (para.t === "paragraph") {
      expect(para.children).toEqual([{ t: "text", text: "<script>alert(1)</script>" }]);
    }
  });
});

describe("parseInline safety", () => {
  it("keeps http links, drops javascript: hrefs to plain text", () => {
    const ok = parseInline("see [docs](https://example.com/x)");
    expect(ok.some((n) => n.t === "link")).toBe(true);
    const bad = parseInline("see [x](javascript:alert(1))");
    expect(bad.every((n) => n.t !== "link")).toBe(true);
    expect(bad.map((n) => (n.t === "text" ? n.text : "")).join("")).toContain("x");
  });

  it("does not mistake emails for handles and parses bold/code", () => {
    const nodes = parseInline("mail ben@example.com, **bold**, `x=1`");
    expect(nodes.some((n) => n.t === "bold")).toBe(true);
    expect(nodes.some((n) => n.t === "code")).toBe(true);
  });
});
