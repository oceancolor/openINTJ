import React from "react";
import { type MdBlock, type MdInline, parseMarkdown } from "./markdown.js";

const renderInline = (nodes: readonly MdInline[], keyPrefix: string): React.ReactNode =>
  nodes.map((node, idx) => {
    const key = `${keyPrefix}-${idx}`;
    switch (node.t) {
      case "text":
        return <React.Fragment key={key}>{node.text}</React.Fragment>;
      case "code":
        return (
          <code
            key={key}
            className="px-1 py-0.5 rounded bg-inset font-mono text-ink text-[11px]"
          >
            {node.text}
          </code>
        );
      case "bold":
        return (
          <strong key={key} className="text-ink font-semibold">
            {renderInline(node.children, key)}
          </strong>
        );
      case "italic":
        return <em key={key}>{renderInline(node.children, key)}</em>;
      case "link":
        return (
          <a
            key={key}
            href={node.href}
            target="_blank"
            rel="noreferrer"
            className="text-moss underline hover:text-mossdeep"
          >
            {renderInline(node.children, key)}
          </a>
        );
      case "image":
        return /^https?:\/\//i.test(node.src) || node.src.startsWith("data:") ? (
          <img key={key} src={node.src} alt={node.alt} className="max-w-full rounded" />
        ) : (
          <span key={key} className="text-muted">
            🖼 {node.alt || node.src}（相对路径图片，请从文件树打开）
          </span>
        );
    }
  });

const HEADING_CLASS: Record<number, string> = {
  1: "text-lg font-semibold text-ink mt-3 mb-1",
  2: "text-base font-semibold text-ink mt-3 mb-1",
  3: "text-sm font-semibold text-ink mt-2 mb-1",
  4: "text-xs font-semibold text-ink mt-2 mb-1",
  5: "text-xs font-semibold text-ink mt-2 mb-1",
  6: "text-xs font-semibold text-muted mt-2 mb-1",
};

const renderBlock = (block: MdBlock, key: string): React.ReactNode => {
  switch (block.t) {
    case "heading": {
      const Tag = `h${block.level}` as "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
      return (
        <Tag key={key} className={HEADING_CLASS[block.level]}>
          {renderInline(block.children, key)}
        </Tag>
      );
    }
    case "paragraph":
      return (
        <p key={key} className="my-1.5 leading-relaxed text-ink">
          {renderInline(block.children, key)}
        </p>
      );
    case "code":
      return (
        <pre
          key={key}
          className="my-2 overflow-x-auto rounded-xl bg-toolbg p-3 font-mono text-[11px] leading-5 text-[#d9dfd4]"
        >
          {block.lang ? <div className="mb-1 text-white/45">{block.lang}</div> : null}
          <code>{block.text}</code>
        </pre>
      );
    case "list": {
      const Tag = block.ordered ? "ol" : "ul";
      return (
        <Tag
          key={key}
          className={`my-1.5 pl-5 text-ink leading-relaxed ${block.ordered ? "list-decimal" : "list-disc"}`}
        >
          {block.items.map((item, idx) => (
            <li key={`${key}-${idx}`}>{renderInline(item, `${key}-${idx}`)}</li>
          ))}
        </Tag>
      );
    }
    case "quote":
      return (
        <blockquote key={key} className="my-2 border-l-2 border-[#cfd4ca] pl-3 text-muted">
          {block.blocks.map((inner, idx) => renderBlock(inner, `${key}-${idx}`))}
        </blockquote>
      );
    case "hr":
      return <hr key={key} className="my-3 border-line" />;
    case "table":
      return (
        <table key={key} className="my-2 border-collapse text-[11px] text-ink">
          <thead>
            <tr>
              {block.header.map((cell, idx) => (
                <th
                  key={`${key}-h-${idx}`}
                  className="border border-line bg-inset px-2 py-1 text-left"
                >
                  {renderInline(cell, `${key}-h-${idx}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, rIdx) => (
              <tr key={`${key}-r-${rIdx}`}>
                {row.map((cell, cIdx) => (
                  <td key={`${key}-r-${rIdx}-${cIdx}`} className="border border-line px-2 py-1">
                    {renderInline(cell, `${key}-r-${rIdx}-${cIdx}`)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
  }
};

/** Rendered Markdown preview (safe subset; raw HTML stays literal text). */
export const MarkdownView: React.FC<{ text: string }> = ({ text }) => {
  const blocks = React.useMemo(() => parseMarkdown(text), [text]);
  return (
    <div className="text-xs" data-testid="markdown-view">
      {blocks.map((block, idx) => renderBlock(block, `b-${idx}`))}
    </div>
  );
};
