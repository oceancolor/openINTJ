import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import type { ToolHandler } from "@openintj/core";
import { Document, Packer, Paragraph, TextRun, HeadingLevel } from "docx";
import ExcelJS from "exceljs";
import PptxGenJS from "pptxgenjs";
import { resolveInRoot } from "./workspace-tools.js";

export interface OfficeToolsOpts {
  root: string;
  /** 单文件写入最大字节（默认 20 MiB）。 */
  maxWriteBytes?: number;
}

export interface OfficeTools {
  createDocx: ToolHandler;
  createPptx: ToolHandler;
  createXlsx: ToolHandler;
}

const DEFAULT_MAX_BYTES = 20 * 1024 * 1024;

const ensureExt = (path: string, ext: ".docx" | ".pptx" | ".xlsx"): string => {
  const re = /\.[^.\\/]+$/u;
  if (path.toLowerCase().endsWith(ext)) return path;
  if (re.test(path)) return path.replace(re, ext);
  return `${path}${ext}`;
};

const asStringArray = (v: unknown): string[] => {
  if (typeof v === "string") return [v];
  if (Array.isArray(v)) {
    return v
      .map((x) => (typeof x === "string" ? x : x == null ? "" : String(x)))
      .filter((s) => s.length > 0);
  }
  return [];
};

/**
 * 原生 Office 生成工具（OOXML），经路径沙箱落盘。
 */
export const createOfficeTools = (opts: OfficeToolsOpts): OfficeTools => {
  const root = resolve(opts.root);
  const maxWriteBytes = opts.maxWriteBytes ?? DEFAULT_MAX_BYTES;

  const writeBuffer = async (relPath: string, buf: Buffer) => {
    if (buf.length > maxWriteBytes) {
      throw new Error(`写入内容过大：${buf.length} 字节 > 上限 ${maxWriteBytes}`);
    }
    const abs = resolveInRoot(root, relPath);
    await fs.mkdir(dirname(abs), { recursive: true });
    await fs.writeFile(abs, buf);
    return { path: relPath, bytesWritten: buf.length, ok: true };
  };

  const createDocx: ToolHandler = async (params) => {
    const rawPath = typeof params["path"] === "string" ? params["path"] : "";
    if (!rawPath) throw new Error("create_docx 需要非空 path 参数");
    const path = ensureExt(rawPath, ".docx");
    const title = typeof params["title"] === "string" ? params["title"] : "";
    const bodyParts = asStringArray(params["body"]);
    if (!title && bodyParts.length === 0) {
      throw new Error("create_docx 需要 title 或 body");
    }

    const children: Paragraph[] = [];
    if (title) {
      children.push(
        new Paragraph({
          text: title,
          heading: HeadingLevel.HEADING_1,
        }),
      );
    }
    for (const para of bodyParts) {
      children.push(new Paragraph({ children: [new TextRun(para)] }));
    }

    const doc = new Document({
      sections: [{ children }],
    });
    const buf = Buffer.from(await Packer.toBuffer(doc));
    return writeBuffer(path, buf);
  };

  const createPptx: ToolHandler = async (params) => {
    const rawPath = typeof params["path"] === "string" ? params["path"] : "";
    if (!rawPath) throw new Error("create_pptx 需要非空 path 参数");
    const path = ensureExt(rawPath, ".pptx");
    const slidesRaw = params["slides"];
    if (!Array.isArray(slidesRaw) || slidesRaw.length === 0) {
      throw new Error("create_pptx 需要非空 slides 数组");
    }

    const pptx = new PptxGenJS();
    for (const slideIn of slidesRaw) {
      if (slideIn === null || typeof slideIn !== "object") {
        throw new Error("create_pptx：slides 项必须是对象");
      }
      const s = slideIn as Record<string, unknown>;
      const title = typeof s["title"] === "string" ? s["title"] : "";
      const bullets = asStringArray(s["bullets"]);
      const slide = pptx.addSlide();
      if (title) {
        slide.addText(title, { x: 0.5, y: 0.4, w: 9, h: 0.8, fontSize: 28, bold: true });
      }
      if (bullets.length > 0) {
        slide.addText(
          bullets.map((b) => ({ text: b, options: { bullet: true } })),
          { x: 0.5, y: 1.4, w: 9, h: 4, fontSize: 18 },
        );
      }
    }

    const out = await pptx.write({ outputType: "nodebuffer" });
    const buf = Buffer.isBuffer(out) ? out : Buffer.from(out as ArrayBuffer);
    return writeBuffer(path, buf);
  };

  const createXlsx: ToolHandler = async (params) => {
    const rawPath = typeof params["path"] === "string" ? params["path"] : "";
    if (!rawPath) throw new Error("create_xlsx 需要非空 path 参数");
    const path = ensureExt(rawPath, ".xlsx");
    const sheetsRaw = params["sheets"];
    if (!Array.isArray(sheetsRaw) || sheetsRaw.length === 0) {
      throw new Error("create_xlsx 需要非空 sheets 数组");
    }

    const wb = new ExcelJS.Workbook();
    for (const sheetIn of sheetsRaw) {
      if (sheetIn === null || typeof sheetIn !== "object") {
        throw new Error("create_xlsx：sheets 项必须是对象");
      }
      const s = sheetIn as Record<string, unknown>;
      const name =
        typeof s["name"] === "string" && s["name"].trim()
          ? s["name"].trim().slice(0, 31)
          : "Sheet1";
      const ws = wb.addWorksheet(name);
      const rows = Array.isArray(s["rows"]) ? s["rows"] : [];
      for (const row of rows) {
        if (!Array.isArray(row)) {
          throw new Error("create_xlsx：rows 项必须是数组");
        }
        const values = row.map((cell) => {
          if (cell !== null && typeof cell === "object" && !Array.isArray(cell)) {
            const c = cell as Record<string, unknown>;
            if (typeof c["formula"] === "string") {
              return { formula: c["formula"], result: c["result"] };
            }
            if ("value" in c) return c["value"];
          }
          return cell;
        });
        ws.addRow(values);
      }
    }

    const buf = Buffer.from(await wb.xlsx.writeBuffer());
    return writeBuffer(path, buf);
  };

  return { createDocx, createPptx, createXlsx };
};
