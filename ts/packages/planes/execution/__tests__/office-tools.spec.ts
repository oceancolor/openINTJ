import { mkdtempSync, rmSync } from "node:fs";
import { readFile as fsReadFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ExcelJS from "exceljs";
import JSZip from "jszip";
import { afterAll, describe, expect, it } from "vitest";
import { createOfficeTools } from "../src/office-tools.js";
import { ToolHub } from "../src/tool-hub.js";

const roots: string[] = [];
const makeRoot = (): string => {
  const d = mkdtempSync(join(tmpdir(), "openintj-office-"));
  roots.push(d);
  return d;
};
afterAll(() => {
  for (const d of roots) {
    try {
      rmSync(d, { recursive: true, force: true });
    } catch {
      // ignore
    }
  }
});

const ZIP_MAGIC = Buffer.from([0x50, 0x4b, 0x03, 0x04]);

describe("createOfficeTools", () => {
  it("create_docx：魔数 PK + Content_Types", async () => {
    const root = makeRoot();
    const tools = createOfficeTools({ root });
    const w = (await tools.createDocx({
      path: "docs/q",
      title: "季度报告",
      body: ["第一段", "第二段"],
    })) as { path: string; bytesWritten: number; ok: boolean };
    expect(w.path).toBe("docs/q.docx");
    expect(w.ok).toBe(true);
    const buf = await fsReadFile(join(root, "docs", "q.docx"));
    expect(Buffer.compare(buf.subarray(0, 4), ZIP_MAGIC)).toBe(0);
    const zip = await JSZip.loadAsync(buf);
    expect(zip.file("[Content_Types].xml")).toBeTruthy();
  });

  it("create_pptx：魔数 PK + Content_Types", async () => {
    const root = makeRoot();
    const tools = createOfficeTools({ root });
    const w = (await tools.createPptx({
      path: "deck.pptx",
      slides: [{ title: "封面", bullets: ["要点 A", "要点 B"] }],
    })) as { path: string };
    expect(w.path).toBe("deck.pptx");
    const buf = await fsReadFile(join(root, "deck.pptx"));
    expect(Buffer.compare(buf.subarray(0, 4), ZIP_MAGIC)).toBe(0);
    const zip = await JSZip.loadAsync(buf);
    expect(zip.file("[Content_Types].xml")).toBeTruthy();
  });

  it("create_xlsx：魔数 + sheet/公式读回一致", async () => {
    const root = makeRoot();
    const tools = createOfficeTools({ root });
    await tools.createXlsx({
      path: "sales",
      sheets: [
        {
          name: "Sales",
          rows: [
            ["Item", "Qty", "Price", "Total"],
            ["Widget", 2, 10, { formula: "B2*C2", result: 20 }],
          ],
        },
      ],
    });
    const buf = await fsReadFile(join(root, "sales.xlsx"));
    expect(Buffer.compare(buf.subarray(0, 4), ZIP_MAGIC)).toBe(0);
    const zip = await JSZip.loadAsync(buf);
    expect(zip.file("[Content_Types].xml")).toBeTruthy();

    const wb = new ExcelJS.Workbook();
    await wb.xlsx.load(buf);
    const ws = wb.getWorksheet("Sales");
    expect(ws).toBeTruthy();
    expect(ws!.getCell("A1").value).toBe("Item");
    expect(ws!.getCell("B2").value).toBe(2);
    const total = ws!.getCell("D2").value;
    expect(total).toMatchObject({ formula: "B2*C2" });
  });

  it("路径越界被拒绝", async () => {
    const tools = createOfficeTools({ root: makeRoot() });
    await expect(
      tools.createDocx({ path: "../evil.docx", title: "x", body: "y" }),
    ).rejects.toThrow(/越界/);
  });
});

describe("ToolHub.registerOfficeTools", () => {
  it("P1-3: has(create_docx|pptx|xlsx) 为 true", () => {
    const hub = new ToolHub();
    const tools = createOfficeTools({ root: makeRoot() });
    hub.registerOfficeTools(tools);
    expect(hub.has("create_docx")).toBe(true);
    expect(hub.has("create_pptx")).toBe(true);
    expect(hub.has("create_xlsx")).toBe(true);
  });
});
