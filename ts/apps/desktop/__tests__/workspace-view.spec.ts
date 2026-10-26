import { mkdirSync, writeFileSync } from "node:fs";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { listWorkspaceEntries, readWorkspaceBytes } from "../src/main/workspace-fs.js";
import {
  artifactsFromAnswer,
  buildWorkspaceTree,
  extractWorkspaceArtifactPaths,
  guessedRasterFromSvgPath,
  preferredPreviewPath,
  previewKindForPath,
  rasterSiblingFromSvg,
  requestsArtifactPreview,
  splitWorkspacePathMarkup,
} from "../src/shared/workspace-view.js";

const roots: string[] = [];
afterAll(() => {
  for (const root of roots) rmSync(root, { recursive: true, force: true });
});

describe("workspace-view helpers", () => {
  it("extracts backticked workspace artifact paths from answers", () => {
    const text =
      "已写入工作区：`artifacts/3128030c-484f-4486-8b01-479af23b5533/cover.png`。";
    expect(extractWorkspaceArtifactPaths(text)).toEqual([
      "artifacts/3128030c-484f-4486-8b01-479af23b5533/cover.png",
    ]);
    expect(artifactsFromAnswer(text)[0]).toMatchObject({ kind: "image" });
    expect(previewKindForPath("notes/a.md")).toBe("text");
  });

  it("splits markup so chat can turn paths into buttons", () => {
    const parts = splitWorkspacePathMarkup("见 `artifacts/t/cover.png` 与说明");
    expect(parts).toEqual([
      { type: "text", value: "见 " },
      { type: "path", value: "artifacts/t/cover.png" },
      { type: "text", value: " 与说明" },
    ]);
  });

  it("prefers a raster file over an SVG preview stub", () => {
    expect(
      preferredPreviewPath([
        "other/photo.png",
        "artifacts/t/cover_preview.svg",
        "artifacts/t/cover.png",
      ]),
    ).toBe("artifacts/t/cover.png");
    expect(requestsArtifactPreview("预览刚才生图的产物")).toBe(true);
    expect(
      rasterSiblingFromSvg(
        '<svg><image href="cover.png" width="1280" height="720"/></svg>',
        "artifacts/t/cover_preview.svg",
      ),
    ).toBe("artifacts/t/cover.png");
    expect(guessedRasterFromSvgPath("artifacts/t/cover_preview.svg")).toBe(
      "artifacts/t/cover.png",
    );
  });

  it("builds a nested tree from a flat listing", () => {
    const tree = buildWorkspaceTree([
      { path: "artifacts", name: "artifacts", type: "dir" },
      { path: "artifacts/t", name: "t", type: "dir" },
      { path: "artifacts/t/cover.png", name: "cover.png", type: "file", bytes: 12 },
    ]);
    expect(tree).toHaveLength(1);
    expect(tree[0]?.children?.[0]?.children?.[0]?.path).toBe("artifacts/t/cover.png");
  });
});

describe("workspace-fs", () => {
  it("lists and reads a PNG as base64 without leaving the sandbox", async () => {
    const root = mkdtempSync(join(tmpdir(), "openintj-wsfs-"));
    roots.push(root);
    mkdirSync(join(root, "artifacts", "t"), { recursive: true });
    writeFileSync(join(root, "artifacts", "t", "cover.png"), "PNGDATA");
    const entries = await listWorkspaceEntries(root);
    expect(entries.map((entry) => entry.path)).toContain("artifacts/t/cover.png");
    const bytes = await readWorkspaceBytes(root, "artifacts/t/cover.png");
    expect(bytes.mimeType).toBe("image/png");
    expect(Buffer.from(bytes.base64, "base64").toString()).toBe("PNGDATA");
    await expect(readWorkspaceBytes(root, "../secret.png")).rejects.toThrow(/越界/);
  });
});
