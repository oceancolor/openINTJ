import { describe, expect, it } from "vitest";
import {
  bufferDirty,
  closeBuffer,
  nextActivePath,
  openBuffer,
  patchBuffer,
} from "../src/renderer/workspace/editor-buffers.js";
import {
  editorKindForPath,
  isEditablePath,
  monacoLanguageForPath,
} from "../src/shared/workspace-view.js";

describe("editor path helpers", () => {
  it("opens rasters as image preview and text/svg in the editor", () => {
    expect(editorKindForPath("artifacts/t/cover.png")).toBe("image");
    expect(editorKindForPath("artifacts/t/cover_preview.svg")).toBe("text");
    expect(editorKindForPath("src/agent.ts")).toBe("text");
    expect(isEditablePath("src/agent.ts")).toBe(true);
    expect(isEditablePath("artifacts/t/cover.png")).toBe(false);
    expect(monacoLanguageForPath("src/agent.ts")).toBe("typescript");
    expect(monacoLanguageForPath("notes/a.md")).toBe("markdown");
    expect(monacoLanguageForPath("notes/a.yml")).toBe("plaintext");
  });
});

describe("editor buffers", () => {
  it("opens a text buffer as loading and an image buffer as ready", () => {
    const text = openBuffer([], "notes/a.md");
    expect(text).toHaveLength(1);
    expect(text[0]).toMatchObject({ path: "notes/a.md", kind: "text", loading: true });
    expect(bufferDirty(text[0]!)).toBe(false);
    const withImage = openBuffer(text, "artifacts/t/cover.png");
    expect(withImage).toHaveLength(2);
    expect(withImage[1]).toMatchObject({ kind: "image", loading: false });
    expect(openBuffer(withImage, "notes/a.md")).toHaveLength(2);
  });

  it("tracks dirty state and focuses the neighbor after close", () => {
    let buffers = openBuffer([], "a.md");
    buffers = openBuffer(buffers, "b.md");
    buffers = patchBuffer(buffers, "a.md", {
      loading: false,
      content: "hello",
      savedContent: "",
    });
    expect(bufferDirty(buffers[0]!)).toBe(true);
    expect(nextActivePath(buffers, "a.md", "a.md")).toBe("b.md");
    expect(closeBuffer(buffers, "a.md").map((buffer) => buffer.path)).toEqual(["b.md"]);
  });
});
