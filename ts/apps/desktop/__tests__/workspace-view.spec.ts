import { describe, expect, it } from "vitest";
import {
  type WorkspaceEntry,
  buildWorkspaceTree,
  filterWorkspaceFiles,
  isMarkdownPath,
} from "../src/shared/workspace-view.js";

const ENTRIES: WorkspaceEntry[] = [
  { path: "src/index.ts", name: "index.ts", type: "file" },
  { path: "src/identity.ts", name: "identity.ts", type: "file" },
  { path: "docs/identity-guide.md", name: "identity-guide.md", type: "file" },
  { path: "src", name: "src", type: "dir" },
  { path: "README.md", name: "README.md", type: "file" },
];

describe("filterWorkspaceFiles (quick-open)", () => {
  it("ranks name-prefix above name-substring above path-substring, files only", () => {
    // Both are name-prefix hits for "iden"; ties break by path order.
    const out = filterWorkspaceFiles(ENTRIES, "iden");
    expect(out.map((e) => e.path)).toEqual(["docs/identity-guide.md", "src/identity.ts"]);
    // A pure name-substring hit ranks below a name-prefix hit.
    const mixed = filterWorkspaceFiles(
      [
        { path: "x/reidentity.ts", name: "reidentity.ts", type: "file" },
        { path: "a/identity.ts", name: "identity.ts", type: "file" },
      ],
      "ident",
    );
    expect(mixed.map((e) => e.path)).toEqual(["a/identity.ts", "x/reidentity.ts"]);
    const byPath = filterWorkspaceFiles(ENTRIES, "docs");
    expect(byPath.map((e) => e.path)).toEqual(["docs/identity-guide.md"]);
    expect(filterWorkspaceFiles(ENTRIES, "src").every((e) => e.type === "file")).toBe(true);
  });

  it("empty query and no-match return empty", () => {
    expect(filterWorkspaceFiles(ENTRIES, "  ")).toEqual([]);
    expect(filterWorkspaceFiles(ENTRIES, "zzz")).toEqual([]);
  });

  it("respects the limit", () => {
    const many: WorkspaceEntry[] = Array.from({ length: 80 }, (_, i) => ({
      path: `f${i}.txt`,
      name: `f${i}.txt`,
      type: "file" as const,
    }));
    expect(filterWorkspaceFiles(many, "f")).toHaveLength(50);
    expect(filterWorkspaceFiles(many, "f", 5)).toHaveLength(5);
  });
});

describe("workspace-view basics", () => {
  it("detects markdown paths", () => {
    expect(isMarkdownPath("a/b.md")).toBe(true);
    expect(isMarkdownPath("a.markdown")).toBe(true);
    expect(isMarkdownPath("a.txt")).toBe(false);
  });

  it("buildWorkspaceTree nests files under dirs", () => {
    const tree = buildWorkspaceTree(ENTRIES);
    const src = tree.find((n) => n.path === "src");
    expect(src?.children?.map((c) => c.name).sort()).toEqual(["identity.ts", "index.ts"]);
  });
});
