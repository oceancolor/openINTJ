import { describe, expect, it } from "vitest";
import { canonicalToolName, canonicalToolNames } from "../src/index.js";

describe("canonicalToolName", () => {
  it("maps camelCase aliases to ToolHub names", () => {
    expect(canonicalToolName("readFile")).toBe("read_file");
    expect(canonicalToolName("writeFile")).toBe("write_file");
    expect(canonicalToolName("executeCommand")).toBe("execute_command");
    expect(canonicalToolName("search")).toBe("search");
  });

  it("keeps already-canonical and unknown names", () => {
    expect(canonicalToolName("read_file")).toBe("read_file");
    expect(canonicalToolName("custom_tool")).toBe("custom_tool");
  });
});

describe("canonicalToolNames", () => {
  it("dedupes aliases that collapse to the same name", () => {
    expect(canonicalToolNames(["readFile", "read_file", "search", "search"])).toEqual([
      "read_file",
      "search",
    ]);
  });
});
