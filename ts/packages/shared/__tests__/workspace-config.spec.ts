import { describe, expect, it } from "vitest";
import { isCommandExecutionAvailable, resolveWorkspaceConfig } from "../src/workspace-config.js";

describe("isCommandExecutionAvailable", () => {
  it("requires both enableCommands and a non-empty allowlist", () => {
    expect(
      isCommandExecutionAvailable({ enableCommands: false, allowedCommands: ["python"] }),
    ).toBe(false);
    expect(isCommandExecutionAvailable({ enableCommands: true, allowedCommands: [] })).toBe(false);
    expect(
      isCommandExecutionAvailable({ enableCommands: true, allowedCommands: ["python"] }),
    ).toBe(true);
  });
});

describe("resolveWorkspaceConfig", () => {
  it("defaults commands off with an empty allowlist", () => {
    const resolved = resolveWorkspaceConfig({}, "/work");
    expect(resolved).toMatchObject({
      root: "/work",
      enableCommands: false,
      allowedCommands: [],
    });
    expect(isCommandExecutionAvailable(resolved)).toBe(false);
  });
});
