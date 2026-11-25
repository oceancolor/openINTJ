import { describe, expect, it } from "vitest";
import {
  CAPABILITY_HOST_MATRIX,
  type CapabilityId,
  type HostId,
  capabilityStatus,
} from "../src/capability-matrix.js";

describe("P1-8 capability × host matrix", () => {
  it("covers required capabilities and hosts", () => {
    const caps = Object.keys(CAPABILITY_HOST_MATRIX) as CapabilityId[];
    expect(caps).toEqual(
      expect.arrayContaining([
        "office_tools",
        "image_generation",
        "shadow_memory",
        "mcp_connectors",
        "web_search",
      ]),
    );
    for (const cap of caps) {
      for (const host of ["cli", "server", "desktop"] as HostId[]) {
        expect(["available", "opt_in", "absent", "partial"]).toContain(
          capabilityStatus(cap, host),
        );
      }
    }
  });

  it("office_tools available on all three hosts after P1-3", () => {
    expect(capabilityStatus("office_tools", "cli")).toBe("available");
    expect(capabilityStatus("office_tools", "server")).toBe("available");
    expect(capabilityStatus("office_tools", "desktop")).toBe("available");
  });

  it("workspace_preview is desktop-only", () => {
    expect(capabilityStatus("workspace_preview", "desktop")).toBe("available");
    expect(capabilityStatus("workspace_preview", "cli")).toBe("absent");
    expect(capabilityStatus("workspace_preview", "server")).toBe("absent");
  });
});
