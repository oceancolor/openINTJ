import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { createDevTools } from "../src/dev-tools.js";
import { ToolHub } from "../src/tool-hub.js";

const roots: string[] = [];
const makeRoot = (): string => {
  const d = mkdtempSync(join(tmpdir(), "openintj-dev-"));
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

describe("createDevTools", () => {
  it("http_request GET 返回 status/body", async () => {
    const tools = createDevTools({ root: makeRoot() });
    const r = (await tools.httpRequest({
      url: "https://example.com",
      method: "GET",
    })) as { status: number; body: string; ok: boolean };
    expect(r.status).toBe(200);
    expect(r.body.toLowerCase()).toContain("example");
    expect(r.ok).toBe(true);
  });

  it("http_request 拒绝非 http(s)", async () => {
    const tools = createDevTools({ root: makeRoot() });
    await expect(tools.httpRequest({ url: "file:///etc/passwd" })).rejects.toThrow(/http/);
  });

  it("start_server 拒绝非白名单 bin", async () => {
    const tools = createDevTools({ root: makeRoot() });
    await expect(tools.startServer({ command: "curl http://x" })).rejects.toThrow(/拒绝/);
  });
});

describe("ToolHub.registerDevTools", () => {
  it("P1-5: has run_tests/typecheck/http_request/start_server", () => {
    const hub = new ToolHub();
    const root = makeRoot();
    writeFileSync(join(root, "package.json"), "{}");
    hub.registerDevTools(createDevTools({ root }));
    expect(hub.has("run_tests")).toBe(true);
    expect(hub.has("typecheck")).toBe(true);
    expect(hub.has("http_request")).toBe(true);
    expect(hub.has("start_server")).toBe(true);
  });
});
