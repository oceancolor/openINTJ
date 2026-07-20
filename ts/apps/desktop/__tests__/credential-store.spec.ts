import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createCredentialStore } from "../src/main/credential-store.js";

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("CredentialStore", () => {
  it("persists only encrypted API keys", () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-credentials-"));
    dirs.push(dir);
    const file = join(dir, "credentials.json");
    const crypto = {
      isEncryptionAvailable: () => true,
      encryptString: (value: string) => Buffer.from(`encrypted:${value}`, "utf8"),
      decryptString: (value: Buffer) => value.toString("utf8").replace(/^encrypted:/, ""),
    };
    const store = createCredentialStore(file, crypto);
    store.set("profile.kimi", "secret-key");

    expect(store.has("profile.kimi")).toBe(true);
    expect(store.get("profile.kimi")).toBe("secret-key");
    expect(readFileSync(file, "utf8")).not.toContain("secret-key");
    expect(store.delete("profile.kimi")).toBe(true);
    expect(store.has("profile.kimi")).toBe(false);
  });

  it("fails closed when secure encryption is unavailable", () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-credentials-"));
    dirs.push(dir);
    const store = createCredentialStore(join(dir, "credentials.json"), {
      isEncryptionAvailable: () => false,
      encryptString: () => Buffer.alloc(0),
      decryptString: () => "",
    });
    expect(() => store.set("profile.glm", "key")).toThrow("安全存储不可用");
  });
});
