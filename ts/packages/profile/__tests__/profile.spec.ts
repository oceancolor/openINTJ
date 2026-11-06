import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { JsonProfileStore, ProfileService } from "../src/index.js";

describe("ProfileService (RFC-014)", () => {
  it("user entries win over dormant writes for the same key", () => {
    const profile = new ProfileService();
    profile.upsert({ section: "preferences", key: "editor", value: "VS Code" });
    const overwritten = profile.upsert({
      section: "preferences",
      key: "editor",
      value: "Vim",
      source: "dormant",
    });
    expect(overwritten).toBeUndefined();
    expect(profile.get("preferences", "editor")?.value).toBe("VS Code");
    // Dormant can still fill keys the user never set.
    profile.upsert({ section: "preferences", key: "tea", value: "绿茶", source: "dormant" });
    expect(profile.get("preferences", "tea")?.source).toBe("dormant");
  });

  it("expired entries are hidden from list and prompt", () => {
    const profile = new ProfileService();
    profile.upsert({ section: "goals", key: "now", value: "旧目标", validUntil: Date.now() - 1 });
    profile.upsert({ section: "goals", key: "next", value: "新目标" });
    expect(profile.list().map((e) => e.key)).toEqual(["next"]);
    expect(profile.renderPromptSection()).toContain("新目标");
    expect(profile.renderPromptSection()).not.toContain("旧目标");
  });

  it("importPersona is idempotent and renders grouped sections", () => {
    const profile = new ProfileService();
    const persona = { preferences: { tea: "绿茶" }, phrases: { greeting: "搞定" } };
    profile.importPersona(persona);
    profile.importPersona(persona);
    expect(profile.list()).toHaveLength(2);
    const rendered = profile.renderPromptSection();
    expect(rendered).toContain("[个人档案]");
    expect(rendered).toContain("偏好：tea=绿茶");
  });

  it("deleting a dormant entry dismisses it from future imports", () => {
    const profile = new ProfileService();
    const persona = { preferences: { tea: "绿茶" } };
    profile.importPersona(persona);
    const entry = profile.get("preferences", "tea")!;
    expect(profile.remove(entry.id)).toBe(true);
    profile.importPersona(persona);
    expect(profile.get("preferences", "tea")).toBeUndefined();
    expect(profile.renderPromptSection()).toBe("");
    // The user can still set the same key explicitly afterwards.
    profile.upsert({ section: "preferences", key: "tea", value: "红茶" });
    expect(profile.get("preferences", "tea")?.value).toBe("红茶");
  });

  it("empty profile renders empty and export/import round-trips", () => {
    const profile = new ProfileService();
    expect(profile.renderPromptSection()).toBe("");
    profile.upsert({ section: "identity", key: "name", value: "Ben" });
    const other = new ProfileService();
    expect(other.importJSON(profile.exportJSON())).toBe(1);
    expect(other.get("identity", "name")?.value).toBe("Ben");
  });

  it("JsonProfileStore persists across instances", () => {
    const dir = mkdtempSync(join(tmpdir(), "openintj-profile-"));
    try {
      const file = join(dir, "profile.json");
      const first = new ProfileService(new JsonProfileStore(file));
      first.upsert({ section: "projects", key: "main", value: "openINTJ" });
      const second = new ProfileService(new JsonProfileStore(file));
      expect(second.get("projects", "main")?.value).toBe("openINTJ");
      expect(second.remove(second.get("projects", "main")!.id)).toBe(true);
      const third = new ProfileService(new JsonProfileStore(file));
      expect(third.list()).toEqual([]);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
