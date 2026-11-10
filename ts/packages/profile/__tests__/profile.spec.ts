import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { JsonProfileStore, ProfileService } from "../src/index.js";

describe("ProfileService F11 assertion ledger (T22/T23)", () => {
  it("T22 恢复矩阵：多层断言撤销逐层回退，过期层不恢复", () => {
    const profile = new ProfileService();
    // P1=A（dormant），P2=B（import）→ 最新者 B 胜
    profile.upsert({ section: "preferences", key: "theme", value: "A", source: "dormant" });
    const p2 = profile.upsert({ section: "preferences", key: "theme", value: "B", source: "import" })!;
    // import 层在 dormant 之上需要绕过 user 守卫——两者都非 user，最新 appliedSeq 胜
    expect(profile.get("preferences", "theme")?.value).toBe("B");
    // 撤 P2 → 回到 A
    expect(profile.revokeAssertion(p2.id)).toBe(true);
    expect(profile.get("preferences", "theme")?.value).toBe("A");
    // 撤 P1 → 无值
    const p1 = profile.listAssertions("preferences", "theme").find((e) => e.value === "A")!;
    profile.revokeAssertion(p1.id);
    expect(profile.get("preferences", "theme")).toBeUndefined();
    // 审计层仍在（撤销不删历史）
    expect(profile.listAssertions("preferences", "theme")).toHaveLength(2);
  });

  it("T22b 用户层不受 proposal 撤销影响；过期推断不复活", () => {
    const profile = new ProfileService();
    profile.upsert({
      section: "preferences",
      key: "font",
      value: "旧推断",
      source: "dormant",
      validUntil: Date.now() - 1000,
    });
    const proposal = profile.upsert({
      section: "preferences",
      key: "font",
      value: "提案值",
      source: "import",
    })!;
    profile.upsert({ section: "preferences", key: "font", value: "用户值" });
    expect(profile.get("preferences", "font")?.value).toBe("用户值");
    profile.revokeAssertion(proposal.id);
    expect(profile.get("preferences", "font")?.value).toBe("用户值");
    // 移除用户层（restore 模式）：过期推断不得复活 → 无值
    profile.removeUserOverride("preferences", "font", { mode: "restore" });
    expect(profile.get("preferences", "font")).toBeUndefined();
  });

  it("T23 dismissInferredKey 持久抑制；removeUserOverride 默认防复活", () => {
    const profile = new ProfileService();
    profile.importPersona({ preferences: { tea: "绿茶" } });
    profile.upsert({ section: "preferences", key: "tea", value: "红茶" });
    // 默认 suppress：撤用户层后推断不回来，且再导入也被抑制
    profile.removeUserOverride("preferences", "tea");
    expect(profile.get("preferences", "tea")).toBeUndefined();
    profile.importPersona({ preferences: { tea: "绿茶" } });
    expect(profile.get("preferences", "tea")).toBeUndefined();
    // 用户仍可显式重设
    profile.upsert({ section: "preferences", key: "tea", value: "乌龙" });
    expect(profile.get("preferences", "tea")?.value).toBe("乌龙");
  });

  it("T23b 导出/导入携带撤销与 suppression，导入端不回潮", () => {
    const profile = new ProfileService();
    const layer = profile.upsert({
      section: "preferences",
      key: "editor",
      value: "Vim",
      source: "dormant",
    })!;
    profile.revokeAssertion(layer.id);
    profile.dismissInferredKey("preferences", "editor");
    const other = new ProfileService();
    other.importJSON(profile.exportJSON());
    expect(other.get("preferences", "editor")).toBeUndefined();
    other.importPersona({ preferences: { editor: "Vim" } });
    expect(other.get("preferences", "editor")).toBeUndefined();
  });
});

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
