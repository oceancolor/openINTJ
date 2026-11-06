import { randomUUID } from "node:crypto";
import { InMemoryProfileStore, type ProfileStore } from "./store.js";
import {
  PROFILE_SECTIONS,
  type PersonaLike,
  type ProfileEntry,
  type ProfileEntryInput,
  type ProfileSection,
} from "./types.js";

const SECTION_LABELS: Record<ProfileSection, string> = {
  identity: "身份",
  preferences: "偏好",
  goals: "目标",
  projects: "项目",
  relationships: "关系",
  constraints: "约束",
  habits: "习惯",
  notes: "备注",
};

const stringifyValue = (value: unknown): string =>
  typeof value === "string" ? value : JSON.stringify(value);

/**
 * RFC-014 memory control plane. One live entry per (section, key);
 * explicit user entries are never overwritten by dormant inference.
 */
export class ProfileService {
  readonly store: ProfileStore;

  constructor(store: ProfileStore = new InMemoryProfileStore()) {
    this.store = store;
  }

  private live(now = Date.now()): ProfileEntry[] {
    return this.store
      .list()
      .filter((entry) => !entry.validUntil || entry.validUntil > now)
      .sort(
        (a, b) =>
          PROFILE_SECTIONS.indexOf(a.section) - PROFILE_SECTIONS.indexOf(b.section) ||
          a.key.localeCompare(b.key),
      );
  }

  list(filter?: { section?: ProfileSection }): readonly ProfileEntry[] {
    const entries = this.live();
    return filter?.section ? entries.filter((e) => e.section === filter.section) : entries;
  }

  get(section: ProfileSection, key: string): ProfileEntry | undefined {
    return this.live().find((e) => e.section === section && e.key === key);
  }

  /**
   * Insert or replace the (section, key) entry. Returns undefined when a
   * dormant/import write tries to overwrite an explicit user entry.
   */
  upsert(input: ProfileEntryInput): ProfileEntry | undefined {
    const source = input.source ?? "user";
    const existing = this.store
      .list()
      .find((e) => e.section === input.section && e.key === input.key);
    if (existing && existing.source === "user" && source !== "user") return undefined;
    const now = Date.now();
    const entry: ProfileEntry = {
      id: existing?.id ?? randomUUID(),
      section: input.section,
      key: input.key,
      value: input.value,
      source,
      ...(input.provenance ? { provenance: input.provenance } : {}),
      ...(input.validUntil ? { validUntil: input.validUntil } : {}),
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    };
    this.store.upsert(entry);
    return entry;
  }

  remove(id: string): boolean {
    const existing = this.store.list().find((e) => e.id === id);
    if (!existing) return false;
    // Deleting an inferred entry is a durable user decision: never re-import it.
    if (existing.source === "dormant") {
      this.store.dismiss(`${existing.section}:${existing.key}`);
    }
    return this.store.remove(id);
  }

  removeByKey(section: ProfileSection, key: string): boolean {
    const existing = this.store.list().find((e) => e.section === section && e.key === key);
    return existing ? this.remove(existing.id) : false;
  }

  /** Idempotently import Dormant persona fields as `dormant` entries. */
  importPersona(persona: PersonaLike): void {
    const dismissed = new Set(this.store.listDismissed());
    const put = (section: ProfileSection, key: string, value: unknown) => {
      if (value === undefined || value === null || value === "") return;
      if (dismissed.has(`${section}:${key}`)) return;
      this.upsert({
        section,
        key,
        value: stringifyValue(value),
        source: "dormant",
        provenance: "dormant-persona",
      });
    };
    for (const [key, value] of Object.entries(persona.preferences ?? {})) {
      put("preferences", key, value);
    }
    for (const [key, value] of Object.entries(persona.phrases ?? {})) put("habits", key, value);
    for (const [key, value] of Object.entries(persona.habits ?? {})) put("habits", key, value);
    for (const [key, value] of Object.entries(persona.context ?? {})) put("notes", key, value);
  }

  /** Render for the RFC-011 user-persona prompt section. Empty profile → "". */
  renderPromptSection(): string {
    const entries = this.live();
    if (entries.length === 0) return "";
    const lines = ["[个人档案]（用户可编辑的结构化事实，优先于推断）"];
    for (const section of PROFILE_SECTIONS) {
      const group = entries.filter((e) => e.section === section);
      if (group.length === 0) continue;
      lines.push(
        `${SECTION_LABELS[section]}：${group.map((e) => `${e.key}=${e.value}`).join("；")}`,
      );
    }
    return lines.join("\n");
  }

  exportJSON(): string {
    return JSON.stringify({ version: 1, entries: this.live() }, null, 2);
  }

  importJSON(json: string): number {
    const parsed = JSON.parse(json) as { entries?: ProfileEntryInput[] };
    if (!Array.isArray(parsed.entries)) throw new Error("invalid profile export: missing entries");
    let count = 0;
    for (const entry of parsed.entries) {
      if (!PROFILE_SECTIONS.includes(entry.section) || !entry.key) continue;
      if (this.upsert({ ...entry, source: entry.source ?? "import" })) count += 1;
    }
    return count;
  }
}
