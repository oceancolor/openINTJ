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
 * RFC-014 memory control plane + RFC-021 F11 assertion ledger.
 *
 * 每个 (section, key) 可有多层断言（不同来源/不同时间），投影裁决：
 * 先过滤 revoked/expired/被 dismiss 的推断层 → 用户显式层优先 →
 * 其余按 appliedSeq 最新者胜。撤销只撤对应来源层，下层有效断言
 * 按恢复矩阵重新生效；过期层永不恢复。
 */
export class ProfileService {
  readonly store: ProfileStore;

  constructor(store: ProfileStore = new InMemoryProfileStore()) {
    this.store = store;
  }

  private seqOf(entry: ProfileEntry): number {
    return entry.appliedSeq ?? entry.createdAt;
  }

  private nextSeq(): number {
    return this.store.list().reduce((max, e) => Math.max(max, this.seqOf(e)), 0) + 1;
  }

  /** 某 key 的全部断言层（含已撤销/过期，审计用）。 */
  listAssertions(section: ProfileSection, key: string): readonly ProfileEntry[] {
    return this.store
      .list()
      .filter((e) => e.section === section && e.key === key)
      .sort((a, b) => this.seqOf(a) - this.seqOf(b));
  }

  private project(now = Date.now()): ProfileEntry[] {
    const dismissed = new Set(this.store.listDismissed());
    const groups = new Map<string, ProfileEntry[]>();
    for (const entry of this.store.list()) {
      if (entry.revokedAt) continue;
      if (entry.validUntil && entry.validUntil <= now) continue;
      if (entry.source !== "user" && dismissed.has(`${entry.section}:${entry.key}`)) continue;
      const groupKey = `${entry.section}:${entry.key}`;
      const group = groups.get(groupKey) ?? [];
      group.push(entry);
      groups.set(groupKey, group);
    }
    const winners: ProfileEntry[] = [];
    for (const group of groups.values()) {
      const userLayers = group.filter((e) => e.source === "user");
      const pool = userLayers.length > 0 ? userLayers : group;
      winners.push(pool.reduce((a, b) => (this.seqOf(a) >= this.seqOf(b) ? a : b)));
    }
    return winners;
  }

  private live(now = Date.now()): ProfileEntry[] {
    return this.project(now).sort(
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
   * 追加一条断言层。Returns undefined when a dormant/import write tries
   * to assert over an explicit user entry. 同来源同值重复断言幂等跳过。
   */
  upsert(input: ProfileEntryInput): ProfileEntry | undefined {
    const source = input.source ?? "user";
    const layers = this.listAssertions(input.section, input.key);
    const hasLiveUser = layers.some((e) => e.source === "user" && !e.revokedAt);
    if (hasLiveUser && source !== "user") return undefined;
    const latestSameSource = [...layers]
      .filter((e) => e.source === source && !e.revokedAt)
      .sort((a, b) => this.seqOf(b) - this.seqOf(a))[0];
    if (latestSameSource && latestSameSource.value === input.value) return latestSameSource;
    const now = Date.now();
    const entry: ProfileEntry = {
      id: randomUUID(),
      section: input.section,
      key: input.key,
      value: input.value,
      source,
      ...(input.provenance ? { provenance: input.provenance } : {}),
      ...(input.validUntil ? { validUntil: input.validUntil } : {}),
      createdAt: now,
      updatedAt: now,
      appliedSeq: this.nextSeq(),
    };
    this.store.upsert(entry);
    return entry;
  }

  /** F11：撤销一条断言（只撤该来源层；下层有效断言按恢复矩阵重新生效）。 */
  revokeAssertion(id: string): boolean {
    const existing = this.store.list().find((e) => e.id === id);
    if (!existing || existing.revokedAt) return false;
    this.store.upsert({ ...existing, revokedAt: Date.now(), updatedAt: Date.now() });
    return true;
  }

  /** F11：用户不再希望用该 key 做推断——持久 suppression，普通导入不可绕过。 */
  dismissInferredKey(section: ProfileSection, key: string): void {
    this.store.dismiss(`${section}:${key}`);
  }

  /**
   * F11：移除用户显式层。mode 必须明确：
   * - "suppress"（默认）：同时 dismiss 推断层，防旧推断静默复活；
   * - "restore"：允许仍有效的推断层重新成为投影值。
   */
  removeUserOverride(
    section: ProfileSection,
    key: string,
    opts: { mode?: "suppress" | "restore" } = {},
  ): boolean {
    const userLayers = this.listAssertions(section, key).filter(
      (e) => e.source === "user" && !e.revokedAt,
    );
    if (userLayers.length === 0) return false;
    for (const layer of userLayers) this.revokeAssertion(layer.id);
    if ((opts.mode ?? "suppress") === "suppress") this.dismissInferredKey(section, key);
    return true;
  }

  remove(id: string): boolean {
    const existing = this.store.list().find((e) => e.id === id);
    if (!existing) return false;
    // Deleting an inferred entry is a durable user decision: never re-import it.
    if (existing.source === "dormant") {
      this.store.dismiss(`${existing.section}:${existing.key}`);
    }
    // F11：删除 = 撤销该断言层（保留审计历史），不再物理抹除。
    return this.revokeAssertion(id);
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

  /** F11：导出包含全部断言层与 suppression（只导 live 值会让撤销在导入端回潮）。 */
  exportJSON(): string {
    return JSON.stringify(
      {
        version: 2,
        entries: this.store.list(),
        dismissed: this.store.listDismissed(),
      },
      null,
      2,
    );
  }

  importJSON(json: string): number {
    const parsed = JSON.parse(json) as {
      version?: number;
      entries?: ProfileEntry[];
      dismissed?: string[];
    };
    if (!Array.isArray(parsed.entries)) throw new Error("invalid profile export: missing entries");
    let count = 0;
    if (parsed.version === 2) {
      // ledger 原样恢复（保留 appliedSeq/revokedAt），再恢复 suppression。
      const valid = parsed.entries.filter(
        (e) => PROFILE_SECTIONS.includes(e.section) && e.key,
      );
      this.store.replaceAll(valid);
      for (const key of parsed.dismissed ?? []) this.store.dismiss(key);
      return valid.length;
    }
    for (const entry of parsed.entries) {
      if (!PROFILE_SECTIONS.includes(entry.section) || !entry.key) continue;
      if (this.upsert({ ...entry, source: entry.source ?? "import" })) count += 1;
    }
    return count;
  }
}
