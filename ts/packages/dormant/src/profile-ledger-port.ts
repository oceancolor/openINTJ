import type { PersonaLedgerPort } from "./internalization-manager.js";

/**
 * G5.5：profile 断言账本的结构适配面（宿主把 ProfileService 适配进来）。
 * section 只用到 preferences/habits/notes 三个（与 ProfileService
 * .importPersona 的字段映射对齐：phrases→habits、context→notes）。
 */
export type DormantProfileSection = "preferences" | "habits" | "notes";

export interface DormantProfileEntryLike {
  id: string;
  section: string;
  key: string;
  value: string;
  provenance?: string;
  revokedAt?: number;
  validUntil?: number;
}

export interface DormantProfileAdapter {
  upsert(input: {
    section: DormantProfileSection;
    key: string;
    value: string;
    provenance: string;
  }): string | undefined;
  revokeAssertion(id: string): boolean;
  dismiss(section: DormantProfileSection, key: string): void;
  /** "section:key" 形式的抑制键全集。 */
  dismissedKeys(): readonly string[];
  /** source=dormant 的全部断言层（含已撤销，过滤在本工厂内做）。 */
  dormantEntries(): readonly DormantProfileEntryLike[];
}

const PROVENANCE_PREFIX = "dormant:";

/** PersonaConfig targetField → 账本 (section, key)；未知段不入账本。 */
export const mapDormantTargetField = (
  targetField: string,
): { section: DormantProfileSection; key: string } | undefined => {
  const dot = targetField.indexOf(".");
  if (dot < 0) return undefined;
  const head = targetField.slice(0, dot);
  const key = targetField.slice(dot + 1);
  switch (head) {
    case "preferences":
      return { section: "preferences", key };
    case "phrases":
    case "habits":
      return { section: "habits", key };
    case "context":
      return { section: "notes", key };
    default:
      return undefined;
  }
};

/**
 * 把档案账本适配成 InternalizationManager 的 PersonaLedgerPort：
 * provenance 保留原始 targetField（`dormant:<targetField>`），投影
 * 时无损还原（phrases 与 habits 在账本里同 section，靠它区分）。
 */
export const createProfileLedgerPort = (adapter: DormantProfileAdapter): PersonaLedgerPort => ({
  applyAssertion: ({ targetField, value }) => {
    const mapped = mapDormantTargetField(targetField);
    if (!mapped) return undefined;
    return adapter.upsert({
      section: mapped.section,
      key: mapped.key,
      value,
      provenance: `${PROVENANCE_PREFIX}${targetField}`,
    });
  },
  revokeAssertion: (assertionId) => adapter.revokeAssertion(assertionId),
  dismissField: (targetField) => {
    const mapped = mapDormantTargetField(targetField);
    if (mapped) adapter.dismiss(mapped.section, mapped.key);
  },
  isFieldDismissed: (targetField) => {
    const mapped = mapDormantTargetField(targetField);
    if (!mapped) return false;
    return adapter.dismissedKeys().includes(`${mapped.section}:${mapped.key}`);
  },
  liveAssertions: () =>
    adapter
      .dormantEntries()
      .filter(
        (e) =>
          !e.revokedAt &&
          (!e.validUntil || e.validUntil > Date.now()) &&
          (e.provenance?.startsWith(PROVENANCE_PREFIX) ?? false),
      )
      .map((e) => ({
        assertionId: e.id,
        targetField: e.provenance!.slice(PROVENANCE_PREFIX.length),
        value: e.value,
      })),
});
