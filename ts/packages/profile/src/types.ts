export const PROFILE_SECTIONS = [
  "identity",
  "preferences",
  "goals",
  "projects",
  "relationships",
  "constraints",
  "habits",
  "notes",
] as const;

export type ProfileSection = (typeof PROFILE_SECTIONS)[number];

export type ProfileSource = "user" | "dormant" | "import";

export interface ProfileEntry {
  id: string;
  section: ProfileSection;
  key: string;
  value: string;
  source: ProfileSource;
  provenance?: string;
  /** Epoch ms; expired entries are neither rendered nor listed. */
  validUntil?: number;
  createdAt: number;
  updatedAt: number;
  /** F11 assertion ledger：提交顺序（同 key 多层断言的裁决依据）。 */
  appliedSeq?: number;
  /** F11：被撤销时间（epoch ms）；撤销只撤该来源层，不删历史。 */
  revokedAt?: number;
}

export interface ProfileEntryInput {
  section: ProfileSection;
  key: string;
  value: string;
  source?: ProfileSource;
  provenance?: string;
  validUntil?: number;
}

/** Structural shape of Dormant's PersonaConfig (avoids a package dependency). */
export interface PersonaLike {
  preferences?: Record<string, unknown>;
  phrases?: Record<string, string>;
  habits?: Record<string, unknown>;
  context?: Record<string, unknown>;
}
