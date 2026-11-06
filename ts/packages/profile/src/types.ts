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
