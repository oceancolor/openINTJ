import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { IdentityPerson } from "./types.js";

/** Whole-graph store: the graph is small (one user's identities), so replace-all is fine. */
export interface IdentityStore {
  list(): readonly IdentityPerson[];
  replaceAll(persons: readonly IdentityPerson[]): void;
}

export class InMemoryIdentityStore implements IdentityStore {
  private persons: IdentityPerson[] = [];

  list(): readonly IdentityPerson[] {
    return [...this.persons];
  }

  replaceAll(persons: readonly IdentityPerson[]): void {
    this.persons = [...persons];
  }
}

interface IdentityFileShape {
  version: 1;
  persons: IdentityPerson[];
}

const isPersonLike = (value: unknown): value is IdentityPerson => {
  if (typeof value !== "object" || value === null) return false;
  const p = value as Record<string, unknown>;
  return (
    typeof p["id"] === "string" &&
    typeof p["displayName"] === "string" &&
    Array.isArray(p["accounts"])
  );
};

/** `<dataDir>/identity.json`. Corrupt content reads as an empty graph, never throws. */
export class JsonIdentityStore implements IdentityStore {
  private readonly filePath: string;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  list(): readonly IdentityPerson[] {
    if (!existsSync(this.filePath)) return [];
    try {
      const parsed = JSON.parse(readFileSync(this.filePath, "utf8")) as IdentityFileShape;
      if (parsed.version !== 1 || !Array.isArray(parsed.persons)) return [];
      return parsed.persons.filter(isPersonLike);
    } catch {
      return [];
    }
  }

  replaceAll(persons: readonly IdentityPerson[]): void {
    mkdirSync(dirname(this.filePath), { recursive: true });
    const payload: IdentityFileShape = { version: 1, persons: [...persons] };
    writeFileSync(this.filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  }
}
