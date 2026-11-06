import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { ProfileEntry } from "./types.js";

export interface ProfileStore {
  list(): readonly ProfileEntry[];
  upsert(entry: ProfileEntry): void;
  remove(id: string): boolean;
  replaceAll(entries: readonly ProfileEntry[]): void;
  /** Keys (`section:key`) of dormant entries the user deleted; never re-imported. */
  listDismissed(): readonly string[];
  dismiss(key: string): void;
}

export class InMemoryProfileStore implements ProfileStore {
  private entries: ProfileEntry[] = [];
  private dismissed = new Set<string>();

  list(): readonly ProfileEntry[] {
    return [...this.entries];
  }

  upsert(entry: ProfileEntry): void {
    const idx = this.entries.findIndex((item) => item.id === entry.id);
    if (idx >= 0) this.entries[idx] = entry;
    else this.entries.push(entry);
  }

  remove(id: string): boolean {
    const before = this.entries.length;
    this.entries = this.entries.filter((item) => item.id !== id);
    return this.entries.length !== before;
  }

  replaceAll(entries: readonly ProfileEntry[]): void {
    this.entries = [...entries];
  }

  listDismissed(): readonly string[] {
    return [...this.dismissed];
  }

  dismiss(key: string): void {
    this.dismissed.add(key);
  }
}

/** Single-file JSON store (<dataDir>/profile.json). Writes are synchronous and whole-file. */
export class JsonProfileStore implements ProfileStore {
  private readonly inner = new InMemoryProfileStore();

  constructor(private readonly filePath: string) {
    if (existsSync(filePath)) {
      try {
        const parsed = JSON.parse(readFileSync(filePath, "utf8")) as {
          entries?: ProfileEntry[];
          dismissed?: string[];
        };
        if (Array.isArray(parsed.entries)) this.inner.replaceAll(parsed.entries);
        for (const key of parsed.dismissed ?? []) this.inner.dismiss(key);
      } catch {
        // Corrupt profile file must never block agent startup; start empty.
      }
    }
  }

  private flush(): void {
    mkdirSync(dirname(this.filePath), { recursive: true });
    writeFileSync(
      this.filePath,
      JSON.stringify(
        { version: 1, entries: this.inner.list(), dismissed: this.inner.listDismissed() },
        null,
        2,
      ),
      "utf8",
    );
  }

  list(): readonly ProfileEntry[] {
    return this.inner.list();
  }

  upsert(entry: ProfileEntry): void {
    this.inner.upsert(entry);
    this.flush();
  }

  remove(id: string): boolean {
    const removed = this.inner.remove(id);
    if (removed) this.flush();
    return removed;
  }

  replaceAll(entries: readonly ProfileEntry[]): void {
    this.inner.replaceAll(entries);
    this.flush();
  }

  listDismissed(): readonly string[] {
    return this.inner.listDismissed();
  }

  dismiss(key: string): void {
    this.inner.dismiss(key);
    this.flush();
  }
}
