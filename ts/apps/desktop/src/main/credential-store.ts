import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export interface CredentialCrypto {
  isEncryptionAvailable(): boolean;
  encryptString(value: string): Buffer;
  decryptString(value: Buffer): string;
}

export interface CredentialStore {
  has(id: string): boolean;
  get(id: string): string | undefined;
  set(id: string, value: string): void;
  delete(id: string): boolean;
}

interface CredentialFile {
  schemaVersion: 1;
  entries: Record<string, string>;
}

const EMPTY: CredentialFile = { schemaVersion: 1, entries: {} };
const VALID_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;

/** DPAPI/Keychain-backed credential persistence. Plaintext is never written to disk. */
export const createCredentialStore = (
  filePath: string,
  crypto: CredentialCrypto,
): CredentialStore => {
  let cache: CredentialFile | undefined;

  const ensureCrypto = (): void => {
    if (!crypto.isEncryptionAvailable()) {
      throw new Error("系统安全存储不可用，无法保存 API Key");
    }
  };
  const validateId = (id: string): void => {
    if (!VALID_ID.test(id)) throw new Error("invalid credential id");
  };
  const load = (): CredentialFile => {
    if (cache) return cache;
    try {
      if (!existsSync(filePath)) {
        cache = { ...EMPTY, entries: {} };
        return cache;
      }
      const raw = JSON.parse(readFileSync(filePath, "utf8")) as Partial<CredentialFile>;
      cache =
        raw.schemaVersion === 1 && raw.entries && typeof raw.entries === "object"
          ? { schemaVersion: 1, entries: { ...raw.entries } }
          : { ...EMPTY, entries: {} };
    } catch {
      cache = { ...EMPTY, entries: {} };
    }
    return cache;
  };
  const persist = (): void => {
    mkdirSync(dirname(filePath), { recursive: true });
    const tempPath = `${filePath}.tmp`;
    writeFileSync(tempPath, JSON.stringify(load(), null, 2), { encoding: "utf8", mode: 0o600 });
    try {
      renameSync(tempPath, filePath);
    } finally {
      if (existsSync(tempPath)) rmSync(tempPath, { force: true });
    }
  };

  return {
    has(id) {
      validateId(id);
      return load().entries[id] !== undefined;
    },
    get(id) {
      validateId(id);
      const encrypted = load().entries[id];
      if (!encrypted) return undefined;
      ensureCrypto();
      return crypto.decryptString(Buffer.from(encrypted, "base64"));
    },
    set(id, value) {
      validateId(id);
      if (!value.trim()) throw new Error("API Key 不能为空");
      ensureCrypto();
      load().entries[id] = crypto.encryptString(value.trim()).toString("base64");
      persist();
    },
    delete(id) {
      validateId(id);
      if (!(id in load().entries)) return false;
      delete load().entries[id];
      persist();
      return true;
    },
  };
};
