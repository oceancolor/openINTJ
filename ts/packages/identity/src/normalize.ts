import type { IdentitySeed, IdentitySeedKind } from "./types.js";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
const CN_MOBILE_RE = /^1[3-9]\d{9}$/;
const PHONE_LIKE_RE = /^\+?\d[\d\s-]{5,18}\d$/;
const WECHAT_RE = /^[a-zA-Z][a-zA-Z0-9_-]{5,19}$/;

export const normalizeEmail = (raw: string): string => raw.trim().toLowerCase();
export const isEmail = (value: string): boolean => EMAIL_RE.test(value);

/** Digits only, with a leading "+" preserved when present. */
export const normalizePhone = (raw: string): string => {
  const trimmed = raw.trim();
  const digits = trimmed.replace(/[^\d]/g, "");
  // +86 / 86-prefixed Chinese mobiles collapse to the bare 11-digit form.
  if (digits.length === 13 && digits.startsWith("86") && CN_MOBILE_RE.test(digits.slice(2))) {
    return digits.slice(2);
  }
  return trimmed.startsWith("+") ? `+${digits}` : digits;
};
export const isPhoneLike = (value: string): boolean => {
  if (CN_MOBILE_RE.test(value)) return true;
  const digits = value.replace(/[^\d]/g, "");
  return PHONE_LIKE_RE.test(value) && digits.length >= 7 && digits.length <= 15;
};

export const normalizeWechat = (raw: string): string => raw.trim();
export const isWechatId = (value: string): boolean => WECHAT_RE.test(value);

export const normalizeUsername = (raw: string): string => raw.trim().replace(/^@+/, "");

export const normalizeUrl = (raw: string): string => {
  const trimmed = raw.trim();
  const withScheme = /^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`;
  // Throws on malformed input — callers treat that as "not a url".
  return new URL(withScheme).toString();
};

/** Best-effort kind detection for free-text seed input. */
export const detectSeedKind = (raw: string): IdentitySeedKind => {
  const value = raw.trim();
  if (isEmail(normalizeEmail(value))) return "email";
  if (/^https?:\/\//i.test(value) || /^[\w-]+(\.[\w-]+)+\//.test(value)) return "url";
  if (isPhoneLike(value)) return "phone";
  if (isWechatId(value)) return "wechat";
  return "username";
};

/** Normalize + validate a seed; throws `identity_invalid_seed` on bad input. */
export const normalizeSeed = (kind: IdentitySeedKind, raw: string): IdentitySeed => {
  const fail = (): never => {
    throw new Error(`identity_invalid_seed: ${kind} "${raw}"`);
  };
  switch (kind) {
    case "email": {
      const value = normalizeEmail(raw);
      return isEmail(value) ? { kind, value } : fail();
    }
    case "phone": {
      const value = normalizePhone(raw);
      return isPhoneLike(value) ? { kind, value } : fail();
    }
    case "wechat": {
      const value = normalizeWechat(raw);
      return isWechatId(value) ? { kind, value } : fail();
    }
    case "url": {
      try {
        return { kind, value: normalizeUrl(raw) };
      } catch {
        return fail();
      }
    }
    default: {
      const value = normalizeUsername(raw);
      return value.length > 0 && !/\s/.test(value) ? { kind, value } : fail();
    }
  }
};

/** Loose equality for account values across resolvers (case/digit folding). */
export const sameAccountValue = (kind: IdentitySeedKind, a: string, b: string): boolean => {
  if (kind === "email" || kind === "username" || kind === "wechat") {
    return a.toLowerCase() === b.toLowerCase();
  }
  if (kind === "phone") {
    return normalizePhone(a) === normalizePhone(b);
  }
  return a === b;
};
