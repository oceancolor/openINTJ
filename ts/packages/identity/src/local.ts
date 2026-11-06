import { isPhoneLike, normalizePhone, sameAccountValue } from "./normalize.js";
import type {
  IdentityDocument,
  IdentityProposal,
  IdentityResolver,
  IdentitySeedKind,
} from "./types.js";

export interface LocalCorpusResolverOpts {
  /** Host-owned read-only corpus (mail bodies, calendar lines, notes). */
  documents: () => readonly IdentityDocument[] | Promise<readonly IdentityDocument[]>;
  /** Hard cap on proposals per resolve (default 50). */
  maxProposals?: number;
}

const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g;
const PHONE_RE = /(?:\+?86[-\s]?)?1[3-9]\d{9}|\+?\d[\d\s-]{7,16}\d/g;
const WECHAT_RE = /(?:微信|wechat|v信|wx)[:：\s]*([a-zA-Z][a-zA-Z0-9_-]{5,19})/gi;
const HANDLE_RE = /(?<![\w.-])@([A-Za-z0-9_.-]{2,30})/g;
const URL_RE = /https?:\/\/[^\s<>"')]+/g;

interface Hit {
  kind: IdentitySeedKind;
  value: string;
  index: number;
}

const platformFor = (kind: IdentitySeedKind, value: string): string => {
  switch (kind) {
    case "email":
      return "mail";
    case "phone":
      return "phone";
    case "wechat":
      return "wechat";
    case "username":
      return "handle";
    case "url":
      try {
        return new URL(value).hostname.replace(/^www\./, "").split(".")[0] ?? "web";
      } catch {
        return "web";
      }
  }
};

const extractHits = (text: string): Hit[] => {
  const hits: Hit[] = [];
  for (const m of text.matchAll(EMAIL_RE)) {
    hits.push({ kind: "email", value: m[0].toLowerCase(), index: m.index });
  }
  for (const m of text.matchAll(PHONE_RE)) {
    if (isPhoneLike(m[0])) {
      hits.push({ kind: "phone", value: normalizePhone(m[0]), index: m.index });
    }
  }
  for (const m of text.matchAll(WECHAT_RE)) {
    hits.push({ kind: "wechat", value: m[1]!, index: m.index });
  }
  for (const m of text.matchAll(HANDLE_RE)) {
    hits.push({ kind: "username", value: m[1]!, index: m.index });
  }
  for (const m of text.matchAll(URL_RE)) {
    hits.push({ kind: "url", value: m[0], index: m.index });
  }
  return hits;
};

const docContainsSeed = (
  doc: IdentityDocument,
  seedKind: IdentitySeedKind,
  seedValue: string,
): boolean => {
  if (seedKind === "phone") {
    const digits = doc.text.replace(/[^\d]/g, "");
    return digits.includes(seedValue.replace(/[^\d]/g, ""));
  }
  const hay = doc.text.toLowerCase();
  const needle = seedValue.toLowerCase();
  if (seedKind === "username") return hay.includes(`@${needle}`) || hay.includes(needle);
  return hay.includes(needle);
};

const snippet = (text: string, index: number): string => {
  const start = Math.max(0, index - 24);
  const end = Math.min(text.length, index + 40);
  return `…${text.slice(start, end).replace(/\s+/g, " ").trim()}…`;
};

/**
 * Co-occurrence linking inside the user's own local data: identifiers that
 * appear in the same mail / calendar entry / note as the seed are proposed
 * as the same person's accounts. There is deliberately no phone/wechat
 * reverse lookup — only what the user's own corpus already contains.
 */
export const createLocalCorpusResolver = (opts: LocalCorpusResolverOpts): IdentityResolver => {
  const maxProposals = opts.maxProposals ?? 50;
  return {
    id: "local-corpus",
    async resolve(seed) {
      const docs = await opts.documents();
      const proposals: IdentityProposal[] = [];
      const seen = new Set<string>();
      for (const doc of docs) {
        if (!docContainsSeed(doc, seed.kind, seed.value)) continue;
        for (const hit of extractHits(doc.text)) {
          if (sameAccountValue(hit.kind, hit.value, seed.value) && hit.kind === seed.kind) {
            continue;
          }
          const key = `${hit.kind}:${hit.value.toLowerCase()}`;
          if (seen.has(key)) continue;
          seen.add(key);
          proposals.push({
            id: `local:${proposals.length}`,
            personId: null,
            groupKey: `local:${doc.source}:${doc.title}`,
            account: {
              platform: platformFor(hit.kind, hit.value),
              kind: hit.kind,
              value: hit.value,
              ...(hit.kind === "url" ? { url: hit.value } : {}),
              source: doc.source,
              evidence: `${doc.title} — ${snippet(doc.text, hit.index)}`,
              confidence: 0.6,
              verifiedAt: null,
            },
          });
          if (proposals.length >= maxProposals) return proposals;
        }
      }
      return proposals;
    },
  };
};
