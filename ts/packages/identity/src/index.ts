export {
  detectSeedKind,
  isEmail,
  isPhoneLike,
  isWechatId,
  normalizeEmail,
  normalizePhone,
  normalizeSeed,
  normalizeUsername,
  normalizeUrl,
  normalizeWechat,
  sameAccountValue,
} from "./normalize.js";
export { createGravatarResolver, type GravatarResolverOpts } from "./gravatar.js";
export { createLocalCorpusResolver, type LocalCorpusResolverOpts } from "./local.js";
export {
  IdentityService,
  derivePrimaryIdentity,
  type ConfirmResult,
  type IdentityServiceOpts,
} from "./service.js";
export {
  InMemoryIdentityStore,
  JsonIdentityStore,
  type IdentityStore,
} from "./store.js";
export type {
  AssertAccountInput,
  IdentityDocument,
  IdentityPerson,
  IdentityProposal,
  IdentityResolver,
  IdentitySeed,
  IdentitySeedKind,
  LinkedAccount,
  LinkedAccountSource,
  ResolveResult,
  ResolverReport,
} from "./types.js";
