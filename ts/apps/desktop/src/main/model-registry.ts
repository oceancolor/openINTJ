import type { HookBus, LlmClient } from "@openintj/core";
import { type LlmProviderId, resolveLlmClient } from "@openintj/model-runtime";
import { DEFAULT_MODEL_PROFILES, type ModelProfile } from "../shared/ipc-protocol.js";
import type { ConfigService } from "./config-store.js";
import type { CredentialStore } from "./credential-store.js";

const ENV_BY_PROVIDER: Partial<
  Record<LlmProviderId, { key: string; baseUrl: string; model: string }>
> = {
  hunyuan: { key: "HUNYUAN_API_KEY", baseUrl: "HUNYUAN_BASE_URL", model: "HUNYUAN_MODEL" },
  kimi: { key: "KIMI_API_KEY", baseUrl: "KIMI_BASE_URL", model: "KIMI_MODEL" },
  minimax: { key: "MINIMAX_API_KEY", baseUrl: "MINIMAX_BASE_URL", model: "MINIMAX_MODEL" },
  glm: { key: "GLM_API_KEY", baseUrl: "GLM_BASE_URL", model: "GLM_MODEL" },
  ollama: { key: "", baseUrl: "OLLAMA_BASE_URL", model: "OLLAMA_MODEL" },
};

export interface ModelRegistry {
  list(): ModelProfile[];
  resolve(profileId: string): Promise<LlmClient>;
  test(
    profileId: string,
  ): Promise<{ ok: boolean; provider: string; model: string; error?: string }>;
  clear(profileId?: string): void;
}

export const createModelRegistry = (opts: {
  config: ConfigService;
  credentials: CredentialStore;
  hooks?: HookBus;
  env?: NodeJS.ProcessEnv;
}): ModelRegistry => {
  const cache = new Map<string, Promise<LlmClient>>();
  const baseEnv = opts.env ?? process.env;

  const list = (): ModelProfile[] => {
    const custom = opts.config.get().modelProfiles ?? [];
    const profiles = [...DEFAULT_MODEL_PROFILES, ...custom] as ModelProfile[];
    const merged = new Map(profiles.map((profile) => [profile.id, profile]));
    return [...merged.values()].map((profile) => ({
      ...profile,
      hasCredential:
        profile.hasCredential === true ||
        opts.credentials.has(profile.id) ||
        profile.provider === "auto" ||
        profile.provider === "ollama" ||
        profile.provider === "mock" ||
        Boolean(baseEnv[ENV_BY_PROVIDER[profile.provider]?.key ?? ""]),
    }));
  };
  const find = (profileId: string): ModelProfile => {
    const profile = list().find((candidate) => candidate.id === profileId);
    if (!profile) throw new Error(`unknown model profile: ${profileId}`);
    return profile;
  };
  const create = async (profile: ModelProfile): Promise<LlmClient> => {
    const env = { ...baseEnv };
    const names = ENV_BY_PROVIDER[profile.provider];
    if (names) {
      const credential = opts.credentials.get(profile.id);
      if (credential && names.key) env[names.key] = credential;
      if (profile.baseUrl) env[names.baseUrl] = profile.baseUrl;
      if (profile.model && profile.model !== "auto") env[names.model] = profile.model;
    }
    const resolved = await resolveLlmClient({
      provider: profile.provider,
      env,
      ...(opts.hooks ? { hooks: opts.hooks } : {}),
    });
    return resolved.client;
  };

  return {
    list,
    resolve(profileId) {
      let pending = cache.get(profileId);
      if (!pending) {
        pending = create(find(profileId));
        cache.set(profileId, pending);
        void pending.catch(() => cache.delete(profileId));
      }
      return pending;
    },
    async test(profileId) {
      const profile = find(profileId);
      try {
        const client = await create(profile);
        await client.chat([{ role: "user", content: "只回复 OK" }], {
          temperature: 0,
          maxTokens: 8,
        });
        return { ok: true, provider: profile.provider, model: profile.model };
      } catch (error) {
        return {
          ok: false,
          provider: profile.provider,
          model: profile.model,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    },
    clear(profileId) {
      if (profileId) cache.delete(profileId);
      else cache.clear();
    },
  };
};
