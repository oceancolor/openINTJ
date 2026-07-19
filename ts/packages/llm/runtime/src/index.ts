export * from "./types.js";
export * from "./mock-llm.js";
export * from "./strict-wrapper.js";
export * from "./health.js";
export * from "./resolve-llm.js";
export * from "./resolve-embedder.js";
export * from "./embedding-fingerprint.js";
export * from "./embedding-persistence.js";

import { resolveEmbedder } from "./resolve-embedder.js";
import { resolveLlmClient } from "./resolve-llm.js";
import type { EmbeddingFingerprint, ModelRuntimeStatus, ResolveModelRuntimeOpts } from "./types.js";

/** 同时解析 LLM 与 embedding（三端装配入口）。 */
export const resolveModelRuntime = async (
  opts: ResolveModelRuntimeOpts = {},
): Promise<{
  llm: Awaited<ReturnType<typeof resolveLlmClient>>;
  embed: Awaited<ReturnType<typeof resolveEmbedder>>;
  embeddingFingerprint: EmbeddingFingerprint;
  status: ModelRuntimeStatus;
}> => {
  const llm = await resolveLlmClient({
    ...(opts.provider !== undefined ? { provider: opts.provider } : {}),
    ...(opts.env !== undefined ? { env: opts.env } : {}),
    ...(opts.fetch !== undefined ? { fetch: opts.fetch } : {}),
  });
  const embed = await resolveEmbedder({
    ...(opts.embedProvider !== undefined ? { embedProvider: opts.embedProvider } : {}),
    ...(opts.env !== undefined ? { env: opts.env } : {}),
    ...(opts.fetch !== undefined ? { fetch: opts.fetch } : {}),
  });
  const embeddingFingerprint: EmbeddingFingerprint = {
    schemaVersion: 1,
    provider: embed.status.provider,
    model: embed.status.model,
    dimension: embed.dimension,
  };
  return {
    llm,
    embed,
    embeddingFingerprint,
    status: { llm: llm.status, embed: embed.status },
  };
};
