import { createHttpSttProvider, createMockSttProvider } from "./providers.js";
import type { SpeechToTextProvider } from "./types.js";

/**
 * OPENINTJ_STT_URL (+ OPENINTJ_STT_KEY / OPENINTJ_STT_MODEL) → OpenAI-compatible
 * endpoint. OPENINTJ_STT=mock → fixed-text dev provider. Unset → undefined
 * (host surfaces report 503 / stt_not_configured rather than faking it).
 */
export function sttFromEnv(
  env: Record<string, string | undefined> = process.env,
): SpeechToTextProvider | undefined {
  const url = env.OPENINTJ_STT_URL?.trim();
  if (url !== undefined && url.length > 0) {
    return createHttpSttProvider({
      baseUrl: url,
      ...(env.OPENINTJ_STT_KEY !== undefined ? { apiKey: env.OPENINTJ_STT_KEY } : {}),
      ...(env.OPENINTJ_STT_MODEL !== undefined ? { model: env.OPENINTJ_STT_MODEL } : {}),
    });
  }
  if (env.OPENINTJ_STT === "mock") return createMockSttProvider();
  return undefined;
}
