import type { SpeechToTextProvider, SttTranscribeInput } from "./types.js";

export interface HttpSttOptions {
  /** e.g. https://api.openai.com/v1 — POSTs to <baseUrl>/audio/transcriptions */
  baseUrl: string;
  apiKey?: string;
  model?: string;
  fetchFn?: typeof fetch;
  timeoutMs?: number;
}

const EXT_BY_MIME: Record<string, string> = {
  "audio/webm": "webm",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/mpeg": "mp3",
  "audio/mp4": "m4a",
  "audio/ogg": "ogg",
};

/** OpenAI-compatible transcription provider (whisper-1 style multipart). */
export function createHttpSttProvider(opts: HttpSttOptions): SpeechToTextProvider {
  const base = opts.baseUrl.replace(/\/+$/, "");
  const fetchFn = opts.fetchFn ?? fetch;
  const timeoutMs = opts.timeoutMs ?? 60_000;
  return {
    id: "http",
    async transcribe(input: SttTranscribeInput) {
      const ext = EXT_BY_MIME[input.mimeType] ?? "bin";
      const form = new FormData();
      form.append(
        "file",
        new Blob([input.audio.buffer as ArrayBuffer], { type: input.mimeType }),
        `audio.${ext}`,
      );
      form.append("model", opts.model ?? "whisper-1");
      if (input.language !== undefined) form.append("language", input.language);
      const res = await fetchFn(`${base}/audio/transcriptions`, {
        method: "POST",
        headers: opts.apiKey !== undefined ? { authorization: `Bearer ${opts.apiKey}` } : {},
        body: form,
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!res.ok) throw new Error(`stt http ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const body = (await res.json()) as { text?: unknown };
      if (typeof body.text !== "string") throw new Error("stt response missing text");
      return { text: body.text };
    },
    status() {
      return { id: "http", ok: true, detail: base };
    },
  };
}

/** Deterministic fixed-text provider for dev and tests. */
export function createMockSttProvider(text = "（模拟转写）你好 OpenINTJ"): SpeechToTextProvider {
  return {
    id: "mock",
    async transcribe() {
      return { text };
    },
    status() {
      return { id: "mock", ok: true, detail: "fixed text" };
    },
  };
}
