import { z } from "zod";

export const OllamaConfigSchema = z.object({
  baseUrl: z.string().url().default("http://localhost:11434"),
  model: z.string().default("qwen2.5:7b"),
  visionModel: z.string().default("llava:7b"),
  temperature: z.number().min(0).max(2).default(0.7),
  topP: z.number().min(0).max(1).default(0.9),
  numCtx: z.number().int().positive().default(4096),
  timeoutMs: z.number().int().positive().default(120_000),
});
export type OllamaConfig = z.infer<typeof OllamaConfigSchema>;

export const loadOllamaConfigFromEnv = (env: NodeJS.ProcessEnv = process.env): OllamaConfig =>
  OllamaConfigSchema.parse({
    baseUrl: env["OLLAMA_BASE_URL"] ?? "http://localhost:11434",
    model: env["OLLAMA_MODEL"] ?? "qwen2.5:7b",
    visionModel: env["OLLAMA_VISION_MODEL"] ?? "llava:7b",
    temperature: env["OLLAMA_TEMPERATURE"] ? Number.parseFloat(env["OLLAMA_TEMPERATURE"]) : 0.7,
    topP: env["OLLAMA_TOP_P"] ? Number.parseFloat(env["OLLAMA_TOP_P"]) : 0.9,
    numCtx: env["OLLAMA_NUM_CTX"] ? Number.parseInt(env["OLLAMA_NUM_CTX"], 10) : 4096,
    timeoutMs: env["OLLAMA_TIMEOUT_MS"] ? Number.parseInt(env["OLLAMA_TIMEOUT_MS"], 10) : 120_000,
  });
