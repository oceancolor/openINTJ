import { z } from "zod";

export const HunyuanConfigSchema = z.object({
  apiKey: z.string().default(""),
  baseUrl: z.string().url().default("https://api.hunyuan.cloud.tencent.com/v1"),
  model: z.string().default("hunyuan-turbos-latest"),
  visionModel: z.string().default("hunyuan-vision"),
  maxTokens: z.number().int().positive().default(2048),
  temperature: z.number().min(0).max(2).default(0.7),
  topP: z.number().min(0).max(1).default(0.9),
  /** 单次请求超时（毫秒）。 */
  timeoutMs: z.number().int().positive().default(60_000),
});
export type HunyuanConfig = z.infer<typeof HunyuanConfigSchema>;

export const loadHunyuanConfigFromEnv = (env: NodeJS.ProcessEnv = process.env): HunyuanConfig =>
  HunyuanConfigSchema.parse({
    apiKey: env["HUNYUAN_API_KEY"] ?? "",
    baseUrl: env["HUNYUAN_BASE_URL"] ?? "https://api.hunyuan.cloud.tencent.com/v1",
    model: env["HUNYUAN_MODEL"] ?? "hunyuan-turbos-latest",
    visionModel: env["HUNYUAN_VISION_MODEL"] ?? "hunyuan-vision",
    maxTokens: env["HUNYUAN_MAX_TOKENS"] ? Number.parseInt(env["HUNYUAN_MAX_TOKENS"], 10) : 2048,
    temperature: env["HUNYUAN_TEMPERATURE"] ? Number.parseFloat(env["HUNYUAN_TEMPERATURE"]) : 0.7,
    topP: env["HUNYUAN_TOP_P"] ? Number.parseFloat(env["HUNYUAN_TOP_P"]) : 0.9,
    timeoutMs: env["HUNYUAN_TIMEOUT_MS"] ? Number.parseInt(env["HUNYUAN_TIMEOUT_MS"], 10) : 60_000,
  });
