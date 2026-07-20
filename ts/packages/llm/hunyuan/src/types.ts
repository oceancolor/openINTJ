import { z } from "zod";

export const HUNYUAN_DEFAULT_BASE_URL = "https://tokenhub.tencentmaas.com/v1";
export const HUNYUAN_DEFAULT_MODEL = "hy3";
export const HUNYUAN_DEPRECATED_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1";
export const HUNYUAN_DEPRECATED_MODELS = new Set(["hunyuan-turbos-latest", "hy3-preview"]);

export const HunyuanConfigSchema = z.object({
  apiKey: z.string().default(""),
  baseUrl: z.string().url().default(HUNYUAN_DEFAULT_BASE_URL),
  model: z.string().default(HUNYUAN_DEFAULT_MODEL),
  visionModel: z.string().default("hunyuan-vision"),
  maxTokens: z.number().int().positive().default(2048),
  temperature: z.number().min(0).max(2).default(0.7),
  topP: z.number().min(0).max(1).default(0.9),
  /** 单次请求超时（毫秒）。 */
  timeoutMs: z.number().int().positive().default(60_000),
  /**
   * 联网搜索（功能增强）总开关，对应混元 OpenAI 兼容端点的 `enable_enhancement`。
   * 自 2025-04-20 起官方默认关闭；开启后模型回复会接入实时联网搜索结果。
   * 注意：hunyuan-lite 无此能力。
   */
  enableEnhancement: z.boolean().default(false),
  /** 强制走 AI 搜索（`force_search_enhancement`）；开启时自动隐含开启联网搜索。 */
  forceSearch: z.boolean().default(false),
  /** 命中搜索时返回来源信息（`search_info`）。 */
  searchInfo: z.boolean().default(false),
  /** 引文角标（`citation`），需配合 enableEnhancement + searchInfo。 */
  citation: z.boolean().default(false),
});
export type HunyuanConfig = z.infer<typeof HunyuanConfigSchema>;

/** 解析布尔型 env：`"1"` / `"true"`（大小写不敏感）为真，其余为假。 */
const envBool = (raw: string | undefined): boolean => {
  if (!raw) return false;
  const v = raw.trim().toLowerCase();
  return v === "1" || v === "true" || v === "yes" || v === "on";
};

export type HunyuanConfigWarningHandler = (message: string) => void;

/** Maps only known retired defaults; arbitrary custom endpoints and model IDs are preserved. */
export const migrateHunyuanConfig = (
  config: HunyuanConfig,
  warn: HunyuanConfigWarningHandler = console.warn,
): HunyuanConfig => {
  let baseUrl = config.baseUrl;
  let model = config.model;
  if (baseUrl.replace(/\/+$/, "") === HUNYUAN_DEPRECATED_BASE_URL) {
    warn(
      `[hunyuan] deprecated default endpoint '${baseUrl}' mapped to TokenHub '${HUNYUAN_DEFAULT_BASE_URL}'`,
    );
    baseUrl = HUNYUAN_DEFAULT_BASE_URL;
  }
  if (HUNYUAN_DEPRECATED_MODELS.has(model)) {
    warn(`[hunyuan] deprecated default model '${model}' mapped to '${HUNYUAN_DEFAULT_MODEL}'`);
    model = HUNYUAN_DEFAULT_MODEL;
  }
  return { ...config, baseUrl, model };
};

export const loadHunyuanConfigFromEnv = (
  env: NodeJS.ProcessEnv = process.env,
  warn?: HunyuanConfigWarningHandler,
): HunyuanConfig =>
  migrateHunyuanConfig(
    HunyuanConfigSchema.parse({
      apiKey: env["HUNYUAN_API_KEY"] ?? "",
      baseUrl: env["HUNYUAN_BASE_URL"] ?? HUNYUAN_DEFAULT_BASE_URL,
      model: env["HUNYUAN_MODEL"] ?? HUNYUAN_DEFAULT_MODEL,
      visionModel: env["HUNYUAN_VISION_MODEL"] ?? "hunyuan-vision",
      maxTokens: env["HUNYUAN_MAX_TOKENS"] ? Number.parseInt(env["HUNYUAN_MAX_TOKENS"], 10) : 2048,
      temperature: env["HUNYUAN_TEMPERATURE"] ? Number.parseFloat(env["HUNYUAN_TEMPERATURE"]) : 0.7,
      topP: env["HUNYUAN_TOP_P"] ? Number.parseFloat(env["HUNYUAN_TOP_P"]) : 0.9,
      timeoutMs: env["HUNYUAN_TIMEOUT_MS"]
        ? Number.parseInt(env["HUNYUAN_TIMEOUT_MS"], 10)
        : 60_000,
      enableEnhancement: envBool(env["HUNYUAN_ENABLE_SEARCH"]),
      forceSearch: envBool(env["HUNYUAN_FORCE_SEARCH"]),
      searchInfo: envBool(env["HUNYUAN_SEARCH_INFO"]),
      citation: envBool(env["HUNYUAN_CITATION"]),
    }),
    warn,
  );
