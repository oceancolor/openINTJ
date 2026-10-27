import { AgentError, ErrorCode } from "@openintj/core";

export const HUNYUAN_IMAGE_MODEL = "hy-image-v3";
export const HUNYUAN_IMAGE_DEFAULT_SIZE = "1280x720";
export const HUNYUAN_IMAGE_TIMEOUT_MS = 120_000;

export interface HunyuanImageResult {
  url: string;
  bytes: Uint8Array;
  mimeType: string;
  revisedPrompt?: string;
}

interface HunyuanImageEnvelope {
  url?: string;
  data?: Array<{ url?: string; b64_json?: string; revised_prompt?: string }>;
  error?: { message?: string; code?: string; type?: string };
}

const imageEndpoint = (baseUrl: string): string =>
  `${baseUrl.replace(/\/$/, "")}/wand/hunyuan-image/v3-generation`;

const parseImageUrl = (parsed: HunyuanImageEnvelope): string | undefined =>
  parsed.data?.[0]?.url ?? (typeof parsed.url === "string" ? parsed.url : undefined);

const mimeFromResponse = (contentType: string | null): string => {
  const raw = contentType?.split(";")[0]?.trim().toLowerCase();
  if (raw === "image/jpeg" || raw === "image/jpg") return "image/jpeg";
  if (raw === "image/webp") return "image/webp";
  return "image/png";
};

export const generateHunyuanImage = async (opts: {
  apiKey: string;
  baseUrl: string;
  prompt: string;
  size?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
}): Promise<HunyuanImageResult> => {
  const prompt = opts.prompt.trim();
  if (!opts.apiKey) {
    throw new AgentError({
      code: ErrorCode.CONFIG_MISSING,
      message: "Hunyuan 生图不可用: missing API key",
      retriable: false,
      details: { provider: "hunyuan-image" },
    });
  }
  if (!prompt) {
    throw new AgentError({
      code: ErrorCode.VALIDATION_ERROR,
      message: "Hunyuan 生图需要非空 prompt",
      retriable: false,
    });
  }

  const timeoutMs = opts.timeoutMs ?? HUNYUAN_IMAGE_TIMEOUT_MS;
  const controller = new AbortController();
  const abortFromCaller = (): void => controller.abort(opts.signal?.reason);
  if (opts.signal?.aborted) abortFromCaller();
  else opts.signal?.addEventListener("abort", abortFromCaller, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort(new Error(`timeout after ${timeoutMs}ms`));
  }, timeoutMs);

  try {
    const res = await fetch(imageEndpoint(opts.baseUrl), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${opts.apiKey}`,
      },
      body: JSON.stringify({
        model: HUNYUAN_IMAGE_MODEL,
        prompt,
        size: opts.size ?? HUNYUAN_IMAGE_DEFAULT_SIZE,
        revise: true,
      }),
      signal: controller.signal,
    });
    const text = await res.text();
    let parsed: HunyuanImageEnvelope = {};
    try {
      parsed = JSON.parse(text) as HunyuanImageEnvelope;
    } catch {
      parsed = {};
    }
    if (!res.ok) {
      const message = parsed.error?.message ?? text.slice(0, 200);
      const billingBlocked = res.status === 402;
      throw new AgentError({
        code:
          billingBlocked || res.status === 401 || res.status === 403
            ? ErrorCode.CONFIG_MISSING
            : ErrorCode.INTERNAL_ERROR,
        message: billingBlocked
          ? `Hunyuan 生图失败 (402): 未开通 Hy-Image-3.0 后付费且无免费额度。请到 TokenHub 控制台开通：https://console.cloud.tencent.com/tokenhub/inference`
          : `Hunyuan 生图失败 (${res.status}): ${message}`,
        retriable: res.status === 429 || res.status >= 500,
        details: { provider: "hunyuan-image", status: res.status },
      });
    }
    const b64 = parsed.data?.[0]?.b64_json;
    if (typeof b64 === "string" && b64.trim()) {
      const bytes = Uint8Array.from(Buffer.from(b64, "base64"));
      return {
        url: "",
        bytes,
        mimeType: "image/png",
        ...(parsed.data?.[0]?.revised_prompt ? { revisedPrompt: parsed.data[0].revised_prompt } : {}),
      };
    }
    const url = parseImageUrl(parsed);
    if (!url) {
      throw new AgentError({
        code: ErrorCode.INTERNAL_ERROR,
        message: "Hunyuan 生图响应缺少图片 URL",
        retriable: true,
        details: { provider: "hunyuan-image" },
      });
    }
    const imgRes = await fetch(url, { signal: controller.signal });
    if (!imgRes.ok) {
      throw new AgentError({
        code: ErrorCode.INTERNAL_ERROR,
        message: `Hunyuan 生图下载失败 (${imgRes.status})`,
        retriable: true,
        details: { provider: "hunyuan-image", status: imgRes.status },
      });
    }
    const bytes = new Uint8Array(await imgRes.arrayBuffer());
    return {
      url,
      bytes,
      mimeType: mimeFromResponse(imgRes.headers.get("content-type")),
      ...(parsed.data?.[0]?.revised_prompt ? { revisedPrompt: parsed.data[0].revised_prompt } : {}),
    };
  } catch (error) {
    if (error instanceof AgentError) throw error;
    if (opts.signal?.aborted) {
      if (opts.signal.reason instanceof Error) throw opts.signal.reason;
      throw new AgentError({
        code: ErrorCode.EXECUTION_FAILED,
        message: "Hunyuan 生图已取消",
        retriable: false,
        details: { provider: "hunyuan-image" },
      });
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new AgentError({
      code: timedOut ? ErrorCode.TIMEOUT : ErrorCode.INTERNAL_ERROR,
      message: timedOut ? `Hunyuan 生图超时: ${message}` : `Hunyuan 生图网络错误: ${message}`,
      retriable: true,
      cause: error,
    });
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener("abort", abortFromCaller);
  }
};
