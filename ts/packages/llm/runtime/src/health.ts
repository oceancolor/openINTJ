/** Ollama 轻量健康探测：GET /api/tags，3s 超时。 */
const normalizeModelName = (model: string): string =>
  model.includes(":") ? model : `${model}:latest`;

export const probeOllama = async (
  baseUrl: string,
  timeoutMs = 3000,
  model?: string,
  fetchImpl: typeof globalThis.fetch = globalThis.fetch,
): Promise<{ ok: boolean; reason?: string; models?: string[] }> => {
  const url = `${baseUrl.replace(/\/$/, "")}/api/tags`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetchImpl(url, { signal: controller.signal });
    if (!res.ok) return { ok: false, reason: `HTTP ${res.status}` };
    const payload = (await res.json().catch(() => undefined)) as
      | { models?: Array<{ name?: unknown; model?: unknown }> }
      | undefined;
    const models = (payload?.models ?? [])
      .flatMap((entry) => [entry.name, entry.model])
      .filter((name): name is string => typeof name === "string");
    if (
      model &&
      models.length > 0 &&
      !models.some((installed) => normalizeModelName(installed) === normalizeModelName(model))
    ) {
      return { ok: false, reason: `model_not_installed:${model}`, models };
    }
    return { ok: true, models };
  } catch (e) {
    return {
      ok: false,
      reason: e instanceof Error ? e.message : String(e),
    };
  } finally {
    clearTimeout(timer);
  }
};

export const hasHunyuanCredentials = (env: NodeJS.ProcessEnv): boolean =>
  Boolean(env["HUNYUAN_API_KEY"]?.trim());
