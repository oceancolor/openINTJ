import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { type AppConfig, AppConfigPatchSchema, AppConfigSchema } from "../shared/ipc-protocol.js";

/**
 * 应用配置服务：把 {@link AppConfig} 持久化为一个 JSON 文件（通常在 userData 目录）。
 *
 * 设计：
 *  - 读时容错：文件不存在 / 损坏 / schema 不符 → 回退到 `{}`（绝不崩主进程）。
 *  - 写时校验：`update` 用 zod 校验补丁，浅合并后整体再校验一次。
 *  - 同步 IO：配置文件很小，启动期与偶发写入用同步 IO 足够，避免引入异步竞态。
 */
export interface ConfigService {
  get(): AppConfig;
  update(patch: unknown): AppConfig;
  /** 配置文件绝对路径（用于诊断 / 测试）。 */
  readonly path: string;
}

export const createConfigService = (filePath: string): ConfigService => {
  let cache: AppConfig | undefined;

  const load = (): AppConfig => {
    if (cache) return cache;
    try {
      if (!existsSync(filePath)) {
        cache = {};
        return cache;
      }
      const raw = readFileSync(filePath, "utf8");
      const parsed = AppConfigSchema.safeParse(JSON.parse(raw));
      cache = parsed.success ? parsed.data : {};
    } catch {
      cache = {};
    }
    return cache;
  };

  const persist = (cfg: AppConfig): void => {
    mkdirSync(dirname(filePath), { recursive: true });
    writeFileSync(filePath, JSON.stringify(cfg, null, 2), "utf8");
  };

  return {
    path: filePath,
    get(): AppConfig {
      return { ...load() };
    },
    update(patch: unknown): AppConfig {
      const parsed = AppConfigPatchSchema.safeParse(patch ?? {});
      if (!parsed.success) {
        throw new Error(`invalid config patch: ${parsed.error.message}`);
      }
      const next = AppConfigSchema.parse({ ...load(), ...parsed.data });
      cache = next;
      persist(next);
      return { ...next };
    },
  };
};
