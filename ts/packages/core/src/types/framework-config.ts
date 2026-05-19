import { z } from "zod";
import { AgentError, ErrorCode } from "./errors.js";
import { ShaderMode, ShaderModeSchema } from "./shader.js";

export const FrameworkEnvSchema = z.enum(["dev", "test", "prod"]);
export type FrameworkEnv = z.infer<typeof FrameworkEnvSchema>;

export const FrameworkConfigSchema = z.object({
  env: FrameworkEnvSchema.default("dev"),
  maxRetry: z.number().int().min(0).default(2),
  defaultTimeoutS: z.number().int().positive().default(30),
  governanceStrict: z.boolean().default(true),
  maxContextTokens: z.number().int().min(1024).default(128_000),
  shaderMode: ShaderModeSchema.default(ShaderMode.ADAPTIVE),
  memoryHalfLifeHours: z.number().positive().default(24),
});

export type FrameworkConfig = z.infer<typeof FrameworkConfigSchema>;

const truthy = new Set(["true", "1", "yes", "on"]);

const parseBool = (v: string | undefined, fallback: boolean): boolean => {
  if (v === undefined || v === "") return fallback;
  return truthy.has(v.toLowerCase());
};

const parseInt10 = (v: string | undefined, key: string): number | undefined => {
  if (v === undefined || v === "") return undefined;
  const n = Number.parseInt(v, 10);
  if (Number.isNaN(n)) {
    throw new AgentError({
      code: ErrorCode.VALIDATION_ERROR,
      message: `配置项 ${key} 不是合法整数: ${v}`,
    });
  }
  return n;
};

const parseFloatStrict = (v: string | undefined, key: string): number | undefined => {
  if (v === undefined || v === "") return undefined;
  const n = Number.parseFloat(v);
  if (Number.isNaN(n)) {
    throw new AgentError({
      code: ErrorCode.VALIDATION_ERROR,
      message: `配置项 ${key} 不是合法数字: ${v}`,
    });
  }
  return n;
};

export const loadFrameworkConfigFromEnv = (
  env: NodeJS.ProcessEnv = process.env,
): FrameworkConfig => {
  const required = ["AGENT_ENV", "AGENT_MAX_RETRY", "AGENT_DEFAULT_TIMEOUT_S"];
  const missing = required.filter((k) => env[k] === undefined);
  if (missing.length > 0) {
    throw new AgentError({
      code: ErrorCode.CONFIG_MISSING,
      message: "缺少必需的配置项",
      details: { missingKeys: missing },
    });
  }

  const candidate = {
    env: env["AGENT_ENV"] ?? "dev",
    maxRetry: parseInt10(env["AGENT_MAX_RETRY"], "AGENT_MAX_RETRY") ?? 2,
    defaultTimeoutS: parseInt10(env["AGENT_DEFAULT_TIMEOUT_S"], "AGENT_DEFAULT_TIMEOUT_S") ?? 30,
    governanceStrict: parseBool(env["AGENT_GOVERNANCE_STRICT"], true),
    maxContextTokens:
      parseInt10(env["AGENT_MAX_CONTEXT_TOKENS"], "AGENT_MAX_CONTEXT_TOKENS") ?? 128_000,
    shaderMode: env["AGENT_SHADER_MODE"] ?? ShaderMode.ADAPTIVE,
    memoryHalfLifeHours:
      parseFloatStrict(env["AGENT_MEMORY_HALF_LIFE_HOURS"], "AGENT_MEMORY_HALF_LIFE_HOURS") ?? 24,
  };

  const parsed = FrameworkConfigSchema.safeParse(candidate);
  if (!parsed.success) {
    throw new AgentError({
      code: ErrorCode.VALIDATION_ERROR,
      message: "配置校验失败",
      details: { issues: parsed.error.flatten() },
    });
  }
  return parsed.data;
};
