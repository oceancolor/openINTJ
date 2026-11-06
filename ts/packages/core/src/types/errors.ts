import { z } from "zod";

export const ErrorCode = {
  CONFIG_MISSING: "CONFIG_MISSING",
  VALIDATION_ERROR: "VALIDATION_ERROR",
  POLICY_BLOCKED: "POLICY_BLOCKED",
  TOOL_FAILED: "TOOL_FAILED",
  EXECUTION_FAILED: "EXECUTION_FAILED",
  TIMEOUT: "TIMEOUT",
  INTERNAL_ERROR: "INTERNAL_ERROR",
  SHADER_ERROR: "SHADER_ERROR",
  MEMORY_ERROR: "MEMORY_ERROR",
  CONTEXT_OVERFLOW: "CONTEXT_OVERFLOW",
  CIRCUIT_OPEN: "CIRCUIT_OPEN",
  HOOK_ERROR: "HOOK_ERROR",
  STATE_TRANSITION_INVALID: "STATE_TRANSITION_INVALID",
  LOOP_LIMIT_REACHED: "LOOP_LIMIT_REACHED",
  REACT_DUPLICATE_LOOP: "REACT_DUPLICATE_LOOP",
} as const;

export type ErrorCodeType = (typeof ErrorCode)[keyof typeof ErrorCode];

export const ErrorCodeSchema = z.enum(
  Object.values(ErrorCode) as [ErrorCodeType, ...ErrorCodeType[]],
);

export class AgentError extends Error {
  override readonly name = "AgentError";
  readonly code: ErrorCodeType;
  readonly retriable: boolean;
  readonly details: Record<string, unknown>;

  constructor(opts: {
    code: ErrorCodeType;
    message: string;
    retriable?: boolean;
    details?: Record<string, unknown>;
    cause?: unknown;
  }) {
    super(`[${opts.code}] ${opts.message}`);
    this.code = opts.code;
    this.retriable = opts.retriable ?? false;
    this.details = opts.details ?? {};
    if (opts.cause !== undefined) {
      (this as { cause?: unknown }).cause = opts.cause;
    }
  }

  toJSON(): {
    name: string;
    code: ErrorCodeType;
    message: string;
    retriable: boolean;
    details: Record<string, unknown>;
  } {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      retriable: this.retriable,
      details: this.details,
    };
  }
}

export const isAgentError = (e: unknown): e is AgentError =>
  e instanceof AgentError ||
  (typeof e === "object" && e !== null && (e as { name?: string }).name === "AgentError");

/**
 * Provider phrasings for "the conversation no longer fits the context window".
 * Kept deliberately strict (status codes alone never qualify): a false
 * positive costs one wasted summarization + retry, a false negative just
 * falls back to surfacing the original error.
 */
const OVERFLOW_MESSAGE_PATTERNS: readonly RegExp[] = [
  /context_length_exceeded/i,
  /maximum context length/i,
  /prompt is too long/i,
  /exceeds the available context size/i,
  /context (length|window) (exceeded|limit)/i,
  /input is too long for/i,
  /超过[^。]{0,16}上下文(长度|上限|限制)?/,
  /上下文(长度)?(超限|过长|超出)/,
];

export const matchesContextOverflowText = (text: string): boolean =>
  OVERFLOW_MESSAGE_PATTERNS.some((pattern) => pattern.test(text));

/**
 * RFC-012 overflow classification. Structured signals win (adapters throw
 * `CONTEXT_OVERFLOW` or set `details.overflow`), message text is the fallback
 * for unknown providers and wrapped clients.
 */
export const isContextOverflowError = (e: unknown): boolean => {
  if (typeof e === "object" && e !== null) {
    const candidate = e as { code?: unknown; details?: unknown };
    if (candidate.code === ErrorCode.CONTEXT_OVERFLOW) return true;
    if (
      typeof candidate.details === "object" &&
      candidate.details !== null &&
      (candidate.details as { overflow?: unknown }).overflow === true
    ) {
      return true;
    }
  }
  const message = e instanceof Error ? e.message : typeof e === "string" ? e : "";
  return message.length > 0 && matchesContextOverflowText(message);
};
