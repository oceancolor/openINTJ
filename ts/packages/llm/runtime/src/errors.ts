import type { ModelRuntimeStatus } from "./types.js";

export type ModelRuntimeErrorCode =
  | "MODEL_PROVIDER_UNAVAILABLE"
  | "MODEL_NOT_INSTALLED"
  | "MODEL_CREDENTIAL_MISSING"
  | "MODEL_AUTH_FAILED"
  | "MODEL_REQUEST_FAILED"
  | "EMBEDDING_DIMENSION_UNKNOWN"
  | "EMBEDDING_FINGERPRINT_MISSING"
  | "EMBEDDING_FINGERPRINT_MISMATCH";

export interface ModelRuntimeErrorInit {
  code: ModelRuntimeErrorCode;
  message: string;
  retriable: boolean;
  provider?: string;
  cause?: unknown;
  status?: ModelRuntimeStatus;
}

const SECRET_PATTERNS = [
  /\bBearer\s+[^\s,;]+/gi,
  /\b(?:api[_-]?key|authorization|token)\s*[:=]\s*[^\s,;]+/gi,
  /\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]+\b/g,
];

/** Error text safe for status, hooks, logs, and telemetry attributes. */
export const sanitizeModelRuntimeErrorMessage = (value: unknown, maxLength = 256): string => {
  let message = value instanceof Error ? value.message : String(value);
  for (const pattern of SECRET_PATTERNS) {
    message = message.replace(pattern, (match) => `${match.split(/[:=\s]/, 1)[0]} [REDACTED]`);
  }
  return message.length > maxLength ? `${message.slice(0, maxLength - 1)}…` : message;
};

export class ModelRuntimeError extends Error {
  readonly code: ModelRuntimeErrorCode;
  readonly retriable: boolean;
  readonly provider?: string;
  readonly cause?: unknown;
  readonly status?: ModelRuntimeStatus;

  constructor(init: ModelRuntimeErrorInit) {
    super(sanitizeModelRuntimeErrorMessage(init.message));
    this.name = "ModelRuntimeError";
    this.code = init.code;
    this.retriable = init.retriable;
    if (init.provider !== undefined) this.provider = init.provider;
    if (init.cause !== undefined) this.cause = init.cause;
    if (init.status !== undefined) this.status = init.status;
  }
}

export const runtimeErrorInfo = (
  error: ModelRuntimeError,
  at = Date.now(),
): {
  code: ModelRuntimeErrorCode;
  message: string;
  retriable: boolean;
  at: number;
} => ({
  code: error.code,
  message: sanitizeModelRuntimeErrorMessage(error.message),
  retriable: error.retriable,
  at,
});
