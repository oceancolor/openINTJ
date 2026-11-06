/**
 * `@openintj/runtime` — RFC-011 bounded factory surface.
 *
 * One PromptSectionRegistry + one run prelude shared by CLI/server/desktop.
 * Not a plugin kernel: TaoLoop and the planes are never hot-swapped, and
 * this package must not depend on Electron.
 */
export * from "./prompt/types.js";
export * from "./prompt/registry.js";
export * from "./prompt/defaults.js";
export * from "./run/prelude.js";
export * from "./run/journal-ctx.js";
export * from "./compact.js";
export * from "./subagent.js";
