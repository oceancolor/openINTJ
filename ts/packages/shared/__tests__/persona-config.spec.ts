import { afterEach, describe, expect, it } from "vitest";
import { resolvePersonaInjection } from "../src/persona-config.js";

const ORIG = process.env["OPENINTJ_PERSONA"];
afterEach(() => {
  if (ORIG === undefined) delete process.env["OPENINTJ_PERSONA"];
  else process.env["OPENINTJ_PERSONA"] = ORIG;
});

describe("resolvePersonaInjection (A/B 杠杆)", () => {
  it("默认开（未传 opts、未设 env）", () => {
    delete process.env["OPENINTJ_PERSONA"];
    expect(resolvePersonaInjection()).toBe(true);
    expect(resolvePersonaInjection({})).toBe(true);
  });

  it("显式 opts.enablePersona 优先于 env", () => {
    process.env["OPENINTJ_PERSONA"] = "0";
    expect(resolvePersonaInjection({ enablePersona: true })).toBe(true);
    process.env["OPENINTJ_PERSONA"] = "1";
    expect(resolvePersonaInjection({ enablePersona: false })).toBe(false);
  });

  it("env OPENINTJ_PERSONA=0 / false 关闭注入", () => {
    process.env["OPENINTJ_PERSONA"] = "0";
    expect(resolvePersonaInjection()).toBe(false);
    process.env["OPENINTJ_PERSONA"] = "false";
    expect(resolvePersonaInjection()).toBe(false);
  });

  it("env 其余值视为开启", () => {
    process.env["OPENINTJ_PERSONA"] = "1";
    expect(resolvePersonaInjection()).toBe(true);
    process.env["OPENINTJ_PERSONA"] = "on";
    expect(resolvePersonaInjection()).toBe(true);
  });
});
