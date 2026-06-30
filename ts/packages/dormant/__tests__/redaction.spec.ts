import { describe, expect, it } from "vitest";
import { createRedactor, defaultRedactor } from "../src/redaction.js";

describe("defaultRedactor", () => {
  it("脱敏邮箱", () => {
    const out = defaultRedactor("联系我 alice@example.com 谢谢");
    expect(out).not.toContain("alice@example.com");
    expect(out).toContain("[REDACTED_EMAIL]");
  });

  it("脱敏 API key", () => {
    const out = defaultRedactor("key=sk-ABCDEFGH12345678ijkl");
    expect(out).toContain("[REDACTED_KEY]");
    expect(out).not.toContain("sk-ABCDEFGH12345678ijkl");
  });

  it("脱敏身份证号", () => {
    const out = defaultRedactor("身份证 11010519900307123X 已记录");
    expect(out).toContain("[REDACTED_ID]");
    expect(out).not.toContain("11010519900307123X");
  });

  it("脱敏银行卡号（含分组）", () => {
    const out = defaultRedactor("卡号 4111 1111 1111 1111");
    expect(out).toContain("[REDACTED_CARD]");
  });

  it("不动普通文本", () => {
    const text = "今天天气不错，我们去喝茶吧";
    expect(defaultRedactor(text)).toBe(text);
  });

  it("空串安全", () => {
    expect(defaultRedactor("")).toBe("");
  });
});

describe("createRedactor", () => {
  it("可禁用某条内置规则", () => {
    const r = createRedactor({ disable: ["email"] });
    const out = r("alice@example.com");
    expect(out).toContain("alice@example.com");
  });

  it("可追加自定义规则", () => {
    const r = createRedactor({
      extraRules: [{ name: "secret", pattern: /SECRET-\d+/g, placeholder: "[X]" }],
    });
    expect(r("token SECRET-42")).toBe("token [X]");
  });
});
