import { describe, expect, it } from "vitest";
import { formatElapsed, formatNumber, fromDisplayLength, parseNumber, toDisplayLength } from "../src/util/format";

describe("number formatting (same rules as v1 CommitInput)", () => {
  it("switches to exponent notation outside [1e-3, 1e5)", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(100)).toBe("100");
    expect(formatNumber(0.004)).toBe("0.004");
    expect(formatNumber(13.56e6)).toBe("1.356e7");
    expect(formatNumber(5e14)).toBe("5e14");
    expect(formatNumber(4.17e-10)).toBe("4.17e-10");
    expect(formatNumber(-2.5e-4)).toBe("-2.5e-4");
    expect(formatNumber(1 / 3)).toBe("0.333333");
  });

  it("parses decimals and exponents and rejects anything else", () => {
    expect(parseNumber(" 1e14 ")).toBe(1e14);
    expect(parseNumber("-.5")).toBe(-0.5);
    expect(parseNumber("3.")).toBe(3);
    expect(parseNumber("")).toBeNull();
    expect(parseNumber("abc")).toBeNull();
    expect(parseNumber("1e")).toBeNull();
    expect(parseNumber("0x10")).toBeNull();
  });

  it("converts lengths and formats elapsed time", () => {
    expect(toDisplayLength(0.004, "mm")).toBeCloseTo(4);
    expect(toDisplayLength(0.004, "um")).toBeCloseTo(4000);
    expect(fromDisplayLength(4, "mm")).toBeCloseTo(0.004);
    expect(formatElapsed(59)).toBe("0:59");
    expect(formatElapsed(3725)).toBe("1:02:05");
  });
});
