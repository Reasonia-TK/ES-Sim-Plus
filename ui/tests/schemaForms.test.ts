import { describe, expect, it } from "vitest";
import { BUNDLED_SCHEMA, fieldInfo, getIn, objectAt, propertyKeys, schemaDefaults, setIn } from "../src/schema/schema";
import { displayUnit, formatQuantity, parseQuantity, unitFactor, type UnitContext } from "../src/schema/units";
import { fieldLabel, enumLabel } from "../src/forms/labels";
import { fieldsEn } from "../src/i18n/fields.en";
import { fieldsJa } from "../src/i18n/fields.ja";

const S = BUNDLED_SCHEMA;
const mm: UnitContext = { lengthUnit: "mm", axisymmetric: false };

describe("schema navigation", () => {
  it("resolves nested, nullable and array fields with their metadata", () => {
    const gp = fieldInfo(S, ["fluid2d", "gas_pressure_pa"])!;
    expect(gp).toMatchObject({ kind: "number", model: "Fluid2dSettings", unit: "Pa", nullable: false, min: 0, exclusiveMin: true });
    const dt = fieldInfo(S, ["pic", "dt"])!;
    expect(dt).toMatchObject({ kind: "number", nullable: true, unit: "s" });
    const dens = fieldInfo(S, ["pic", "initial_plasma", "density"])!;
    expect(dens).toMatchObject({ model: "InitialPlasma", unit: "m^-3", required: true });
    const center = fieldInfo(S, ["geometry", "regions", 0, "shape", "center"])!;
    expect(center).toMatchObject({ kind: "point", geom: true, model: "CircleShape" });
    const threads = fieldInfo(S, ["pic", "threads"])!;
    expect(threads).toMatchObject({ kind: "integer", advanced: true, min: 0, max: 128 });
    const mode = fieldInfo(S, ["mesh", "mode"])!;
    expect(mode.kind).toBe("enum");
    expect(mode.enumValues).toEqual(["unstructured", "structured", "cartesian"]);
    expect(fieldInfo(S, ["mesh", "amr", "regions", 2, "level"])).toMatchObject({ kind: "integer", model: "AmrRegion", min: 1 });
    // voltage_rf (単一 | リスト) の中もたどれる
    expect(fieldInfo(S, ["geometry", "boundaries", 0, "voltage_rf", 1, "freq_hz"])).toMatchObject({ model: "VoltageRF", unit: "Hz" });
    expect(fieldInfo(S, ["geometry", "boundaries", 0, "voltage_rf"])?.kind).toBe("union");
    expect(fieldInfo(S, ["nope", "x"])).toBeNull();
    expect(objectAt(S, ["dsmc", "gas"])?.name).toBe("DsmcGas");
    expect(propertyKeys(S, ["pic", "merge"])).toEqual(["n_max", "every"]);
  });

  it("builds defaults and reads/writes by path", () => {
    expect(schemaDefaults(S, "PicMerge")).toEqual({ n_max: 100000, every: 100 });
    const obj: Record<string, unknown> = { a: { list: [{ x: 1 }] } };
    expect(getIn(obj, ["a", "list", 0, "x"])).toBe(1);
    setIn(obj, ["a", "list", 0, "x"], 2);
    setIn(obj, ["b", "c"], 3);
    setIn(obj, ["d", 0], "z");
    expect(obj).toEqual({ a: { list: [{ x: 2 }] }, b: { c: 3 }, d: ["z"] });
    setIn(obj, ["a", "list", 0], undefined);
    expect((obj.a as { list: unknown[] }).list).toEqual([]);
  });
});

describe("units", () => {
  it("parses values with SI prefixes and aliases into SI", () => {
    expect(parseQuantity("13.56 MHz", "Hz", false, mm)).toEqual({ ok: true, value: 13.56e6 });
    expect(parseQuantity("2ns", "s", false, mm)).toEqual({ ok: true, value: 2e-9 });
    expect(parseQuantity("1e-10", "s", false, mm)).toEqual({ ok: true, value: 1e-10 });
    expect(parseQuantity("5 keV", "eV", false, mm)).toEqual({ ok: true, value: 5000 });
    const torr = parseQuantity("10 mTorr", "Pa", false, mm);
    expect(torr.ok && torr.value).toBeCloseTo(1.33322368);
    const cm3 = parseQuantity("1e10 cm^-3", "m^-3", false, mm);
    expect(cm3.ok && cm3.value).toBeCloseTo(1e16);
    expect(parseQuantity("", "Hz", false, mm)).toEqual({ ok: true, value: null });
    expect(parseQuantity("abc", "Hz", false, mm)).toEqual({ ok: false, error: "number" });
    expect(parseQuantity("5 kV", "Hz", false, mm)).toEqual({ ok: false, error: "unit" });
    expect(parseQuantity("5 cV", "V", false, mm)).toEqual({ ok: false, error: "unit" }); // c はメートルだけ
  });

  it("uses the display length unit for geometric lengths", () => {
    const r = parseQuantity("12.5", "m", true, mm);
    expect(r.ok && r.value).toBeCloseTo(0.0125);
    const um = parseQuantity("250", "m", true, { lengthUnit: "um", axisymmetric: false });
    expect(um.ok && um.value).toBeCloseTo(250e-6);
    const cm = parseQuantity("0.5 cm", "m", true, mm);
    expect(cm.ok && cm.value).toBeCloseTo(0.005);
    expect(formatQuantity(0.0125, true, mm)).toBe("12.5");
    expect(displayUnit("m", true, mm)).toBe("mm");
    expect(displayUnit("m^-3", false, mm)).toBe("m⁻³");
    expect(displayUnit("1", false, mm)).toBe("");
  });

  it("reads per-metre quantities as totals in axisymmetric mode (A/m → A)", () => {
    const axi: UnitContext = { lengthUnit: "mm", axisymmetric: true };
    expect(displayUnit("A/m", false, axi)).toBe("A");
    expect(unitFactor("mA", "A/m", axi)).toBe(1e-3);
    expect(displayUnit("A/m", false, mm)).toBe("A/m");
  });
});

describe("labels", () => {
  it("uses model labels, then common labels, then the key", () => {
    expect(fieldLabel({ model: "Fluid2dSettings", key: "linear_solver" })).toBe("線形ソルバー");
    expect(fieldLabel({ model: "Fluid2dSettings", key: "gas_pressure_pa" })).toBe("ガス圧");
    expect(fieldLabel({ model: "X", key: "unknown_key" })).toBe("unknown_key");
    expect(enumLabel({ model: "MeshSettings", key: "mode" }, "cartesian")).toContain("直交格子");
    expect(enumLabel({ model: "Fluid1dSettings", key: "ion_mobility_model" }, "const")).toBe("一定");
  });

  it("has an English label for every Japanese one", () => {
    const missing: string[] = [];
    const walk = (ja: unknown, en: unknown, path: string) => {
      if (typeof ja === "string") {
        if (typeof en !== "string") missing.push(path);
        return;
      }
      for (const [k, v] of Object.entries(ja as Record<string, unknown>)) walk(v, (en as Record<string, unknown> | undefined)?.[k], `${path}.${k}`);
    };
    walk(fieldsJa, fieldsEn, "");
    expect(missing).toEqual([]);
  });

  it("labels every simple field of the settings models", () => {
    const models = ["Fluid2dSettings", "Fluid1dSettings", "Pic1dSettings", "PicSettings", "DsmcSettings", "TlSettings", "InitialPlasma", "MccGas", "AmrSettings", "Emitter"];
    const unlabeled: string[] = [];
    for (const m of models) {
      for (const [k, p] of Object.entries(S.$defs![m].properties!)) {
        const simple = p.type === "number" || p.type === "integer" || p.type === "boolean" || (p.anyOf ?? []).some((a) => a.type === "number");
        if (simple && fieldLabel({ model: m, key: k }) === k) unlabeled.push(`${m}.${k}`);
      }
    }
    expect(unlabeled).toEqual([]);
  });
});
