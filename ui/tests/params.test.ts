/// <reference types="node" />
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { produce } from "immer";
import { beforeEach, describe, expect, it } from "vitest";
import { evaluateExpr, evaluateParams, ExprError, nameProblem, renameInExpr } from "../src/model/expr";
import { useDocument } from "../src/model/documentStore";
import {
  addParam,
  addressOf,
  bindingAt,
  deleteParam,
  paramsOf,
  paramUsers,
  paramValues,
  renameParam,
  resolveAddress,
  setBinding,
  setParamExpr,
  stageBinding,
} from "../src/model/params";
import { newProject, normalizeProject, type Project } from "../src/model/project";
import { renameRegion, splitRegionEdge } from "../src/model/regionOps";
import { buildSweepCandidates, prepareSweepProject, sweepModule, valueAtPath } from "../src/model/sweep";
import { parseQuantity } from "../src/schema/units";
import { t } from "../src/i18n";

interface Cases {
  expressions: { expr: string; vars?: Record<string, number>; value?: number; error?: string }[];
  params: { vars: { name: string; expr: string }[]; overrides?: Record<string, number>; values?: Record<string, number>; error?: string }[];
  names: { name: string; ok: boolean }[];
}

// backend と共通の試験データ
// (vitest は ui/ で動く)
const CASES = JSON.parse(readFileSync(resolve(process.cwd(), "../backend/tests/data/param_expr_cases.json"), "utf-8")) as Cases;

const kindOf = (f: () => unknown): string | null => {
  try {
    f();
    return null;
  } catch (e) {
    return e instanceof ExprError ? e.kind : "other";
  }
};

describe("expressions shared with the backend", () => {
  it("evaluates every shared expression like params.py", () => {
    for (const c of CASES.expressions) {
      if (c.error) expect([c.expr, kindOf(() => evaluateExpr(c.expr, c.vars))]).toEqual([c.expr, c.error]);
      else {
        const v = evaluateExpr(c.expr, c.vars);
        expect(Math.abs(v - c.value!) <= 1e-12 * Math.abs(c.value!) + 1e-300, `${c.expr}: ${v}`).toBe(true);
      }
    }
  });

  it("orders parameters by their dependencies and finds cycles", () => {
    for (const c of CASES.params) {
      if (c.error) expect(kindOf(() => evaluateParams(c.vars, c.overrides))).toBe(c.error);
      else {
        const got = evaluateParams(c.vars, c.overrides);
        for (const [k, v] of Object.entries(c.values!)) expect(got[k]).toBeCloseTo(v, 12);
      }
    }
  });

  it("checks names", () => {
    for (const c of CASES.names) expect([c.name, nameProblem(c.name) === null]).toEqual([c.name, c.ok]);
  });

  it("renames a parameter inside expressions but not units", () => {
    expect(renameInExpr("2*gap + gap^2", "gap", "g")).toBe("2*g + g^2");
    // m は単位の位置では変えない
    expect(renameInExpr("m*2 + 3 m", "m", "k")).toBe("k*2 + 3 m");
    expect(renameInExpr("gapx + gap", "gap", "g")).toBe("gapx + g");
  });
});

function withParams(): Project {
  return produce(newProject(), (d) => {
    d.geometry.regions.push({ id: "a", type: "conductor", voltage: 5, polygon: [[0.01, 0.01], [0.02, 0.01], [0.02, 0.02], [0.01, 0.02]] });
    d.params = { vars: [{ name: "gap", expr: "2 mm" }, { name: "V0", expr: "100" }], bindings: [] };
  });
}

describe("bindings in the document", () => {
  beforeEach(() => useDocument.getState().replace(withParams(), null));

  it("addresses regions by id and boundaries by edge id", () => {
    const p = withParams();
    expect(addressOf(p, ["geometry", "regions", 0, "voltage"])).toEqual(["geometry", "regions", { id: "a" }, "voltage"]);
    expect(addressOf(p, ["geometry", "boundaries", 1, "voltage"])).toEqual(["geometry", "boundaries", { edge: "e2" }, "voltage"]);
    expect(addressOf(p, ["geometry", "domain", "polygon", 1, 0])).toEqual(["geometry", "domain", "polygon", 1, 0]);
    expect(resolveAddress(p, ["geometry", "boundaries", { edge: "e2" }, "voltage"])).toEqual(["geometry", "boundaries", 1, "voltage"]);
    expect(resolveAddress(p, ["geometry", "regions", { id: "zz" }, "voltage"])).toBeNull();
    expect(paramValues(p).values).toEqual({ gap: 0.002, V0: 100 });
    // 同じ params には同じ結果 (セレクタで毎回新しいオブジェクトを返さない)
    expect(paramValues(p)).toBe(paramValues(p));
  });

  it("binds a field through the staged binding and recomputes it when the parameter changes", () => {
    const st = useDocument.getState();
    stageBinding(["geometry", "regions", 0, "voltage"], "V0/2");
    st.update("v", (d) => void (d.geometry.regions[0].voltage = 50));
    let p = useDocument.getState().project;
    expect(bindingAt(p, ["geometry", "regions", 0, "voltage"])).toBe("V0/2");
    expect(useDocument.getState().past).toHaveLength(1);
    st.update("expr", (d) => setParamExpr(d, "V0", "300"));
    p = useDocument.getState().project;
    expect(p.geometry.regions[0].voltage).toBe(150);
    expect(paramsOf(p).vars[1].value).toBe(300);
    // 元に戻すと式と値が一緒に戻る
    useDocument.getState().undo();
    expect(useDocument.getState().project.geometry.regions[0].voltage).toBe(50);
    // 名前の変更は式の中も
    st.update("rename", (d) => renameParam(d, "V0", "Vd"));
    expect(bindingAt(useDocument.getState().project, ["geometry", "regions", 0, "voltage"])).toBe("Vd/2");
  });

  it("drops a binding when the field is changed by hand, the element disappears or vertices shift", () => {
    const st = useDocument.getState();
    st.update("bind", (d) => {
      setBinding(d, ["geometry", "regions", 0, "voltage"], "V0");
      setBinding(d, ["geometry", "regions", 0, "polygon", 2, 0], "0.01 + gap*5");
      setBinding(d, ["geometry", "boundaries", 1, "voltage"], "V0*2");
    });
    let p = useDocument.getState().project;
    expect(p.geometry.regions[0].voltage).toBe(100);
    expect(p.geometry.regions[0].polygon![2][0]).toBeCloseTo(0.02, 15);
    expect(p.geometry.boundaries[1].voltage).toBe(200);
    expect(paramsOf(p).bindings).toHaveLength(3);
    // 手で変えた欄の束縛は外れる
    st.update("hand", (d) => void (d.geometry.boundaries[1].voltage = 7));
    expect(paramsOf(useDocument.getState().project).bindings).toHaveLength(2);
    // 名前を変えても領域の束縛は残る
    st.update("rename", (d) => renameRegion(d, "a", "anode"));
    p = useDocument.getState().project;
    expect(bindingAt(p, ["geometry", "regions", 0, "voltage"])).toBe("V0");
    // 頂点を足すと (番号がずれる) その形の頂点の束縛は外れる
    st.update("split", (d) => void splitRegionEdge(d, "anode", 0));
    p = useDocument.getState().project;
    expect(bindingAt(p, ["geometry", "regions", 0, "polygon", 2, 0])).toBeNull();
    expect(paramsOf(p).bindings).toHaveLength(1);
    // 領域を消すと外れる
    st.update("del", (d) => void (d.geometry.regions = []));
    expect(paramsOf(useDocument.getState().project).bindings).toHaveLength(0);
  });

  it("does not bind when the staged value did not reach the document", () => {
    stageBinding(["geometry", "regions", 0, "voltage"], "V0");
    // 別の欄を変えただけ (電圧は 5 のまま) なら付けない
    useDocument.getState().update("x", (d) => void (d.mesh.size = 0.003));
    expect(bindingAt(useDocument.getState().project, ["geometry", "regions", 0, "voltage"])).toBeNull();
  });

  it("applies parameters when a document is opened", () => {
    const raw = JSON.parse(JSON.stringify(withParams()));
    raw.params.vars[1].expr = "40";
    raw.params.bindings = [
      { path: ["geometry", "regions", { id: "a" }, "voltage"], expr: "V0" },
      { path: ["geometry", "regions", { id: "gone" }, "voltage"], expr: "V0" },
    ];
    const { project } = normalizeProject(raw);
    expect(project.geometry.regions[0].voltage).toBe(40);
    expect(paramsOf(project).bindings).toHaveLength(1);
    expect(paramsOf(project).vars[1].value).toBe(40);
  });

  it("adds, uses and deletes parameters", () => {
    let p = produce(withParams(), (d) => {
      addParam(d, "w", "2*gap");
      setBinding(d, ["mesh", "size"], "w/4");
    });
    expect(p.mesh.size).toBeCloseTo(0.001, 15);
    expect(paramUsers(p, "gap")).toEqual({ vars: ["w"], bindings: 0 });
    expect(paramUsers(p, "w")).toEqual({ vars: [], bindings: 1 });
    let ok = true;
    p = produce(p, (d) => void (ok = deleteParam(d, "gap")));
    expect(ok).toBe(false);
    p = produce(p, (d) => void deleteParam(d, "w"));
    expect(paramsOf(p).bindings).toHaveLength(0);
    expect(p.mesh.size).toBeCloseTo(0.001, 15);
  });
});

describe("fields and sweeps", () => {
  it("reads expressions with parameters in SI and marks them for binding", () => {
    const ctx = { lengthUnit: "mm" as const, axisymmetric: false, vars: { gap: 0.002 } };
    expect(parseQuantity("gap/2", undefined, true, ctx)).toEqual({ ok: true, value: 0.001, expr: "gap/2" });
    expect(parseQuantity("gap + 1 mm", undefined, true, ctx)).toEqual({ ok: true, value: 0.003, expr: "gap + 1 mm" });
    // 数だけは表示の単位 (mm) のまま、単位付き・関数も今までどおり
    expect(parseQuantity("3", undefined, true, ctx)).toEqual({ ok: true, value: 0.003 });
    expect(parseQuantity("13.56 MHz", "Hz", false, ctx)).toEqual({ ok: true, value: 13.56e6 });
    expect(parseQuantity("sqrt(4)", "V", false, ctx)).toEqual({ ok: true, value: 2 });
    const bad = parseQuantity("gapp*2", undefined, true, ctx);
    expect(bad.ok).toBe(false);
    expect(!bad.ok && bad.error).toBe("param");
    expect(!bad.ok && bad.message).toContain("gapp");
  });

  it("offers parameters as sweep targets", () => {
    const p = produce(withParams(), (d) => {
      setBinding(d, ["geometry", "regions", 0, "voltage"], "V0");
    });
    const cands = buildSweepCandidates(p, t);
    expect(cands[0]).toEqual({ label: "パラメータ gap", path: "params.gap" });
    expect(valueAtPath(p, "params.V0")).toBe(100);
    expect(valueAtPath(p, "params.nothing")).toBeUndefined();
    expect(prepareSweepProject(p, "params.V0")).toEqual(p);
    expect(sweepModule({ param_path: "params.V0" }, p)).toBe("pic");
    expect(sweepModule({ param_path: "params.V0", module: "fluid2d" }, p)).toBe("fluid2d");
    const one = produce(p, (d) => {
      d.pic1d = { gap_m: 0.02 };
      setBinding(d, ["pic1d", "gap_m"], "gap*10");
    });
    // 式を付けた欄に 1D・流体のブロックがあればそれ
    expect(sweepModule({ param_path: "params.gap" }, one)).toBe("pic1d");
    expect(sweepModule({ param_path: "pic1d.gap_m" }, p)).toBe("pic1d");
  });
});
