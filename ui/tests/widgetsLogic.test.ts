import { produce } from "immer";
import { describe, expect, it } from "vitest";
import { t } from "../src/i18n";
import { edgeBcType, setEdgeType } from "../src/model/boundaryOps";
import { newProject, type Project } from "../src/model/project";
import { buildSweepCandidates, parseListValues, rangeValues, sweepModuleForPath, valueAtPath } from "../src/model/sweep";
import { computeProcessesHash, processesHashJson, pyFloatRepr } from "../src/util/boltzHash";
import { formatSi } from "../src/util/format";
import { dsmcParticlesPerCell, electrodeFreqs, phaseBinAdvice, projectFreqs, stepsPerPeriod } from "../src/util/runHints";
import { evalVoltage, evalWaveform, parseWaveformCsv, rfComponents, rfValue } from "../src/util/waveform";

describe("boundary conditions per edge", () => {
  it("changes types, splits shared entries and pairs periodic edges", () => {
    let p: Project = newProject();
    p = produce(p, (d) => void d.geometry.boundaries.push({ edges: [0, 2], type: "symmetry" }));
    p = produce(p, (d) => void setEdgeType(d, 0, "dirichlet"));
    expect(p.geometry.boundaries.find((b) => b.type === "symmetry")?.edges).toEqual([2]);
    expect(edgeBcType(p, 0)).toBe("dirichlet");
    p = produce(p, (d) => void setEdgeType(d, 0, "periodic"));
    expect(p.geometry.boundaries.find((b) => b.type === "periodic")?.edges).toEqual([0, 2]);
    expect(edgeBcType(p, 2)).toBe("periodic");
    // 周期を外すと向かいの辺も外れる
    p = produce(p, (d) => void setEdgeType(d, 2, "neumann"));
    expect(edgeBcType(p, 0)).toBe("neumann");
    expect(edgeBcType(p, 2)).toBe("neumann");
    // 軸対称の軸の辺は変えられない
    const axi = { ...p, coord: "rz" as const };
    let ok = true;
    produce(axi, (d) => void (ok = setEdgeType(d, 0, "dirichlet")));
    expect(ok).toBe(false);
    expect(edgeBcType(axi, 0)).toBe("axis");
  });
});

describe("waveforms", () => {
  it("parses CSV like v1 (skips headers, sorts, drops the wrap-around row)", () => {
    const r = parseWaveformCsv("t,v\n0 0\n2e-8\t5\n1e-8,10\n");
    expect(r).toEqual({ phase: [0, 0.5], v: [0, 10], freqHz: 5e7 });
    expect(parseWaveformCsv("x\n1,2")).toEqual({ error: "fewRows" });
    expect(parseWaveformCsv("1,2\n1,3")).toEqual({ error: "zeroSpan" });
  });

  it("evaluates DC + RF + waveform exactly like pic.py", () => {
    const wf = { freq_hz: 1, phase: [0, 0.5], v: [0, 10] };
    expect(evalWaveform(wf, 0.25)).toBeCloseTo(5);
    expect(evalWaveform(wf, 0.75)).toBeCloseTo(5); // 0.5→10 と 1.0→0 (折り返し) の中点
    const rf = rfComponents({ amplitude: 100, freq_hz: 1e6, phase_deg: 90 });
    expect(evalVoltage(10, rf, [], 0)).toBeCloseTo(110);
    expect(rfValue([])).toBeNull();
    expect(rfValue(rf)).toEqual(rf[0]);
    expect(rfValue([...rf, ...rf])).toHaveLength(2);
  });
});

describe("run hints", () => {
  it("counts steps per RF period and advises phase bins", () => {
    const p = newProject();
    p.geometry.boundaries[1].voltage_rf = [
      { amplitude: 100, freq_hz: 13.56e6, phase_deg: 0 },
      { amplitude: 50, freq_hz: 2e6, phase_deg: 0 },
    ];
    expect(projectFreqs(p, false)).toEqual([2e6, 13.56e6]);
    expect(stepsPerPeriod(2e6, 1e-9)).toBeCloseTo(500);
    const a = phaseBinAdvice([2e6], 1e-9, 600, 100, 2000)!;
    expect(a).toMatchObject({ steps: 500, binsTooMany: true, avgTooShort: true, recommendedAvg: 1500 });
    expect(phaseBinAdvice([2e6], null, 40, null, 2000)).toBeNull();
    expect(electrodeFreqs({ waveforms: [{ freq_hz: 1e5, phase: [], v: [] }] }, {})).toEqual([1e5]);
    expect(dsmcParticlesPerCell(p, 50000, 1, 1000)).toBe(50);
  });

  it("formats with SI prefixes", () => {
    expect(formatSi(13.56e6, "Hz")).toBe("13.56 MHz");
    expect(formatSi(2e-9, "s")).toBe("2 ns");
    expect(formatSi(0, "V")).toBe("0 V");
  });
});

describe("parameter sweep", () => {
  it("lists candidates like v1 and resolves the module", () => {
    const p = newProject();
    p.geometry.boundaries[1].voltage_rf = { amplitude: 100, freq_hz: 13.56e6, phase_deg: 0 };
    p.pic1d = { left: { voltage_rf: null }, right: { voltage_rf: { amplitude: 1, freq_hz: 1, phase_deg: 0 } }, mcc: { gas: {} } };
    const paths = buildSweepCandidates(p, t).map((c) => c.path);
    expect(paths).toContain("geometry.boundaries.1.voltage_rf.freq_hz");
    expect(paths).toContain("pic1d.right.voltage_rf.0.amplitude");
    expect(paths).not.toContain("pic1d.left.voltage_rf.0.amplitude");
    expect(paths).toContain("pic1d.mcc.gas.pressure_pa");
    expect(sweepModuleForPath("fluid2d.gas_pressure_pa")).toBe("fluid2d");
    expect(sweepModuleForPath("geometry.boundaries.0.voltage")).toBe("pic");
    expect(valueAtPath(p, "geometry.boundaries.1.voltage")).toBe(100);
    expect(valueAtPath(p, "geometry.boundaries.9.voltage")).toBeUndefined();
  });

  it("builds value lists and ranges", () => {
    expect(parseListValues("50, 1e2, x, ,150")).toEqual([50, 100, 150]);
    expect(rangeValues(0, 10, 3, false)).toEqual([0, 5, 10]);
    const log = rangeValues(1, 100, 3, true);
    expect(log[1]).toBeCloseTo(10);
    expect(rangeValues(-1, 10, 3, true)).toEqual([]);
    expect(rangeValues(7, 9, 1, false)).toEqual([7]);
  });
});

describe("Boltzmann table hash (bit-exact with Python json.dumps + sha256)", () => {
  it("formats floats like Python's repr", () => {
    expect(pyFloatRepr(3)).toBe("3.0");
    expect(pyFloatRepr(1e-5)).toBe("1e-05");
    expect(pyFloatRepr(0.0001)).toBe("0.0001");
    expect(pyFloatRepr(1.36e-5)).toBe("1.36e-05");
    expect(pyFloatRepr(12345678901234567)).toBe("1.2345678901234568e+16");
    expect(pyFloatRepr(-2.5e-19)).toBe("-2.5e-19");
  });

  it("matches the backend's _hash_processes", async () => {
    const procs = [
      { kind: "elastic", label: "Ar elastic (Phelps)", threshold_ev: 0, mass_ratio: 1.36e-5, energy_ev: [0, 1, 15.76, 1e-5, 12345678901234567], sigma_m2: [1e-20, 2.5e-19, 3, 0.0001, 7e-21] },
      { kind: "ionization", label: "Ar → Ar⁺ 電離", threshold_ev: 15.76, mass_ratio: 0, energy_ev: [15.76, 100], sigma_m2: [0, 2.8e-20] },
    ];
    expect(processesHashJson(procs)).toContain('"label": "Ar \\u2192 Ar\\u207a \\u96fb\\u96e2"');
    // backend: es_sim.boltz._hash_processes (同じ入力で計算した値)
    expect(await computeProcessesHash(procs)).toBe("e2be0cef256ff7c3a0fd2ffb09a1f8fa513e52f04ac9e3f0e2266fa7511bf904");
  });
});
