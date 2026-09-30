import { describe, expect, it } from "vitest";
import { t } from "../src/i18n";
import { EXAMPLES } from "../src/io/examples";
import { newProject, normalizeProject, type Project } from "../src/model/project";
import { buildTree, edgeSummary, filterTree, visibleNodes, type TreeNode } from "../src/tree/treeModel";

const find = (n: TreeNode, id: string): TreeNode | undefined =>
  n.id === id ? n : (n.children ?? []).map((c) => find(c, id)).find(Boolean);

describe("model tree", () => {
  it("summarises the geometry, boundaries and studies", () => {
    const p = newProject();
    p.geometry.boundaries[1].voltage_rf = { amplitude: 100, freq_hz: 13.56e6 };
    p.pic = { n_macro: 1000 };
    const root = buildTree(p, t, "mm", "demo");
    expect(root.label).toBe("demo");
    expect(find(root, "domain")?.detail).toBe("100 × 50 mm");
    expect(find(root, "edge:e4")?.label).toBe("左 (x=0)");
    expect(find(root, "edge:e4")?.detail).toBe("Dirichlet 0 V");
    expect(find(root, "edge:e2")?.detail).toBe("Dirichlet 100 V + RF");
    expect(find(root, "edge:e1")?.detail).toBe("なし");
    expect(find(root, "mesh")?.detail).toBe("4 mm · unstructured");
    expect(find(root, "study:pic")?.badge?.text).toBe("設定済み");
    expect(find(root, "study:fluid2d")?.badge?.text).toBe("未設定");
    expect(find(root, "study:fem")?.badge).toBeUndefined();
    expect(find(root, "regions.empty")?.placeholder).toBe(true);
    // 1e5 以上は指数表記 (v1 と同じ規則)
    expect(find(buildTree(p, t, "um", "demo"), "domain")?.detail).toBe("1e5 × 50000 µm");
  });

  it("labels axisymmetric edges and polygon domains", () => {
    const p: Project = { ...newProject(), coord: "rz" };
    expect(edgeSummary(p, 0, t)).toBe("対称軸");
    expect(find(buildTree(p, t, "mm", "x"), "edge:e3")?.label).toBe("上 (r=R)");
    const coax = normalizeProject(EXAMPLES.find((e) => e.key === "coaxial")!.data).project;
    const root = buildTree(coax, t, "mm", "coax");
    const edges = find(root, "boundaries")!.children!;
    expect(edges).toHaveLength(coax.geometry.domain.polygon.length);
    expect(edges[5].label).toBe("辺 5");
  });

  it("filters by search text and keeps the ancestors", () => {
    const p = newProject();
    p.geometry.regions.push({ id: "anode", type: "conductor", polygon: [[0, 0], [1, 0], [1, 1]] });
    const root = buildTree(p, t, "mm", "demo");
    const f = filterTree(root, "ANODE")!;
    const ids = visibleNodes(f, new Set(), true).map((r) => r.node.id);
    expect(ids).toEqual(["project", "geometry", "regions", "region:anode"]);
    expect(filterTree(root, "zzz")).toBeNull();
    expect(filterTree(root, "  ")).toBe(root);
  });

  it("lists only the open branches for keyboard navigation", () => {
    const root = buildTree(newProject(), t, "mm", "demo");
    const rows = visibleNodes(root, new Set(["project", "geometry"]));
    expect(rows.map((r) => r.node.id)).toEqual(["project", "params", "geometry", "domain", "regions", "sketch", "layers", "boundaries", "mesh", "bfield", "studies", "results"]);
    expect(rows.find((r) => r.node.id === "domain")?.parent).toBe("geometry");
    expect(rows.find((r) => r.node.id === "domain")?.level).toBe(3);
  });
});
