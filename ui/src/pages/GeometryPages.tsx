// ジオメトリの設定ページ: プロジェクト・ドメイン・領域 (一覧と各領域)・境界条件 (一覧と各辺)・磁場。

import type { TFunction } from "i18next";
import type { Draft } from "immer";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { askConfirm } from "../app/dialogs";
import { segFromBulge, segLength, type ArcSeg } from "../cad/geom";
import type { BoolOp } from "../cad/boolean";
import { bulgesOf, moveVertex, removeVertex, setBulge, splitEdge, type PathData } from "../cad/path";
import { SchemaField, Toggle } from "../forms/SchemaField";
import { useViewer } from "../graphics/viewerStore";
import { applyRegionBoolean, type BooleanOutcome } from "../model/booleanOps";
import { setValue } from "../forms/useField";
import { edgeBcType, periodicPartnerOf, setEdgeType, boundaryIndexOfEdge, type EdgeBcType } from "../model/boundaryOps";
import { documentName, useDocument, useIsDirty } from "../model/documentStore";
import {
  applyDomainEdit,
  moveDomainVertex,
  periodicPartners,
  removeDomainVertex,
  reshapeDomain,
  setDomainBulge,
  splitDomainEdge,
  type DomainEdit,
} from "../model/domainOps";
import { regionToDomain } from "../model/cadActions";
import {
  axisEdges,
  coordOf,
  domainBounds,
  domainPath,
  edgeIdsOf,
  edgeIndexOf,
  isRectDomain,
  regionHoles,
  regionPath,
  type Coord,
  type Project,
  type Region,
  type RegionType,
} from "../model/project";
import {
  addCircleRegion,
  addRectRegion,
  deleteRegion,
  removeRegionHole,
  removeRegionVertex,
  renameRegion,
  setRegionHole,
  setRegionPath,
  setRegionType,
  splitRegionEdge,
  validateRegionId,
} from "../model/regionOps";
import { pickedRegionIds, useSelection } from "../model/selection";
import { usePrefs } from "../prefs/prefs";
import { edgeLabel, edgeSummary } from "../tree/treeModel";
import { formatNumber, lengthUnitLabel, toDisplayLength, type LengthUnit } from "../util/format";
import { rfComponents, type VoltageWaveform } from "../util/waveform";
import { CommitText, Field, LengthInput, Select } from "./inputs";
import { useLengthUnitLabel } from "./useLengthUnitLabel";
import { CapacitorEditor } from "./widgets/CapacitorEditor";
import { DefQuantity, Hint } from "./widgets/common";
import { LayerSelect } from "./LayersPage";
import { PathTable, type PathTableOps } from "./widgets/PathTable";
import { TransformPanel } from "./widgets/TransformPanel";
import { RfEditor } from "./widgets/RfEditor";
import { VoltagePreview } from "./widgets/VoltagePreview";
import { WaveformEditor } from "./widgets/WaveformEditor";
import { logInfo, logWarning } from "../app/messages";

const useUpdate = () => useDocument((s) => s.update);

/** 外周を変える (辺の番号の参照も付け替える)。外れた境界条件があれば知らせる */
export function editDomain(t: TFunction, label: string, build: (p: Project) => DomainEdit | null): void {
  let removed = 0;
  useDocument.getState().update(label, (d) => {
    const e = build(d as Project);
    if (!e) return;
    const r = applyDomainEdit(d, e);
    removed = r.boundariesRemoved + r.periodicRemoved;
  });
  if (removed > 0) logInfo(t("msg.source.app"), t("settings.bcRemoved", { n: removed }));
}

/** 対称軸 (軸対称で r = 0 の辺) の境界条件を外す。周期境界は対の片方だけ残すと backend が拒むのでまるごと外す */
function removeAxisBoundaries(d: Draft<Project>): boolean {
  const axis = axisEdges(d as Project);
  if (axis.length === 0) return false;
  const before = JSON.stringify(d.geometry.boundaries);
  d.geometry.boundaries = d.geometry.boundaries.filter((b) => !(b.type === "periodic" && b.edges.some((e) => axis.includes(e))));
  for (const b of d.geometry.boundaries) b.edges = b.edges.filter((e) => !axis.includes(e));
  d.geometry.boundaries = d.geometry.boundaries.filter((b) => b.edges.length > 0);
  return JSON.stringify(d.geometry.boundaries) !== before;
}

export function CoordSelect() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const update = useUpdate();
  const coords: Coord[] = ["xy", "rz", "rz_x0"];
  return (
    <Field label={t("coord.label")}>
      {(id) => (
        <Select<Coord>
          id={id}
          value={coordOf(project)}
          options={coords.map((c) => ({ value: c, label: t(`coord.${c}`) }))}
          onChange={(c) =>
          {
            let removed = false;
            update(t("action.coord"), (d) => {
              d.coord = c;
              if (c === "xy") return;
              // 軸対称に切り替えたら対称軸 (r = 0 の上の辺) の境界条件を外す (v1 と同じ)
              removed = removeAxisBoundaries(d);
              // 磁場は平面 2D だけ (軸対称では backend が拒む)
              if (d.b_field) d.b_field = null;
            });
            if (removed) logInfo(t("msg.source.app"), t("settings.axisBcRemoved"));
          }
          }
        />
      )}
    </Field>
  );
}

export function ProjectPage() {
  const { t } = useTranslation();
  const name = useDocument((s) => documentName(s, t("app.untitled")));
  const file = useDocument((s) => s.file);
  const dirty = useIsDirty();
  const unit = usePrefs((s) => s.lengthUnit);
  const setUnit = usePrefs((s) => s.setLengthUnit);
  return (
    <>
      <div className="kv">
        <span>{t("settings.file")}</span>
        <span>{name}</span>
        <span>{t("settings.path")}</span>
        <span className="mono">{file?.path ?? (file ? file.name : t("settings.notSaved"))}</span>
        <span>{t("settings.state")}</span>
        <span className={dirty ? "text-warn" : undefined}>{dirty ? t("settings.stateDirty") : t("settings.stateSaved")}</span>
      </div>
      <CoordSelect />
      <Field label={t("menu.lengthUnit")}>
        {(id) => (
          <Select<LengthUnit>
            id={id}
            value={unit}
            options={[
              { value: "mm", label: "mm" },
              { value: "um", label: "µm" },
            ]}
            onChange={setUnit}
          />
        )}
      </Field>
    </>
  );
}

export function DomainPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const u = useLengthUnitLabel();
  const coord = coordOf(project);
  const b = domainBounds(project);
  const rect = isRectDomain(project);
  const mode = project.mesh.mode ?? "unstructured";
  const setSize = (w: number, h: number) =>
    editDomain(t, t("action.domainSize"), (p) =>
      reshapeDomain(p, [
        [b.x0, b.y0],
        [b.x0 + w, b.y0],
        [b.x0 + w, b.y0 + h],
        [b.x0, b.y0 + h],
      ]),
    );
  const ops: PathTableOps = {
    setVertex: (i, q) => editDomain(t, t("action.domainShape"), (p) => moveDomainVertex(p, i, q)),
    setBulge: (i, bulge) => editDomain(t, t("action.domainShape"), (p) => setDomainBulge(p, i, bulge)),
    split: (i) => editDomain(t, t("action.splitEdge"), (p) => splitDomainEdge(p, i)),
    removeVertex: (i) => editDomain(t, t("action.domainShape"), (p) => removeDomainVertex(p, i)),
  };
  const wLabel = coord === "rz" ? t("settings.widthZ") : coord === "rz_x0" ? t("settings.widthR") : t("settings.width");
  const hLabel = coord === "rz" ? t("settings.heightR") : coord === "rz_x0" ? t("settings.heightZ") : t("settings.height");
  return (
    <>
      <CoordSelect />
      {coord !== "xy" && <Hint>{t("settings.axisHint")}</Hint>}
      {rect ? (
        <>
          <Field label={wLabel} unit={u}>
            {(id, onError) => <LengthInput id={id} onError={onError} value={b.x1 - b.x0} min={0} exclusive onCommit={(w) => setSize(w, b.y1 - b.y0)} />}
          </Field>
          <Field label={hLabel} unit={u}>
            {(id, onError) => <LengthInput id={id} onError={onError} value={b.y1 - b.y0} min={0} exclusive onCommit={(h) => setSize(b.x1 - b.x0, h)} />}
          </Field>
        </>
      ) : (
        <p className="hint">{t("settings.nonRectDomain", { n: project.geometry.domain.polygon.length })}</p>
      )}
      {!rect && mode !== "unstructured" && <Hint tone="warn">{t("settings.gridNeedsRect", { mode })}</Hint>}
      {/* 矩形でも頂点を足す・辺を円弧にすると任意の形にできる (矩形のうちは閉じておく) */}
      <details className="subsection" open={!rect}>
        <summary className="subsection-title">{t("settings.domainVertices")}</summary>
        <PathTable path={domainPath(project)} label={t("tree.domain")} ops={ops} docPath={["geometry", "domain", "polygon"]} />
      </details>
    </>
  );
}

export function RegionsPage() {
  const { t } = useTranslation();
  const regions = useDocument((s) => s.project.geometry.regions);
  const update = useUpdate();
  const select = useSelection((s) => s.select);
  const add = (kind: "rect" | "circle") => {
    let id = "";
    update(t("action.addRegion"), (d) => {
      id = kind === "rect" ? addRectRegion(d) : addCircleRegion(d);
    });
    select(`region:${id}`);
  };
  return (
    <>
      <p className="muted">{t("settings.regionCount", { n: regions.length })}</p>
      <ul className="list">
        {regions.map((r) => (
          <li key={r.id}>
            <button type="button" className="link" onClick={() => select(`region:${r.id}`)}>
              {r.id}
            </button>{" "}
            <span className="muted">
              {t(`region.${r.type}`)} · {r.shape ? t("region.circle") : t("region.polygon")}
              {r.holes?.length ? ` · ${t("region.holes", { n: r.holes.length })}` : ""}
            </span>
          </li>
        ))}
      </ul>
      <div className="button-row">
        <button type="button" className="button" onClick={() => add("rect")}>
          {t("tree.addRectangle")}
        </button>
        <button type="button" className="button" onClick={() => add("circle")}>
          {t("tree.addCircle")}
        </button>
      </div>
      <Hint>{t("settings.regionsCanvasHint")}</Hint>
    </>
  );
}

/**
 * ブーリアン (和・差・積): この領域と相手 (キャンバスで一緒に選んだ領域、無ければ選ぶ 1 つ)。結果はこの領域
 * (値はそのまま)、分かれたらほかは同じ値の新しい領域
 */
function RegionBoolean({ region }: { region: Region }) {
  const { t } = useTranslation();
  const regions = useDocument((s) => s.project.geometry.regions);
  const picked = useSelection((s) => s.picked);
  const keep = useViewer((s) => s.booleanKeepTools);
  const setKeep = useViewer((s) => s.setBooleanKeepTools);
  const [other, setOther] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const others = regions.filter((r) => r.id !== region.id);
  if (others.length === 0) return null;
  const pickedOthers = pickedRegionIds(picked).filter((id) => id !== region.id && others.some((r) => r.id === id));
  const tools = pickedOthers.length > 0 ? pickedOthers : [(others.find((r) => r.id === other) ?? others[0]).id];
  const run = (op: BoolOp) => {
    let out: BooleanOutcome = { ok: false, error: "failed" };
    useDocument.getState().update(t("boolean.action", { op: t(`boolean.${op}`) }), (d) => void (out = applyRegionBoolean(d, op, region.id, tools, keep)));
    const o = out as BooleanOutcome;
    if (!o.ok) {
      setError(t(`boolean.error.${o.error}`));
      return;
    }
    setError(null);
    if (o.ids.length > 1) logInfo(t("msg.source.app"), t("boolean.pieces", { n: o.ids.length, ids: o.ids.slice(1).join(", ") }));
    useSelection.getState().selectRegion(region.id);
  };
  return (
    <div className="subsection">
      <div className="subsection-title">{t("boolean.title")}</div>
      {pickedOthers.length > 0 ? (
        <p className="muted">{t("boolean.withPicked", { ids: pickedOthers.join(", ") })}</p>
      ) : (
        <Field label={t("boolean.with")}>
          {(fid) => (
            <select
              id={fid}
              className="input"
              value={tools[0]}
              onChange={(e) => {
                setOther(e.target.value);
                setError(null);
              }}
            >
              {others.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id}
                </option>
              ))}
            </select>
          )}
        </Field>
      )}
      <div className="button-row">
        {(["union", "difference", "intersection"] as const).map((op) => (
          <button key={op} type="button" className="button" title={t(`boolean.${op}Title`)} onClick={() => run(op)}>
            {t(`boolean.${op}`)}
          </button>
        ))}
        <Toggle checked={keep} onChange={setKeep} label={t("boolean.keepTools")} />
      </div>
      {error && <Hint tone="error">{error}</Hint>}
      <Hint>{t("boolean.hint")}</Hint>
    </div>
  );
}

/** 穴の頂点の表の操作 (外周の外に出る・ほかの穴と交わる変更はしないで知らせる) */
function holePathOps(id: string, k: number, t: TFunction): PathTableOps {
  const edit = (next: (p: PathData) => PathData | null) => {
    let ok = false;
    useDocument.getState().update(t("action.regionShape"), (d) => {
      const r = d.geometry.regions.find((x) => x.id === id) as Region | undefined;
      const cur = r ? regionHoles(r)[k] : undefined;
      const path = cur ? next(cur) : null;
      ok = path !== null && setRegionHole(d, id, k, path);
    });
    if (!ok) logWarning(t("msg.source.app"), t("holes.invalid"));
  };
  return {
    setVertex: (i, q) => edit((p) => moveVertex(p, i, q)),
    setBulge: (i, bulge) => edit((p) => setBulge(p, i, bulge)),
    split: (i) => edit((p) => splitEdge(p, i)),
    removeVertex: (i) => edit((p) => removeVertex(p, i)),
  };
}

function RegionHoles({ region, index }: { region: Region; index: number }) {
  const { t } = useTranslation();
  const update = useUpdate();
  return (
    <>
      {regionHoles(region).map((h, k) => {
        const label = t("holes.title", { n: k + 1 });
        return (
          <div key={k} className="subsection">
            <div className="subsection-title">{label}</div>
            <PathTable path={h} label={label} ops={holePathOps(region.id, k, t)} docPath={["geometry", "regions", index, "holes", k, "polygon"]} />
            <div className="button-row">
              <button type="button" className="button small" onClick={() => update(t("holes.fill"), (d) => removeRegionHole(d, region.id, k))}>
                {t("holes.fill")}
              </button>
            </div>
          </div>
        );
      })}
    </>
  );
}

function RegionLocalSize({ id }: { id: string }) {
  const { t } = useTranslation();
  const sizes = (useDocument((s) => s.project.mesh.local_sizes) as { region: string; size: number }[] | undefined) ?? [];
  const mode = useDocument((s) => s.project.mesh.mode ?? "unstructured");
  const cur = sizes.find((l) => l.region === id)?.size ?? 0;
  return (
    <>
      <DefQuantity
        def="LocalSize"
        prop="size"
        value={cur}
        label={t("settings.localMeshSize")}
        onCommit={(v) => {
          const rest = sizes.filter((l) => l.region !== id);
          setValue(["mesh", "local_sizes"], v && v > 0 ? [...rest, { region: id, size: v }] : rest, t("settings.localMeshSize"));
        }}
      />
      <Hint>{mode === "unstructured" ? t("settings.localMeshSizeHint") : t("settings.localMeshSizeIgnored")}</Hint>
    </>
  );
}

/** 領域の頂点の表の操作 (円弧を保つ。穴が外周からはみ出す変更はしないで知らせる) */
function regionPathOps(id: string, t: TFunction): PathTableOps {
  const edit = (recipe: (d: Draft<Project>, r: Region) => boolean | void) => {
    let ok = true;
    let holes = false;
    useDocument.getState().update(t("action.regionShape"), (d) => {
      const r = d.geometry.regions.find((x) => x.id === id);
      if (!r) return;
      holes = (r.holes?.length ?? 0) > 0;
      ok = recipe(d, r as Region) !== false;
    });
    if (!ok && holes) logWarning(t("msg.source.app"), t("holes.invalid"));
  };
  return {
    setVertex: (i, q) => edit((d, r) => setRegionPath(d, id, moveVertex(regionPath(r), i, q))),
    setBulge: (i, bulge) => edit((d, r) => setRegionPath(d, id, setBulge(regionPath(r), i, bulge))),
    split: (i) => edit((d) => void splitRegionEdge(d, id, i)),
    removeVertex: (i) => edit((d) => void removeRegionVertex(d, id, i)),
  };
}

export function RegionPage({ id }: { id: string }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const update = useUpdate();
  const select = useSelection((s) => s.select);
  const idx = project.geometry.regions.findIndex((x) => x.id === id);
  const r = project.geometry.regions[idx];
  if (!r) return <p className="muted">{t("settings.selectRegion")}</p>;
  const base = ["geometry", "regions", idx] as const;
  const types: RegionType[] = ["conductor", "dielectric", "charge"];
  return (
    <>
      <Field label={t("settings.id")}>
        {(fid, onError) => (
          <CommitText
            id={fid}
            value={r.id}
            onError={onError}
            validate={(v) => {
              const k = validateRegionId(project, r.id, v);
              return k ? t(k, { id: v.trim() }) : null;
            }}
            onCommit={(v) => {
              const to = v.trim();
              update(t("action.renameRegion"), (d) => renameRegion(d, r.id, to));
              select(`region:${to}`);
            }}
          />
        )}
      </Field>
      <Field label={t("settings.type")}>
        {(fid) => (
          <Select<RegionType>
            id={fid}
            value={r.type}
            options={types.map((ty) => ({ value: ty, label: t(`region.${ty}`) }))}
            onChange={(ty) => update(t("action.regionType"), (d) => setRegionType(d, r.id, ty))}
          />
        )}
      </Field>
      <LayerSelect item={{ kind: "region", id: r.id }} />
      {r.type === "conductor" && (
        <>
          <SchemaField path={[...base, "voltage"]} required />
          {r.blocking_capacitor && <Hint>{t("capacitor.dcBlocked")}</Hint>}
          <RfEditor path={[...base, "voltage_rf"]} />
          <div className="subsection-title">{t("widgets.waveform")}</div>
          <WaveformEditor path={[...base, "voltage_waveform"]} />
          <VoltagePreview
            dc={r.voltage ?? 0}
            rf={rfComponents(r.voltage_rf)}
            waveforms={r.voltage_waveform ? [r.voltage_waveform as VoltageWaveform] : []}
          />
          <SchemaField path={[...base, "see_gamma"]} />
          <CapacitorEditor path={[...base]} kind="2d" />
        </>
      )}
      {r.type === "dielectric" && (
        <>
          <SchemaField path={[...base, "eps_r"]} />
          <SchemaField path={[...base, "see_gamma"]} hint={t("settings.gammaPicOnly")} />
        </>
      )}
      {r.type === "charge" && <SchemaField path={[...base, "rho"]} />}
      <div className="subsection-title">{t("settings.shape")}</div>
      {r.shape ? (
        <>
          <SchemaField path={[...base, "shape", "center"]} />
          <SchemaField path={[...base, "shape", "radius"]} />
        </>
      ) : (
        <>
          <PathTable path={regionPath(r)} label={t("action.regionShape")} ops={regionPathOps(r.id, t)} docPath={[...base, "polygon"]} />
          <RegionHoles region={r} index={idx} />
        </>
      )}
      <RegionBoolean region={r} />
      <RegionLocalSize id={r.id} />
      <TransformPanel />
      <div className="button-row">
        <button
          type="button"
          className="button"
          disabled={(r.holes?.length ?? 0) > 0}
          title={(r.holes?.length ?? 0) > 0 ? t("holes.noDomain") : t("settings.regionToDomainHint")}
          onClick={() => {
            let removed = 0;
            update(t("cad.toDomain"), (d) => {
              const rep = regionToDomain(d, r.id);
              removed = rep ? rep.boundariesRemoved + rep.periodicRemoved : 0;
            });
            if (removed > 0) logInfo(t("msg.source.app"), t("settings.bcRemoved", { n: removed }));
            select("domain");
          }}
        >
          {t("cad.toDomain")}
        </button>
        <button
          type="button"
          className="button danger"
          onClick={async () => {
            if (!(await askConfirm(t("tree.deleteRegionTitle"), t("tree.deleteRegionMessage", { id: r.id }), { okLabel: t("tree.delete"), danger: true })))
              return;
            update(t("action.deleteRegion"), (d) => deleteRegion(d, r.id));
            select("regions");
          }}
        >
          {t("tree.delete")}
        </button>
      </div>
    </>
  );
}

export function BoundariesPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const select = useSelection((s) => s.select);
  return (
    <>
      <table className="table">
        <tbody>
          {edgeIdsOf(project).map((id, i) => (
            <tr key={id} className="clickable" onClick={() => select(`edge:${id}`)}>
              <td>{edgeLabel(project, i, t)}</td>
              <td>{edgeSummary(project, i, t)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Hint>{t("bcPage.overviewHint")}</Hint>
    </>
  );
}

/** 辺の形の説明 (直線は長さ、円弧は中心角と半径) */
function edgeShapeText(p: Project, i: number, t: TFunction, unit: LengthUnit): string {
  const poly = p.geometry.domain.polygon;
  const s = segFromBulge(poly[i], poly[(i + 1) % poly.length], bulgesOf(domainPath(p))[i]);
  const len = (m: number) => `${formatNumber(toDisplayLength(m, unit))} ${lengthUnitLabel(unit)}`;
  if (s.kind === "line") return t("bcPage.line", { len: len(segLength(s)) });
  const deg = (4 * Math.atan(bulgesOf(domainPath(p))[i]) * 180) / Math.PI;
  return t("bcPage.arc", { angle: formatNumber(deg), r: len((s as ArcSeg).r) });
}

export function EdgePage({ id }: { id: string }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const unit = usePrefs((s) => s.lengthUnit);
  const update = useUpdate();
  const [error, setError] = useState<string | null>(null);
  const edge = edgeIndexOf(project, id);
  if (edge < 0) return <Hint>{t("bcPage.missing")}</Hint>;
  const type = edgeBcType(project, edge);
  const shape = (
    <>
      <div className="subsection-title">{t("bcPage.shape")}</div>
      <p className="muted">{edgeShapeText(project, edge, t, unit)}</p>
      <div className="button-row">
        <button type="button" className="button small" onClick={() => editDomain(t, t("action.splitEdge"), (p) => splitDomainEdge(p, edgeIndexOf(p, id)))}>
          {t("bcPage.split")}
        </button>
      </div>
      <Hint>{t("bcPage.splitHint")}</Hint>
    </>
  );
  if (type === "axis")
    return (
      <>
        <Hint>{t("bcPage.axisLocked")}</Hint>
        {shape}
      </>
    );
  const bi = boundaryIndexOfEdge(project, edge);
  const bc = bi >= 0 ? project.geometry.boundaries[bi] : null;
  const partners = periodicPartners(project, edge);
  const partner = periodicPartnerOf(project, edge);
  const types: EdgeBcType[] = ["neumann", "dirichlet", "symmetry", "periodic"];
  const base = ["geometry", "boundaries", bi] as const;
  const setType = (ty: EdgeBcType, with_?: number) => {
    let ok = true;
    update(t("bcPage.changeType"), (d) => {
      ok = setEdgeType(d, edge, ty, with_ ?? (ty === "periodic" ? partners[0] : undefined));
    });
    setError(ok ? null : t("bcPage.periodicNeedsPartner"));
  };
  return (
    <>
      <Field label={t("bcPage.typeLabel")}>
        {(fid) => (
          <Select<EdgeBcType>
            id={fid}
            value={type}
            options={types.map((ty) => ({
              value: ty,
              label: t(`bcPage.type.${ty}`),
              disabled: ty === "periodic" && partners.length === 0,
            }))}
            onChange={(ty) => setType(ty)}
          />
        )}
      </Field>
      {error && <Hint tone="error">{error}</Hint>}
      {type === "periodic" && partner !== null && partners.length > 1 && (
        <Field label={t("bcPage.periodicPartner")}>
          {(fid) => (
            <Select<string>
              id={fid}
              value={String(partner)}
              options={partners.map((j) => ({ value: String(j), label: edgeLabel(project, j, t) }))}
              onChange={(j) => setType("periodic", Number(j))}
            />
          )}
        </Field>
      )}
      {type === "periodic" && partner !== null && partners.length <= 1 && <Hint>{t("bcPage.periodicPair", { edge: edgeLabel(project, partner, t) })}</Hint>}
      {bc && bc.edges.length > 1 && type !== "periodic" && <Hint>{t("bcPage.shared", { n: bc.edges.length })}</Hint>}
      {type === "dirichlet" && bc && (
        <>
          <SchemaField path={[...base, "voltage"]} required />
          {bc.blocking_capacitor && <Hint>{t("capacitor.dcBlocked")}</Hint>}
          <RfEditor path={[...base, "voltage_rf"]} />
          <div className="subsection-title">{t("widgets.waveform")}</div>
          <WaveformEditor path={[...base, "voltage_waveform"]} />
          <VoltagePreview
            dc={bc.voltage ?? 0}
            rf={rfComponents(bc.voltage_rf)}
            waveforms={bc.voltage_waveform ? [bc.voltage_waveform as VoltageWaveform] : []}
          />
          <SchemaField path={[...base, "see_gamma"]} />
          <CapacitorEditor path={[...base]} kind="2d" />
        </>
      )}
      {type === "symmetry" && <Hint>{t("bcPage.symmetryHint")}</Hint>}
      {type === "neumann" && <Hint>{t("bcPage.neumannHint")}</Hint>}
      {shape}
    </>
  );
}

export function BFieldPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  if (coordOf(project) !== "xy") return <Hint>{t("bfield.axisymmetric")}</Hint>;
  const b = project.b_field ?? { bx: 0, by: 0, bz: 0 };
  const set = (key: "bx" | "by" | "bz", v: number | null) => {
    const next = { bx: b.bx ?? 0, by: b.by ?? 0, bz: b.bz ?? 0, [key]: v ?? 0 };
    // 全て 0 なら磁場なし (null、v1 と同じ)
    setValue(["b_field"], next.bx || next.by || next.bz ? next : null, t("tree.bfield"));
  };
  return (
    <>
      {(["bx", "by", "bz"] as const).map((k) => (
        <DefQuantity key={k} def="BField" prop={k} value={b[k] ?? 0} onCommit={(v) => set(k, v)} />
      ))}
      <Hint>{t("bfield.hint")}</Hint>
    </>
  );
}
