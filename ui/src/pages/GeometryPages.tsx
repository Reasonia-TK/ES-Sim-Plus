// ジオメトリの設定ページ: プロジェクト・ドメイン・領域 (一覧と各領域)・境界条件 (一覧と各辺)・磁場。

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { askConfirm } from "../app/dialogs";
import { SchemaField } from "../forms/SchemaField";
import { setValue } from "../forms/useField";
import { edgeBcType, oppositeEdge, setEdgeType, boundaryIndexOfEdge, type EdgeBcType } from "../model/boundaryOps";
import { documentName, useDocument, useIsDirty } from "../model/documentStore";
import { unionPolygons } from "../model/mergeRegions";
import { axisEdge, coordOf, isRectDomain, polygonBounds, type Coord, type Project, type Region, type RegionType } from "../model/project";
import { addCircleRegion, addRectRegion, deleteRegion, renameRegion, setRegionType, validateRegionId } from "../model/regionOps";
import { useSelection } from "../model/selection";
import { usePrefs } from "../prefs/prefs";
import { edgeLabel, edgeSummary } from "../tree/treeModel";
import type { LengthUnit } from "../util/format";
import { rfComponents, type VoltageWaveform } from "../util/waveform";
import { CommitText, Field, LengthInput, Select } from "./inputs";
import { useLengthUnitLabel } from "./useLengthUnitLabel";
import { DefQuantity, Hint } from "./widgets/common";
import { RfEditor } from "./widgets/RfEditor";
import { VertexTable } from "./widgets/VertexTable";
import { VoltagePreview } from "./widgets/VoltagePreview";
import { WaveformEditor } from "./widgets/WaveformEditor";
import { logInfo } from "../app/messages";

const useUpdate = () => useDocument((s) => s.update);

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
              // 軸対称に切り替えたら対称軸の辺の境界条件を外す (v1 と同じ)。周期境界は対の片方だけ残すと backend が
              // 拒むので、軸の辺を含むものはまるごと外す
              const axis = axisEdge(d as Project);
              if (axis !== null) {
                const before = JSON.stringify(d.geometry.boundaries);
                d.geometry.boundaries = d.geometry.boundaries.filter((b) => !(b.type === "periodic" && b.edges.includes(axis)));
                for (const b of d.geometry.boundaries) b.edges = b.edges.filter((e) => e !== axis);
                d.geometry.boundaries = d.geometry.boundaries.filter((b) => b.edges.length > 0);
                removed = JSON.stringify(d.geometry.boundaries) !== before;
                // 磁場は平面 2D だけ (軸対称では backend が拒む)
                if (d.b_field) d.b_field = null;
              }
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
      <SchemaField path={["solver", "backend"]} />
    </>
  );
}

export function DomainPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const update = useUpdate();
  const u = useLengthUnitLabel();
  const coord = coordOf(project);
  const b = polygonBounds(project.geometry.domain.polygon);
  const setSize = (w: number, h: number) =>
    update(t("action.domainSize"), (d) => {
      d.geometry.domain.polygon = [
        [b.x0, b.y0],
        [b.x0 + w, b.y0],
        [b.x0 + w, b.y0 + h],
        [b.x0, b.y0 + h],
      ];
    });
  const wLabel = coord === "rz" ? t("settings.widthZ") : coord === "rz_x0" ? t("settings.widthR") : t("settings.width");
  const hLabel = coord === "rz" ? t("settings.heightR") : coord === "rz_x0" ? t("settings.heightZ") : t("settings.height");
  return (
    <>
      <CoordSelect />
      {coord !== "xy" && <Hint>{t("settings.axisHint")}</Hint>}
      {isRectDomain(project) ? (
        <>
          <Field label={wLabel} unit={u}>
            {(id, onError) => <LengthInput id={id} onError={onError} value={b.x1 - b.x0} min={0} exclusive onCommit={(w) => setSize(w, b.y1 - b.y0)} />}
          </Field>
          <Field label={hLabel} unit={u}>
            {(id, onError) => <LengthInput id={id} onError={onError} value={b.y1 - b.y0} min={0} exclusive onCommit={(h) => setSize(b.x1 - b.x0, h)} />}
          </Field>
        </>
      ) : (
        <>
          <p className="hint">{t("settings.nonRectDomain", { n: project.geometry.domain.polygon.length })}</p>
          <VertexTable path={["geometry", "domain", "polygon"]} label={t("tree.domain")} />
        </>
      )}
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

function RegionMerge({ region }: { region: Region }) {
  const { t } = useTranslation();
  const regions = useDocument((s) => s.project.geometry.regions);
  const others = regions.filter((r) => r.id !== region.id && r.polygon);
  const [other, setOther] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  if (!region.polygon) return <Hint>{t("merge.onlyPolygons")}</Hint>;
  if (others.length === 0) return null;
  const target = others.find((r) => r.id === other) ?? others[0];
  return (
    <div className="subsection">
      <div className="subsection-title">{t("merge.title")}</div>
      <div className="field-control">
        <select
          className="input"
          value={target.id}
          onChange={(e) => {
            setOther(e.target.value);
            setError(null);
          }}
          aria-label={t("merge.title")}
        >
          {others.map((r) => (
            <option key={r.id} value={r.id}>
              {r.id}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="button small"
          onClick={() => {
            const res = unionPolygons(region.polygon!, target.polygon!);
            if ("error" in res) {
              setError(t(`merge.error.${res.error}`));
              return;
            }
            setError(null);
            useDocument.getState().update(t("merge.title"), (d) => {
              const r = d.geometry.regions.find((x) => x.id === region.id);
              if (r) r.polygon = res.polygon;
              deleteRegion(d, target.id);
            });
          }}
        >
          {t("merge.apply")}
        </button>
      </div>
      {error && <Hint tone="error">{error}</Hint>}
      <Hint>{t("merge.hint")}</Hint>
    </div>
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
      {r.type === "conductor" && (
        <>
          <SchemaField path={[...base, "voltage"]} required />
          <RfEditor path={[...base, "voltage_rf"]} />
          <div className="subsection-title">{t("widgets.waveform")}</div>
          <WaveformEditor path={[...base, "voltage_waveform"]} />
          <VoltagePreview
            dc={r.voltage ?? 0}
            rf={rfComponents(r.voltage_rf)}
            waveforms={r.voltage_waveform ? [r.voltage_waveform as VoltageWaveform] : []}
          />
          <SchemaField path={[...base, "see_gamma"]} />
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
          <VertexTable path={[...base, "polygon"]} label={t("action.regionShape")} />
          <RegionMerge region={r} />
        </>
      )}
      <RegionLocalSize id={r.id} />
      <div className="button-row">
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
          {project.geometry.domain.polygon.map((_, i) => (
            <tr key={i} className="clickable" onClick={() => select(`edge:${i}`)}>
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

export function EdgePage({ edge }: { edge: number }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const update = useUpdate();
  const [error, setError] = useState<string | null>(null);
  const type = edgeBcType(project, edge);
  if (type === "axis") return <Hint>{t("bcPage.axisLocked")}</Hint>;
  const bi = boundaryIndexOfEdge(project, edge);
  const bc = bi >= 0 ? project.geometry.boundaries[bi] : null;
  const opp = oppositeEdge(project, edge);
  const types: EdgeBcType[] = ["neumann", "dirichlet", "symmetry", "periodic"];
  const base = ["geometry", "boundaries", bi] as const;
  return (
    <>
      <Field label={t("bcPage.typeLabel")}>
        {(id) => (
          <Select<EdgeBcType>
            id={id}
            value={type}
            options={types.map((ty) => ({
              value: ty,
              label: t(`bcPage.type.${ty}`),
              disabled: ty === "periodic" && (opp === null || axisEdge(project) === opp),
            }))}
            onChange={(ty) => {
              let ok = true;
              update(t("bcPage.changeType"), (d) => {
                ok = setEdgeType(d, edge, ty);
              });
              setError(ok ? null : t("bcPage.periodicNeedsRect"));
            }}
          />
        )}
      </Field>
      {error && <Hint tone="error">{error}</Hint>}
      {type === "periodic" && opp !== null && <Hint>{t("bcPage.periodicPair", { edge: edgeLabel(project, opp, t) })}</Hint>}
      {bc && bc.edges.length > 1 && type !== "periodic" && <Hint>{t("bcPage.shared", { n: bc.edges.length })}</Hint>}
      {type === "dirichlet" && bc && (
        <>
          <SchemaField path={[...base, "voltage"]} required />
          <RfEditor path={[...base, "voltage_rf"]} />
          <div className="subsection-title">{t("widgets.waveform")}</div>
          <WaveformEditor path={[...base, "voltage_waveform"]} />
          <VoltagePreview
            dc={bc.voltage ?? 0}
            rf={rfComponents(bc.voltage_rf)}
            waveforms={bc.voltage_waveform ? [bc.voltage_waveform as VoltageWaveform] : []}
          />
          <SchemaField path={[...base, "see_gamma"]} />
        </>
      )}
      {type === "symmetry" && <Hint>{t("bcPage.symmetryHint")}</Hint>}
      {type === "neumann" && <Hint>{t("bcPage.neumannHint")}</Hint>}
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
