// 設定欄: 選んだノードのページを出す。P6a はプロジェクト・ドメイン・領域を編集でき、ほかは内容を
// 表示するだけ (スキーマから生成するフォームは P6b)。

import { useTranslation } from "react-i18next";
import { askConfirm } from "../app/dialogs";
import { documentName, useDocument, useIsDirty } from "../model/documentStore";
import {
  axisEdge,
  boundaryOfEdge,
  coordOf,
  isRectDomain,
  polygonBounds,
  type Coord,
  type Project,
  type Region,
  type RegionType,
} from "../model/project";
import { addCircleRegion, addRectRegion, deleteRegion, renameRegion, setRegionType, validateRegionId } from "../model/regionOps";
import { useSelection } from "../model/selection";
import { usePrefs } from "../prefs/prefs";
import { edgeLabel, STUDY_SETTINGS_KEY, type StudyKind } from "../tree/treeModel";
import type { LengthUnit } from "../util/format";
import { CommitText, Field, LengthInput, NumberInput, Select } from "./inputs";
import { useLengthUnitLabel } from "./useLengthUnitLabel";

function useUpdate() {
  return useDocument((s) => s.update);
}

function CoordSelect() {
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
            update(t("action.coord"), (d) => {
              d.coord = c;
              // 軸対称に切り替えたら対称軸の辺の境界条件を外す (v1 と同じ)
              const axis = axisEdge(d as Project);
              if (axis !== null) {
                for (const b of d.geometry.boundaries) b.edges = b.edges.filter((e) => e !== axis);
                d.geometry.boundaries = d.geometry.boundaries.filter((b) => b.edges.length > 0);
              }
            })
          }
        />
      )}
    </Field>
  );
}

function ProjectPage() {
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

function DomainPage() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const update = useUpdate();
  const u = useLengthUnitLabel();
  const coord = coordOf(project);
  const rect = isRectDomain(project);
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
    </>
  );
}

function RegionsPage() {
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
            <span className="muted">{t(`region.${r.type}`)}</span>
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
    </>
  );
}

function RegionPage({ id }: { id: string }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const update = useUpdate();
  const select = useSelection((s) => s.select);
  const u = useLengthUnitLabel();
  const r = project.geometry.regions.find((x) => x.id === id);
  if (!r) return <p className="muted">{t("settings.selectRegion")}</p>;
  const edit = (label: string, fn: (reg: Region) => void) =>
    update(label, (d) => {
      const reg = d.geometry.regions.find((x) => x.id === id);
      if (reg) fn(reg as Region);
    });
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
        <Field label={t("settings.voltage")} unit="V">
          {(fid, onError) => (
            <NumberInput
              id={fid}
              onError={onError}
              value={typeof r.voltage === "number" ? r.voltage : 0}
              onCommit={(v) => edit(t("action.regionValue"), (reg) => void (reg.voltage = v ?? 0))}
            />
          )}
        </Field>
      )}
      {r.type === "dielectric" && (
        <Field label={t("settings.epsR")}>
          {(fid, onError) => (
            <NumberInput
              id={fid}
              onError={onError}
              value={typeof r.eps_r === "number" ? r.eps_r : 1}
              min={0}
              exclusive
              onCommit={(v) => edit(t("action.regionValue"), (reg) => void (reg.eps_r = v ?? 1))}
            />
          )}
        </Field>
      )}
      {r.type === "charge" && (
        <Field label={t("settings.rho")} unit="C/m³">
          {(fid, onError) => (
            <NumberInput
              id={fid}
              onError={onError}
              value={typeof r.rho === "number" ? r.rho : 0}
              onCommit={(v) => edit(t("action.regionValue"), (reg) => void (reg.rho = v ?? 0))}
            />
          )}
        </Field>
      )}
      {r.shape ? (
        <>
          <Field label={t("settings.centerX")} unit={u}>
            {(fid, onError) => (
              <LengthInput
                id={fid}
                onError={onError}
                value={r.shape!.center[0]}
                onCommit={(v) => edit(t("action.regionShape"), (reg) => void (reg.shape!.center[0] = v))}
              />
            )}
          </Field>
          <Field label={t("settings.centerY")} unit={u}>
            {(fid, onError) => (
              <LengthInput
                id={fid}
                onError={onError}
                value={r.shape!.center[1]}
                onCommit={(v) => edit(t("action.regionShape"), (reg) => void (reg.shape!.center[1] = v))}
              />
            )}
          </Field>
          <Field label={t("settings.radius")} unit={u}>
            {(fid, onError) => (
              <LengthInput
                id={fid}
                onError={onError}
                value={r.shape!.radius}
                min={0}
                exclusive
                onCommit={(v) => edit(t("action.regionShape"), (reg) => void (reg.shape!.radius = v))}
              />
            )}
          </Field>
        </>
      ) : (
        <p className="muted">
          {t("region.polygon")} · {t("settings.vertices", { n: r.polygon?.length ?? 0 })}
        </p>
      )}
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

function JsonPage({ value, note }: { value: unknown; note?: string }) {
  const { t } = useTranslation();
  return (
    <>
      {note && <p className="hint">{note}</p>}
      <details open>
        <summary>{t("settings.rawJson")}</summary>
        <pre className="json">{JSON.stringify(value ?? null, null, 2)}</pre>
      </details>
    </>
  );
}

function pageTitle(node: string, project: Project, t: ReturnType<typeof useTranslation>["t"]): string {
  if (node.startsWith("region:")) return `${t("tree.regions")} › ${node.slice(7)}`;
  if (node.startsWith("edge:")) return `${t("tree.boundaries")} › ${edgeLabel(project, Number(node.slice(5)), t)}`;
  if (node.startsWith("study:")) return t(`study.${node.slice(6) as StudyKind}`);
  const titles: Record<string, string> = {
    project: t("tree.project"),
    geometry: t("tree.geometry"),
    domain: t("tree.domain"),
    regions: t("tree.regions"),
    boundaries: t("tree.boundaries"),
    mesh: t("tree.mesh"),
    bfield: t("tree.bfield"),
    studies: t("tree.studies"),
    results: t("tree.results"),
  };
  return titles[node] ?? node;
}

function PageBody({ node }: { node: string }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  if (node === "project") return <ProjectPage />;
  if (node === "domain" || node === "geometry") return <DomainPage />;
  if (node === "regions") return <RegionsPage />;
  if (node.startsWith("region:")) return <RegionPage id={node.slice(7)} />;
  if (node === "boundaries") return <JsonPage value={project.geometry.boundaries} note={t("settings.formsLater")} />;
  if (node.startsWith("edge:")) {
    const i = Number(node.slice(5));
    return <JsonPage value={boundaryOfEdge(project, i)} note={t("settings.formsLater")} />;
  }
  if (node === "mesh") return <JsonPage value={project.mesh} note={t("settings.formsLater")} />;
  if (node === "bfield") return <JsonPage value={project.b_field} note={t("settings.formsLater")} />;
  if (node.startsWith("study:")) {
    const key = STUDY_SETTINGS_KEY[node.slice(6) as StudyKind];
    if (key === null) return <p className="hint">{t("settings.formsLater")}</p>;
    const v = project[key];
    return v === null || v === undefined ? <p className="hint">{t("settings.notConfigured")}</p> : <JsonPage value={v} note={t("settings.formsLater")} />;
  }
  return null;
}

export function SettingsPanel() {
  const { t } = useTranslation();
  const node = useSelection((s) => s.activeNode);
  const project = useDocument((s) => s.project);
  return (
    <div className="settings-panel">
      <div className="panel-header">
        <h2 className="panel-title">{pageTitle(node, project, t)}</h2>
      </div>
      <div className="panel-body">
        <PageBody node={node} />
      </div>
    </div>
  );
}
