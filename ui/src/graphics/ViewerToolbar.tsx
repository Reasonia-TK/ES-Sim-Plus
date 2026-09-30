// ビューアの道具の帯: 作図 (選択・折れ線・矩形・円と描く先 (領域かスケッチ)、線・3 点の円弧・囲まれた所から領域)、配置 (プロファイル線・エミッタ・コレクタ・ガス境界・EEDF 領域・
// 辺のメッシュ幅・シース評価線)、調べる (プローブ・計測)、スナップ・全体表示、表示の設定、書き出し。

import { DropdownMenu, Popover } from "radix-ui";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Toggle } from "../forms/SchemaField";
import { useDocument } from "../model/documentStore";
import { MAX_COLLECTORS, MAX_EEDF_REGIONS, MAX_SHEATH_LINES } from "../model/placements";
import type { Project } from "../model/project";
import { CommitText } from "../pages/inputs";
import { jobName } from "../jobs/jobsStore";
import type { JobSummary } from "../jobs/types";
import { SNAP_KINDS } from "../cad/snap";
import { useChartPref, useResultsView } from "../results/resultsView";
import type { SceneControls } from "./useScene";
import { formatNumber, parseNumber } from "../util/format";
import { colormapCss, COLORMAP_KEYS, type ColormapKey } from "./colormaps";
import {
  IconArc,
  IconChevron,
  IconCircle,
  IconDisplay,
  IconEdit,
  IconExport,
  IconFill,
  IconFit,
  IconLine,
  IconMeasure,
  IconPlace,
  IconPolyline,
  IconProbe,
  IconRect,
  IconSelect,
  IconSnap,
  IconTransform,
} from "./icons";
import type { Scene } from "./scene";
import { EDIT_TOOLS, PLACE_TOOLS, RULER_FONTS, TRANSFORM_TOOLS, useViewer, type DrawTarget, type EditTool, type OverlayKey, type PlaceTool, type RulerFont, type Tool, type TransformTool } from "./viewerStore";
import { usePrefs } from "../prefs/prefs";
import { lengthUnitLabel, toDisplayLength } from "../util/format";

function ToolButton({ tool, icon, label, disabled, title }: { tool: Tool; icon: ReactNode; label: string; disabled?: boolean; title?: string }) {
  const active = useViewer((s) => s.tool === tool);
  const setTool = useViewer((s) => s.setTool);
  return (
    <button
      type="button"
      className={`tool-button${active ? " active" : ""}`}
      aria-pressed={active}
      aria-label={label}
      title={title ?? label}
      disabled={disabled}
      onClick={() => setTool(tool)}
    >
      {icon}
    </button>
  );
}

/** 変換 (移動・回転・ミラー・尺度) とコピーを作るか */
function TransformMenu() {
  const { t } = useTranslation();
  const tool = useViewer((s) => s.tool);
  const setTool = useViewer((s) => s.setTool);
  const copy = useViewer((s) => s.transformCopy);
  const setCopy = useViewer((s) => s.setTransformCopy);
  const active = (TRANSFORM_TOOLS as Tool[]).includes(tool);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className={`tool-button with-label${active ? " active" : ""}`}
          aria-label={active ? `${t("viewer.transform")} (${t(`viewer.tool.${tool as TransformTool}`)})` : t("viewer.transform")}
          title={t("viewer.transformHint")}
        >
          <IconTransform />
          {/* 道具の帯が折り返さないよう、使っている道具の名前だけ出す */}
          {active && <span>{t(`viewer.tool.${tool as TransformTool}`)}</span>}
          <IconChevron />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-content" align="start" sideOffset={4}>
          {TRANSFORM_TOOLS.map((k) => (
            <DropdownMenu.Item key={k} className="menu-item" onSelect={() => setTool(k)}>
              {tool === k && <span className="menu-indicator">●</span>}
              {t(`viewer.tool.${k}`)}
            </DropdownMenu.Item>
          ))}
          <DropdownMenu.Separator className="menu-separator" />
          <DropdownMenu.CheckboxItem className="menu-item menu-radio" checked={copy} onCheckedChange={(v) => setCopy(v === true)} onSelect={(e) => e.preventDefault()}>
            <DropdownMenu.ItemIndicator className="menu-indicator">✓</DropdownMenu.ItemIndicator>
            {t("viewer.transformCopy")}
          </DropdownMenu.CheckboxItem>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

/** 編集 (フィレット・面取り・オフセット・トリム・延長)。半径・長さを添える */
function EditMenu() {
  const { t } = useTranslation();
  const tool = useViewer((s) => s.tool);
  const setTool = useViewer((s) => s.setTool);
  // 値ごとに選ぶ (新しいオブジェクトを返す選択は描き直しが止まらない)
  const fillet = useViewer((s) => s.filletRadius);
  const chamfer = useViewer((s) => s.chamferDistance);
  const offset = useViewer((s) => s.offsetDistance);
  const sizes = { fillet, chamfer, offset };
  const unit = usePrefs((s) => s.lengthUnit);
  const active = (EDIT_TOOLS as Tool[]).includes(tool);
  const size = (k: EditTool) => (k === "fillet" || k === "chamfer" || k === "offset" ? `${toDisplayLength(sizes[k], unit)} ${lengthUnitLabel(unit)}` : null);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          className={`tool-button with-label${active ? " active" : ""}`}
          aria-label={active ? `${t("viewer.edit")} (${t(`viewer.tool.${tool as EditTool}`)})` : t("viewer.edit")}
          title={t("viewer.editHint")}
        >
          <IconEdit />
          {active && <span>{t(`viewer.tool.${tool as EditTool}`)}</span>}
          <IconChevron />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-content" align="start" sideOffset={4}>
          {EDIT_TOOLS.map((k) => (
            <DropdownMenu.Item key={k} className="menu-item" onSelect={() => setTool(k)}>
              {tool === k && <span className="menu-indicator">●</span>}
              {t(`viewer.tool.${k}`)}
              {size(k) && <span className="menu-shortcut">{size(k)}</span>}
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

/** スナップ: 全体のオン・オフと種類ごとの切り替え */
function SnapMenu() {
  const { t } = useTranslation();
  const snap = useViewer((s) => s.snap);
  const setSnap = useViewer((s) => s.setSnap);
  const kinds = useViewer((s) => s.snapKinds);
  const setKind = useViewer((s) => s.setSnapKind);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button type="button" className={`tool-button${snap ? " active" : ""}`} aria-pressed={snap} aria-label={t("viewer.snap")} title={t("viewer.snapHint")}>
          <IconSnap />
          <IconChevron />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-content" align="start" sideOffset={4}>
          <DropdownMenu.CheckboxItem className="menu-item menu-radio" checked={snap} onCheckedChange={(v) => setSnap(v === true)} onSelect={(e) => e.preventDefault()}>
            <DropdownMenu.ItemIndicator className="menu-indicator">✓</DropdownMenu.ItemIndicator>
            {t("viewer.snapAll")}
            <span className="menu-shortcut">F3</span>
          </DropdownMenu.CheckboxItem>
          <DropdownMenu.Separator className="menu-separator" />
          {SNAP_KINDS.map((k) => (
            <DropdownMenu.CheckboxItem
              key={k}
              className="menu-item menu-radio"
              checked={kinds[k]}
              disabled={!snap}
              onCheckedChange={(v) => setKind(k, v === true)}
              onSelect={(e) => e.preventDefault()}
            >
              <DropdownMenu.ItemIndicator className="menu-indicator">✓</DropdownMenu.ItemIndicator>
              {t(`snapKind.${k}`)}
            </DropdownMenu.CheckboxItem>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

/** 折れ線・矩形・円を描く先 (領域かスケッチ) */
function TargetSwitch() {
  const { t } = useTranslation();
  const target = useViewer((s) => s.drawTarget);
  const setTarget = useViewer((s) => s.setDrawTarget);
  const targets: DrawTarget[] = ["region", "sketch"];
  return (
    <div className="segmented draw-target" role="radiogroup" aria-label={t("viewer.target.label")} title={t("viewer.target.hint")}>
      {targets.map((k) => (
        <button key={k} type="button" role="radio" aria-checked={target === k} className={target === k ? "active" : ""} onClick={() => setTarget(k)}>
          {t(`viewer.target.${k}`)}
        </button>
      ))}
    </div>
  );
}

function listLength(p: Project, key: string): number {
  const v = (p.pic as Record<string, unknown> | null | undefined)?.[key];
  return Array.isArray(v) ? v.length : 0;
}

/** 配置の道具の上限の表示 (n / 上限) */
function placeCount(p: Project, tool: PlaceTool): [number, number] | null {
  if (tool === "collector") return [listLength(p, "collectors"), MAX_COLLECTORS];
  if (tool === "eedfbox") return [listLength(p, "eedf_regions"), MAX_EEDF_REGIONS];
  if (tool === "sheathline") return [listLength(p, "sheath_lines"), MAX_SHEATH_LINES];
  return null;
}

function PlaceMenu() {
  const { t } = useTranslation();
  const tool = useViewer((s) => s.tool);
  const setTool = useViewer((s) => s.setTool);
  const project = useDocument((s) => s.project);
  const active = (PLACE_TOOLS as Tool[]).includes(tool);
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button type="button" className={`tool-button with-label${active ? " active" : ""}`} title={t("viewer.placeHint")}>
          <IconPlace />
          <span>{active ? t(`viewer.tool.${tool as PlaceTool}`) : t("viewer.place")}</span>
          <IconChevron />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-content" align="start" sideOffset={4}>
          {PLACE_TOOLS.map((pt) => {
            const count = placeCount(project, pt);
            return (
              <DropdownMenu.Item key={pt} className="menu-item" onSelect={() => setTool(pt)} title={t(`viewer.toolHint.${pt}`)}>
                {tool === pt && <span className="menu-indicator">●</span>}
                {t(`viewer.tool.${pt}`)}
                {count && (
                  <span className="menu-shortcut">
                    {count[0]} / {count[1]}
                  </span>
                )}
              </DropdownMenu.Item>
            );
          })}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

/** 配色の見本 (横長のグラデーション) */
export function ColormapSwatch({ cmap }: { cmap: ColormapKey }) {
  const stops = Array.from({ length: 9 }, (_, i) => `${colormapCss(cmap, i / 8)} ${(i / 8) * 100}%`).join(", ");
  return <span className="colormap-swatch" style={{ background: `linear-gradient(to right, ${stops})` }} aria-hidden="true" />;
}

function RangeInput({ value, onCommit, label }: { value: number | null; onCommit: (v: number | null) => void; label: string }) {
  const { t } = useTranslation();
  return (
    <CommitText
      className="input range-input"
      inputMode="decimal"
      aria-label={label}
      placeholder={t("viewer.auto")}
      value={value === null ? "" : formatNumber(value)}
      validate={(s) => (s.trim() === "" || parseNumber(s) !== null ? null : t("input.notNumber"))}
      onCommit={(s) => onCommit(s.trim() === "" ? null : parseNumber(s))}
    />
  );
}

const FIELD_OVERLAYS: OverlayKey[] = ["mesh", "isolines", "vectors", "particles", "trajectories"];
const PLACEMENT_OVERLAYS: OverlayKey[] = ["emitter", "collectors", "gasBoundaries", "eedf", "edgeSizes", "sheathLines", "amr"];
const CANVAS_OVERLAYS: OverlayKey[] = ["grid", "rulers", "legend"];

function DisplayPopover({ scene, controls, run }: { scene: Scene; controls: SceneControls; run: JobSummary | undefined }) {
  const { t } = useTranslation();
  const vs = useViewer();
  const follow = useResultsView((s) => s.follow);
  const sheathContour = useResultsView((s) => s.sheathContour);
  const sheathAlpha = useResultsView((s) => s.sheathAlpha);
  const setSheath = useResultsView((s) => s.setSheath);
  const [rfMonitor, setRfMonitor] = useChartPref("viewer.rfMonitor", true);
  const overlay = (k: OverlayKey) => (
    <div key={k} className="display-toggle">
      <Toggle checked={vs.overlays[k]} onChange={(v) => vs.setOverlay(k, v)} label={t(`viewer.overlay.${k}`)} />
    </div>
  );
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" className="tool-button with-label" title={t("viewer.display")}>
          <IconDisplay />
          <span>{t("viewer.display")}</span>
          <IconChevron />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="popover-content viewer-display" align="end" sideOffset={4} collisionPadding={8}>
          {run && (
            <div className="display-row">
              <span className="display-label">{t("results.showing")}</span>
              <div className="range-row">
                <span className="ellipsis">{jobName(run)}</span>
                <button type="button" className="button small" onClick={() => useResultsView.getState().setActiveRun(null)}>
                  {t("results.showStatic")}
                </button>
              </div>
            </div>
          )}
          <div className="display-toggle">
            <Toggle checked={follow} onChange={(v) => useResultsView.getState().setFollow(v)} label={t("results.follow")} />
          </div>
          {run && (run.kind === "pic" || run.kind === "fluid2d") && (
            <div className="display-row">
              <span className="display-label">{t("charts.sheath2d")}</span>
              <div className="range-row">
                <Toggle checked={sheathContour} onChange={(v) => setSheath({ contour: v })} label="n_e/n_i = α" />
                <input
                  type="range"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={sheathAlpha}
                  aria-label={t("charts.alpha", { a: sheathAlpha.toFixed(2) })}
                  title={t("charts.alpha", { a: sheathAlpha.toFixed(2) })}
                  onChange={(e) => setSheath({ alpha: Number(e.target.value) })}
                />
                <span className="mono small">{sheathAlpha.toFixed(2)}</span>
              </div>
            </div>
          )}
          {controls.modes.length > 0 && (
            <div className="display-row">
              <span className="display-label">{t("results.modeLabel")}</span>
              <div className="segmented" role="radiogroup" aria-label={t("results.modeLabel")}>
                {controls.modes.map((m) => (
                  <button
                    key={m.value}
                    type="button"
                    role="radio"
                    aria-checked={controls.mode === m.value}
                    className={controls.mode === m.value ? "active" : ""}
                    disabled={!controls.setMode}
                    onClick={() => controls.setMode?.(m.value)}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
          )}
          {controls.quantities.length > 0 && (
            <div className="display-row">
              <label className="display-label" htmlFor="viewer-quantity">
                {t("viewer.quantity")}
              </label>
              <select id="viewer-quantity" className="input" value={controls.quantity ?? ""} onChange={(e) => controls.setQuantity(e.target.value)}>
                {controls.quantities.map((q) => (
                  <option key={q.value} value={q.value}>
                    {q.label}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="display-row">
            <label className="display-label" htmlFor="viewer-colormap">
              {t("viewer.colormapLabel")}
            </label>
            <select id="viewer-colormap" className="input" value={vs.colormap} onChange={(e) => vs.setColormap(e.target.value as ColormapKey)}>
              {COLORMAP_KEYS.map((k) => (
                <option key={k} value={k}>
                  {t(`viewer.colormap.${k}`)}
                </option>
              ))}
            </select>
          </div>
          <div className="display-row">
            <span className="display-label" />
            <ColormapSwatch cmap={vs.colormap} />
          </div>
          <div className="display-row">
            <span className="display-label">{t("viewer.range")}</span>
            <div className="range-row">
              <RangeInput label={t("viewer.rangeMin")} value={vs.range.min} onCommit={(v) => vs.setRange({ ...vs.range, min: v })} />
              <span>–</span>
              <RangeInput label={t("viewer.rangeMax")} value={vs.range.max} onCommit={(v) => vs.setRange({ ...vs.range, max: v })} />
              <span className="field-unit">{scene.field?.unit ?? ""}</span>
            </div>
          </div>
          <div className="display-row">
            <span className="display-label" />
            <div className="range-row">
              <button type="button" className="button small" disabled={vs.range.min === null && vs.range.max === null} onClick={() => vs.setRange({ min: null, max: null })}>
                {t("viewer.auto")}
              </button>
              <Toggle checked={controls.log} onChange={controls.setLog} label={t("viewer.log")} />
            </div>
          </div>
          <div className="display-group">{t("viewer.groupField")}</div>
          {FIELD_OVERLAYS.map(overlay)}
          <div className="display-toggle">
            <Toggle checked={rfMonitor} onChange={setRfMonitor} label={t("results.rfMonitor")} />
          </div>
          <div className="display-group">{t("viewer.groupPlacements")}</div>
          {PLACEMENT_OVERLAYS.map(overlay)}
          <div className="display-group">{t("viewer.groupCanvas")}</div>
          {CANVAS_OVERLAYS.map(overlay)}
          <div className="display-row">
            <span className="display-label">{t("viewer.rulerFont")}</span>
            <div className="segmented" role="radiogroup" aria-label={t("viewer.rulerFont")}>
              {(Object.keys(RULER_FONTS) as RulerFont[]).map((f) => (
                <button key={f} type="button" role="radio" aria-checked={vs.rulerFont === f} className={vs.rulerFont === f ? "active" : ""} onClick={() => vs.setRulerFont(f)}>
                  {t(`viewer.rulerFont_${f}`)}
                </button>
              ))}
            </div>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function ExportMenu({ scene, onExportPng, onExportCsv }: { scene: Scene; onExportPng: () => void; onExportCsv: () => void }) {
  const { t } = useTranslation();
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button type="button" className="tool-button" aria-label={t("viewer.export")} title={t("viewer.export")}>
          <IconExport />
          <IconChevron />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-content" align="end" sideOffset={4}>
          <DropdownMenu.Item className="menu-item" onSelect={onExportPng}>
            {t("viewer.exportPng")}
          </DropdownMenu.Item>
          <DropdownMenu.Item className="menu-item" disabled={!scene.field} onSelect={onExportCsv}>
            {t("viewer.exportCsv")}
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

export function ViewerToolbar({
  scene,
  controls,
  run,
  onExportPng,
  onExportCsv,
}: {
  scene: Scene;
  controls: SceneControls;
  run: JobSummary | undefined;
  onExportPng: () => void;
  onExportCsv: () => void;
}) {
  const { t } = useTranslation();
  const requestFit = useViewer((s) => s.requestFit);
  return (
    <div className="viewer-toolbar" role="toolbar" aria-label={t("viewer.toolbar")}>
      <ToolButton tool="select" icon={<IconSelect />} label={t("viewer.tool.select")} />
      <ToolButton tool="polyline" icon={<IconPolyline />} label={t("viewer.tool.polyline")} title={`${t("viewer.tool.polyline")} — ${t("viewer.clipHint")}`} />
      <ToolButton tool="rect" icon={<IconRect />} label={t("viewer.tool.rect")} title={`${t("viewer.tool.rect")} — ${t("viewer.clipHint")}`} />
      <ToolButton tool="circle" icon={<IconCircle />} label={t("viewer.tool.circle")} title={`${t("viewer.tool.circle")} — ${t("viewer.clipHint")}`} />
      <TargetSwitch />
      <ToolButton tool="line" icon={<IconLine />} label={t("viewer.tool.line")} />
      <ToolButton tool="arc" icon={<IconArc />} label={t("viewer.tool.arc")} />
      <ToolButton tool="fill" icon={<IconFill />} label={t("viewer.tool.fill")} title={`${t("viewer.tool.fill")} — ${t("viewer.hint.fill")}`} />
      <span className="toolbar-sep" />
      <TransformMenu />
      <EditMenu />
      <span className="toolbar-sep" />
      <PlaceMenu />
      <span className="toolbar-sep" />
      <ToolButton tool="probe" icon={<IconProbe />} label={t("viewer.tool.probe")} disabled={!scene.field} title={scene.field ? t("viewer.tool.probe") : t("viewer.probeNeedsField")} />
      <ToolButton tool="measure" icon={<IconMeasure />} label={t("viewer.tool.measure")} />
      <span className="toolbar-sep" />
      <SnapMenu />
      <button type="button" className="tool-button" aria-label={t("viewer.fit")} title={`${t("viewer.fit")} (F)`} onClick={requestFit}>
        <IconFit />
      </button>
      <span className="spacer" />
      <DisplayPopover scene={scene} controls={controls} run={run} />
      <ExportMenu scene={scene} onExportPng={onExportPng} onExportCsv={onExportCsv} />
    </div>
  );
}
