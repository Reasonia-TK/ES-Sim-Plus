// 2D ビューア: 3 枚の層 (下: 背景・グリッド、中: WebGL の場・メッシュ・等値線・粒子、上: ジオメトリ・配置物・
// 作図・プローブ・カラーバー・ルーラー)、カメラ (全体表示・ホイールでカーソル中心に拡大)、書き出し。作図・編集・配置・
// 調べる道具のマウスとキーボードの操作は useCanvasTools (移動は中ボタンか Space + ドラッグ、v1 と同じ)。

import type { TFunction } from "i18next";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { errorText, logError, logWarning } from "../app/messages";
import { saveBinaryFile, saveTextFile } from "../io/fileAccess";
import { useDocument } from "../model/documentStore";
import { useSelection } from "../model/selection";
import { coordOf, domainBounds, edgeIndexOf, type Point, type Project } from "../model/project";
import { drawingBounds } from "./editTargets";
import { visibleSketch } from "../model/layers";
import { usePrefs } from "../prefs/prefs";
import { formatNumber, lengthUnitLabel, toDisplayLength } from "../util/format";
import { fitCamera, toScreen, toWorld, zoomAt, ZOOM_STEP, type Camera } from "./camera";
import { parseColor } from "./color";
import { composePng, fieldCsv, stamp } from "./exporting";
import { resolveRange } from "./fieldScale";
import { GlRenderer, type MeshDraw, type PointDraw, type Rgba, type SegmentDraw } from "./gl/renderer";
import { isoLevels, isolineSegments } from "./isolines";
import { drawBase, drawOverlay, readColors, rulerSize, type Arrow, type OverlayState, type Placement } from "./overlay";
import { amrBoxesOf, emitterOf, injectorOf, placementsOf } from "./projectOverlays";
import { sampleField, sampleVector, statsOf, type Scene } from "./scene";
import { PlaybackBar } from "./PlaybackBar";
import { ViewerRfStrip } from "./ViewerRfStrip";
import type { ActiveScene } from "./useScene";
import { localPoint, toolHint, useCanvasTools, type CoordMode } from "./useCanvasTools";
import { RULER_FONTS, useViewer } from "./viewerStore";
import { ViewerToolbar } from "./ViewerToolbar";

function axisNames(p: Project): [string, string] {
  const c = coordOf(p);
  return c === "rz" ? ["z", "r"] : c === "rz_x0" ? ["r", "z"] : ["x", "y"];
}

function setCanvasSize(cv: HTMLCanvasElement, w: number, h: number, dpr: number): CanvasRenderingContext2D | null {
  const W = Math.max(1, Math.round(w * dpr));
  const H = Math.max(1, Math.round(h * dpr));
  if (cv.width !== W) cv.width = W;
  if (cv.height !== H) cv.height = H;
  let ctx: CanvasRenderingContext2D | null = null;
  try {
    ctx = cv.getContext("2d");
  } catch {
    ctx = null;
  }
  ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** 矢印を置く点 (画面の格子、最大およそ 600 本、v1 と同じ本数) */
function vectorArrows(scene: Scene, cam: Camera, w: number, h: number, top: number): Arrow[] {
  const vf = scene.vectors;
  if (!vf) return [];
  const spacing = Math.max(22, Math.sqrt((w * h) / 600));
  const out: Arrow[] = [];
  for (let sy = top + spacing / 2; sy < h; sy += spacing) {
    for (let sx = top + spacing / 2; sx < w; sx += spacing) {
      const [x, y] = toWorld(cam, sx, sy);
      const e = sampleVector(vf, x, y);
      if (!e) continue;
      const m = Math.hypot(e[0], e[1]);
      if (!(m > 1e-12)) continue;
      out.push({ x: sx, y: sy, dx: e[0] / m, dy: -e[1] / m });
    }
  }
  return out;
}

export function Viewer({ active }: { active: ActiveScene }) {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const docSerial = useDocument((s) => s.docSerial);
  const selectedPlacement = useSelection((s) => s.selectedPlacement);
  const activeNode = useSelection((s) => s.activeNode);
  const lengthUnit = usePrefs((s) => s.lengthUnit);
  const theme = usePrefs((s) => s.theme);
  const vs = useViewer();
  const { scene, controls, run } = active;

  const wrapRef = useRef<HTMLDivElement>(null);
  const baseRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<HTMLCanvasElement>(null);
  const topRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<GlRenderer | null>(null);
  const [glOk, setGlOk] = useState(true);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [camera, setCamera] = useState<Camera | null>(null);
  const userMovedRef = useRef(false);
  const [redraw, setRedraw] = useState(0);

  // ---- 大きさ・WebGL ----

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    // 最初の大きさはすぐに読む (ResizeObserver は描画の度にしか届かず、ウィンドウが隠れていると届かない)
    const r0 = el.getBoundingClientRect();
    setSize({ w: Math.floor(r0.width), h: Math.floor(r0.height) });
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.floor(e.contentRect.width), h: Math.floor(e.contentRect.height) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = glRef.current;
    if (!cv) return;
    let r: GlRenderer | null = null;
    try {
      r = GlRenderer.create(cv);
    } catch (e) {
      logWarning(t("msg.source.app"), t("viewer.webglFailed", { error: String(e) }));
      r = null;
    }
    rendererRef.current = r;
    setGlOk(r !== null);
    if (r) r.onRestored = () => setRedraw((n) => n + 1);
    return () => {
      r?.dispose();
      rendererRef.current = null;
    };
    // t は言語が変わっても作り直さない
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- カメラ ----

  const rulers = vs.overlays.rulers;
  const rulerPx = rulers ? rulerSize(RULER_FONTS[vs.rulerFont]) : 0;
  const fit = (w = size.w, h = size.h): Camera | null => {
    if (w <= 0 || h <= 0) return null;
    const b = drawingBounds(project);
    const c = fitCamera(b, Math.max(1, w - rulerPx), Math.max(1, h - rulerPx));
    return { ...c, ox: c.ox + rulerPx, oy: c.oy + rulerPx };
  };
  // 初回・別の文書・全体表示の要求・(まだ動かしていなければ) 大きさの変化で合わせる
  const fitKey = `${docSerial}|${vs.fitSerial}`;
  const lastFitKey = useRef("");
  useEffect(() => {
    if (size.w === 0 || size.h === 0) return;
    if (lastFitKey.current !== fitKey || camera === null || !userMovedRef.current) {
      lastFitKey.current = fitKey;
      userMovedRef.current = false;
      setCamera(fit());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitKey, size.w, size.h, rulerPx]);

  const zoomLimits = useMemo(() => {
    const f = fit(Math.max(size.w, 100), Math.max(size.h, 100));
    const s = f?.scale ?? 1;
    return { min: s / 200, max: s * 1e6 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.geometry.domain, size.w, size.h]);

  // ホイール (React の onWheel は passive なので自前で付ける)
  useEffect(() => {
    const el = topRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [sx, sy] = localPoint(e, el);
      const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
      const factor = Math.min(2, Math.max(0.5, Math.pow(ZOOM_STEP, -dy / 100)));
      userMovedRef.current = true;
      setCamera((c) => (c ? zoomAt(c, sx, sy, factor, zoomLimits) : c));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomLimits]);

  // ---- 表示するもの ----

  const playback = "playback" in scene ? scene.playback : undefined;
  // 塗る量が変わったら手動の範囲は外す (単位が違う)
  const fieldKey = scene.field ? `${scene.field.label}|${scene.field.unit}` : "";
  const lastFieldKey = useRef(fieldKey);
  useEffect(() => {
    if (lastFieldKey.current !== fieldKey) {
      if (vs.range.min !== null || vs.range.max !== null) vs.setRange({ min: null, max: null });
      // 読む量が変わったらプローブの点も消す (v1 と同じ)
      if (vs.probe) vs.setProbe(null);
    }
    lastFieldKey.current = fieldKey;
  }, [fieldKey, vs]);
  const log = controls.log;
  const range = useMemo(() => (scene.field ? resolveRange(statsOf(scene.field), log, vs.range) : null), [scene.field, log, vs.range]);
  const isoSegs = useMemo(() => {
    if (!vs.overlays.isolines || !scene.iso) return null;
    const iso = scene.iso;
    let levels: number[];
    if (iso === scene.field && range) {
      // 塗っている量と同じなら表示の範囲で (対数なら対数で等分)
      const lv = isoLevels(range.lo, range.hi, 15);
      levels = range.log ? lv.map((x) => Math.pow(10, x)) : lv;
    } else {
      const s = statsOf(iso);
      levels = isoLevels(s.min, s.max, 15);
    }
    return isolineSegments(iso.mesh, iso.values, levels);
  }, [vs.overlays.isolines, scene.iso, scene.field, range]);
  const placements = useMemo(() => placementsOf(project, vs.overlays), [project, vs.overlays]);
  const emitter = useMemo(() => (vs.overlays.emitter ? emitterOf(project) : null), [project, vs.overlays.emitter]);
  const injector = useMemo(() => (vs.overlays.emitter ? injectorOf(project) : null), [project, vs.overlays.emitter]);
  const amrBoxes = useMemo(() => amrBoxesOf(project), [project]);
  const origin = useMemo<[number, number]>(() => {
    const b = domainBounds(project);
    return [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.geometry.domain]);

  const tools = useCanvasTools({
    camera,
    setCamera,
    markUserMoved: () => {
      userMovedRef.current = true;
    },
    project,
    scene,
    placements: placements as Placement[],
    t,
  });
  const { cursor, hover, preview } = tools;
  // 表示しているレイヤのスケッチ (P7f)
  const sketch = useMemo(() => visibleSketch(project), [project]);

  // ---- 描く ----

  const axes = axisNames(project);
  const fieldUnit = scene.field?.unit ?? "";
  const probeLines = (pt: Point): string[] => {
    const unit = lengthUnitLabel(lengthUnit);
    const coord = `${axes[0]}: ${toDisplayLength(pt[0], lengthUnit).toFixed(3)} ${unit}, ${axes[1]}: ${toDisplayLength(pt[1], lengthUnit).toFixed(3)} ${unit}`;
    if (!scene.field) return [coord, t("viewer.noField")];
    const v = sampleField(scene.field, pt[0], pt[1]);
    return [coord, v === null ? t("viewer.outside") : `${scene.field.label} = ${formatNumber(v)} ${fieldUnit}`];
  };

  /** 3 枚の層を描く (書き出しでは WebGL の層を描いた直後に読むので同期で呼ぶ) */
  const paint = (): boolean => {
    // 大きさが変わっていたら測り直してから描く (隣の欄の開閉など。ResizeObserver が遅れても合わせる)
    const wrap = wrapRef.current;
    if (wrap) {
      const r = wrap.getBoundingClientRect();
      const w = Math.floor(r.width);
      const h = Math.floor(r.height);
      if (w !== size.w || h !== size.h) {
        setSize({ w, h });
        return false;
      }
    }
    if (!camera || size.w === 0 || size.h === 0) return false;
    {
      const colors = readColors();
      const dpr = window.devicePixelRatio || 1;
      const view = { camera, width: size.w, height: size.h };
      const base = baseRef.current;
      const bctx = base ? setCanvasSize(base, size.w, size.h, dpr) : null;
      if (bctx) drawBase(bctx, view, project, colors, vs.overlays.grid);

      const renderer = rendererRef.current;
      if (renderer) {
        const meshes: MeshDraw[] = [];
        if (scene.field && range) {
          meshes.push({
            mesh: scene.field.mesh,
            mode: "field",
            field: { values: scene.field.values, location: scene.field.location, range, colormap: vs.colormap, alpha: 1 },
            wire: vs.overlays.mesh ? { color: parseColor(colors.wireField), width: 1 } : null,
          });
        } else if (scene.mesh) {
          const regionColors: Rgba[] = project.geometry.regions.map((r) => parseColor(colors[r.type] ?? colors.edge, 0.16));
          meshes.push({ mesh: scene.mesh, mode: scene.fillRegions ? "regions" : "none", regionColors, wire: { color: parseColor(colors.wire), width: 1 } });
        }
        const segments: SegmentDraw[] = [];
        if (isoSegs && isoSegs.length) segments.push({ data: isoSegs, color: parseColor(colors.isoline), width: 1 });
        // 軌道 (と吸収点) は「軌道」、粒子は「粒子」で出し分ける。シース端はいつも
        for (const s of scene.segments) {
          if (s.id === "trajectories" && !vs.overlays.trajectories) continue;
          segments.push({ data: s.segments, color: parseColor(s.color), width: s.width });
        }
        const points: PointDraw[] = scene.points
          .filter((p) => (p.id === "absorbed" ? vs.overlays.trajectories : p.id === "sheathS" ? vs.overlays.sheathLines : vs.overlays.particles))
          .map((p) => ({ data: p.xy, color: parseColor(p.color), size: p.size }));
        renderer.render({ width: size.w, height: size.h, dpr, camera, origin, meshes, segments, points });
      }

      const top = topRef.current;
      const tctx = top ? setCanvasSize(top, size.w, size.h, dpr) : null;
      if (!tctx) return true;
      const probe = vs.tool === "probe" && vs.probe ? { pt: vs.probe, lines: probeLines(vs.probe) } : null;
      const state: OverlayState = {
        view,
        project,
        colors,
        lengthUnit,
        axisNames: axes,
        overlays: vs.overlays,
        fieldShown: scene.field !== null || scene.mesh !== null,
        selectedPlacement: selectedPlacement as OverlayState["selectedPlacement"],
        selectedEdge: selectedEdgeIndex(project, activeNode),
        picked: tools.picked,
        edit: tools.edit,
        sketch,
        fillPreview: tools.fillPreview,
        toolPreview: tools.toolPreview,
        snapMark: tools.snapMark,
        snapLabel: tools.snapMark ? t(`snapKind.${tools.snapMark.kind}`) : null,
        tool: vs.tool,
        preview,
        drawing: tools.drawing,
        placements: placements as Placement[],
        emitter,
        injector,
        amrBoxes,
        profile: vs.profile,
        measure: tools.measure,
        arrows: vs.overlays.vectors ? vectorArrows(scene, camera, size.w, size.h, rulerPx) : [],
        colorbar: scene.field && range ? { colormap: vs.colormap, range, label: scene.field.label, unit: scene.field.unit } : null,
        probe,
        cursor,
        rulerFont: RULER_FONTS[vs.rulerFont],
      };
      drawOverlay(tctx, state);
    }
    return true;
  };
  const paintRef = useRef(paint);
  useLayoutEffect(() => {
    paintRef.current = paint;
  });
  // 描き直しは React の反映ごと (イベントごとにまとまる)。requestAnimationFrame はウィンドウが隠れていると
  // 止まるので使わない
  useEffect(() => {
    paintRef.current();
  });

  const exportPng = async () => {
    const layers = [baseRef.current, glRef.current, topRef.current].filter((c): c is HTMLCanvasElement => c !== null);
    if (!paintRef.current() || layers.length === 0) return;
    try {
      const blob = await composePng(layers);
      await saveBinaryFile(`es-sim-view-${stamp()}.png`, blob, "png", "PNG");
    } catch (e) {
      logError(t("msg.source.app"), t("viewer.exportFailed", { error: errorText(e) }));
    }
  };
  const exportCsv = async () => {
    if (!scene.field) return;
    try {
      const name = (controls.quantity ?? scene.field.label).replace(/[^A-Za-z0-9_]+/g, "") || "field";
      await saveTextFile(`${name}-${stamp()}.csv`, fieldCsv(scene.field, axes), "csv", "CSV");
    } catch (e) {
      logError(t("msg.source.app"), t("viewer.exportFailed", { error: errorText(e) }));
    }
  };

  // ---- 表示 ----

  const hint = toolHint(vs.tool, tools.drawing, t, vs, lengthUnit);
  const unit = lengthUnitLabel(lengthUnit);
  const hoverValue = hover && scene.field ? sampleField(scene.field, hover[0], hover[1]) : null;
  const cursorText = cursor
    ? `${axes[0]}: ${toDisplayLength(cursor[0], lengthUnit).toFixed(3)} ${unit}  ${axes[1]}: ${toDisplayLength(cursor[1], lengthUnit).toFixed(3)} ${unit}`
    : "";
  const cls = `viewer-top${tools.panning ? " panning" : vs.tool === "select" ? "" : " crosshair"}`;

  return (
    <div className="viewer" data-scheme={theme}>
      <ViewerToolbar scene={scene} controls={controls} run={run} onExportPng={exportPng} onExportCsv={exportCsv} />
      <div ref={wrapRef} className="viewer-canvas">
        <canvas ref={baseRef} className="viewer-layer" aria-hidden="true" />
        <canvas ref={glRef} className="viewer-layer" aria-hidden="true" />
        <canvas
          ref={topRef}
          className={`viewer-layer ${cls}`}
          tabIndex={0}
          role="application"
          aria-label={t("viewer.canvasLabel")}
          onPointerDown={tools.onPointerDown}
          onMouseDown={(e) => {
            // 中ボタンの自動スクロールを出さない
            if (e.button === 1) e.preventDefault();
          }}
          onPointerMove={tools.onPointerMove}
          onPointerUp={tools.onPointerUp}
          onPointerCancel={tools.onPointerCancel}
          onPointerLeave={tools.onPointerLeave}
          onDoubleClick={tools.onDoubleClick}
          onKeyDown={tools.onKeyDown}
          onContextMenu={(e) => e.preventDefault()}
          data-redraw={redraw}
        />
        {(scene.title || scene.stale) && (
          <div className="viewer-title" style={{ left: rulerPx + 8, top: rulerPx + 6 }}>
            {scene.title && <span>{scene.title}</span>}
            {scene.stale && <span className="badge badge-warn">{t("viewer.stale")}</span>}
          </div>
        )}
        {!glOk && <div className="viewer-nogl">{t("viewer.noWebgl")}</div>}
        {tools.coordText !== null && (
          <CoordBox
            text={tools.coordText}
            error={tools.coordError}
            mode={tools.coordMode}
            at={camera && hover ? toScreen(camera, hover) : null}
            size={size}
            onChange={tools.setCoordText}
            onSubmit={() => {
              if (tools.submitCoord()) topRef.current?.focus({ preventScroll: true });
            }}
            onCancel={() => {
              tools.closeCoord();
              topRef.current?.focus({ preventScroll: true });
            }}
          />
        )}
        {vs.overlays.legend && (
          <div className="viewer-legend" aria-label={t("viewer.legend")}>
            <span>
              <i className="swatch dirichlet" />
              Dirichlet
            </span>
            <span>
              <i className="swatch symmetry" />
              {t("bc.symmetry")}
            </span>
            <span>
              <i className="swatch periodic" />
              {t("bc.periodic")}
            </span>
            {axisEdgeLegend(project, t)}
          </div>
        )}
      </div>
      {playback && <PlaybackBar playback={playback} />}
      {run && run.kind === "pic" && controls.mode === "live" && <ViewerRfStrip run={run} />}
      <div className="viewer-status">
        <span className="viewer-cursor mono">{cursorText}</span>
        {hoverValue !== null && scene.field && (
          <span className="viewer-hover mono">
            {scene.field.label} = {formatNumber(hoverValue)} {scene.field.unit}
          </span>
        )}
        <span className="spacer" />
        <span className="viewer-hint">{hint}</span>
      </div>
    </div>
  );
}

/** 数値入力の欄 (カーソルの右下、キャンバスの中に収める) */
function CoordBox({
  text,
  error,
  mode,
  at,
  size,
  onChange,
  onSubmit,
  onCancel,
}: {
  text: string;
  error: string | null;
  mode: CoordMode;
  at: Point | null;
  size: { w: number; h: number };
  onChange: (s: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation();
  const W = 280;
  const H = 56;
  const x = at ? Math.max(4, Math.min(at[0] + 16, size.w - W - 4)) : 8;
  const y = at ? Math.max(4, Math.min(at[1] + 16, size.h - H - 4)) : Math.max(4, size.h - H - 8);
  return (
    <div className="coord-input" style={{ left: x, top: y, width: W }}>
      <input
        className="input mono"
        autoFocus
        aria-label={t("coordInput.label")}
        value={text}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            onCancel();
          }
        }}
      />
      <div className={error ? "coord-error" : "coord-hint"}>{error ?? t(mode === "point" ? "coordInput.hint" : `coordInput.hint_${mode}`)}</div>
    </div>
  );
}

/** 選んだ外周の辺 (ツリーの "edge:<ID>") の番号 */
function selectedEdgeIndex(p: Project, node: string): number | null {
  if (!node.startsWith("edge:")) return null;
  const i = edgeIndexOf(p, node.slice(5));
  return i >= 0 ? i : null;
}

function axisEdgeLegend(p: Project, t: TFunction) {
  if (coordOf(p) === "xy") return null;
  return (
    <span>
      <i className="swatch axis" />
      {t("viewer.axis")}
    </span>
  );
}
