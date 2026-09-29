// ジオメトリの簡易表示 (P6a、Canvas2D)。ドメイン・境界条件の辺の色分け・対称軸・領域を描き、クリックで
// 領域を選ぶ (小さい領域を優先)。拡大・移動・場の表示・CAD は P6c の WebGL ビューアで。

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useDocument } from "../model/documentStore";
import {
  axisEdge,
  boundaryOfEdge,
  polygonBounds,
  regionArea,
  regionOutline,
  type Point,
  type Project,
  type Region,
} from "../model/project";
import { useSelection } from "../model/selection";
import { usePrefs } from "../prefs/prefs";

interface View {
  scale: number;
  ox: number;
  oy: number;
}

function fitView(p: Project, w: number, h: number): View {
  const b = polygonBounds(p.geometry.domain.polygon);
  const bw = Math.max(b.x1 - b.x0, 1e-12);
  const bh = Math.max(b.y1 - b.y0, 1e-12);
  const scale = Math.min((w * 0.86) / bw, (h * 0.86) / bh);
  return { scale, ox: w / 2 - ((b.x0 + b.x1) / 2) * scale, oy: h / 2 + ((b.y0 + b.y1) / 2) * scale };
}

const toScreen = (v: View, [x, y]: Point): Point => [v.ox + x * v.scale, v.oy - y * v.scale];
const toWorld = (v: View, sx: number, sy: number): Point => [(sx - v.ox) / v.scale, (v.oy - sy) / v.scale];

function insidePolygon(poly: Point[], [x, y]: Point): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function hitRegion(p: Project, pt: Point): Region | null {
  const hits = p.geometry.regions.filter((r) =>
    r.shape ? Math.hypot(pt[0] - r.shape.center[0], pt[1] - r.shape.center[1]) <= r.shape.radius : insidePolygon(r.polygon ?? [], pt),
  );
  hits.sort((a, b) => regionArea(a) - regionArea(b));
  return hits[0] ?? null;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

function draw(ctx: CanvasRenderingContext2D, p: Project, v: View, selected: string | null, w: number, h: number) {
  ctx.clearRect(0, 0, w, h);
  const poly = p.geometry.domain.polygon.map((q) => toScreen(v, q));
  // ドメイン
  ctx.beginPath();
  poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = cssVar("--canvas-domain");
  ctx.fill();
  // 辺 (境界条件で色分け)
  const axis = axisEdge(p);
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const bc = boundaryOfEdge(p, i);
    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = cssVar("--canvas-edge");
    if (axis === i) {
      ctx.strokeStyle = cssVar("--color-axis");
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 4, 2, 4]);
    } else if (bc?.type === "dirichlet") {
      ctx.strokeStyle = cssVar("--color-dirichlet");
      ctx.lineWidth = 3;
    } else if (bc?.type === "symmetry") {
      ctx.strokeStyle = cssVar("--color-symmetry");
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
    } else if (bc?.type === "periodic") {
      ctx.strokeStyle = cssVar("--color-periodic");
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
    }
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.restore();
  }
  // 領域
  for (const r of p.geometry.regions) {
    const color = cssVar(`--color-${r.type}`);
    ctx.beginPath();
    if (r.shape) {
      const [cx, cy] = toScreen(v, r.shape.center);
      ctx.arc(cx, cy, r.shape.radius * v.scale, 0, 2 * Math.PI);
    } else {
      (r.polygon ?? []).map((q) => toScreen(v, q)).forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
    }
    ctx.save();
    ctx.globalAlpha = 0.28;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
    ctx.lineWidth = r.id === selected ? 2.5 : 1.5;
    ctx.strokeStyle = r.id === selected ? cssVar("--color-selection") : color;
    if (r.id === selected) ctx.setLineDash([5, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    const outline = regionOutline(r, 32);
    const c = outline.reduce<Point>((s, q) => [s[0] + q[0] / outline.length, s[1] + q[1] / outline.length], [0, 0]);
    const [lx, ly] = toScreen(v, r.shape ? r.shape.center : c);
    ctx.fillStyle = cssVar("--text");
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(r.id, lx, ly);
  }
}

export function GeometryView() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const selected = useSelection((s) => s.selectedRegion);
  const selectRegion = useSelection((s) => s.selectRegion);
  const theme = usePrefs((s) => s.theme);
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: Math.floor(e.contentRect.width), h: Math.floor(e.contentRect.height) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || size.w === 0 || size.h === 0) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size.w * dpr;
    cv.height = size.h * dpr;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, project, fitView(project, size.w, size.h), selected, size.w, size.h);
  }, [project, selected, size, theme]);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const v = fitView(project, size.w, size.h);
    const hit = hitRegion(project, toWorld(v, e.clientX - rect.left, e.clientY - rect.top));
    selectRegion(hit ? hit.id : null);
  };

  return (
    <div ref={wrapRef} className="geometry-view">
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h }}
        onClick={onClick}
        role="img"
        aria-label={t("graphics.geometry")}
      />
    </div>
  );
}
