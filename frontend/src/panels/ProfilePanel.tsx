import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { saveTextFile } from "../saveFile";
import { LENGTH_UNIT_LABEL, mToUnit } from "../units";
import type { LengthUnit } from "../units";
import type { Point, Project, ProfileResult } from "../types";
import { arrayMin, arrayMax } from "../mathUtils";

/**
 * ラインプロファイルパネル
 * - キャンバス上で確定した2点間の V / |E| 分布を折れ線グラフで表示する
 * - グラフは依存追加禁止のため canvas に直描きする
 * - 領域外 (null) の区間は線を切って表示する
 */

interface Props {
  project: Project;
  // 長さの表示単位 (mm/µm)。project 内部は常に m のまま
  lengthUnit: LengthUnit;
  p1: Point;
  p2: Point;
  onClose: () => void;
}

const V_COLOR = "#4da3ff";  // 電位 V (アクセント色)
const E_COLOR = "#ffb84d";  // |E| (別色)

const PAD_L = 52;
const PAD_R = 52;
const PAD_T = 14;
const PAD_B = 26;
const TICKS = 4;

// 描画に使ったスケールをホバー処理でも再利用するための情報
interface Scale {
  padL: number;
  plotW: number;
  sMinDisp: number; // 表示単位 (mm/µm) 換算後の s 最小値
  sMaxDisp: number; // 表示単位 (mm/µm) 換算後の s 最大値
}

export default function ProfilePanel({ project, lengthUnit, p1, p2, onClose }: Props) {
  const [data, setData] = useState<ProfileResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scaleRef = useRef<Scale | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [resizeTick, setResizeTick] = useState(0);

  // プロファイル線が変わるたびに /profile を呼び直す
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    api
      .profile(project, p1, p2, 200)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [project, p1, p2]);

  // ウィンドウリサイズでもグラフを再描画する
  useEffect(() => {
    const onResize = () => setResizeTick((n) => n + 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // グラフ描画
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = el.getBoundingClientRect();
    el.width = rect.width * dpr;
    el.height = rect.height * dpr;
    const ctx = el.getContext("2d")!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (!data || data.s.length === 0) {
      scaleRef.current = null;
      return;
    }

    const plotW = rect.width - PAD_L - PAD_R;
    const plotH = rect.height - PAD_T - PAD_B;

    const sMinDisp = mToUnit(data.s[0], lengthUnit);
    const sMaxDisp = mToUnit(data.s[data.s.length - 1], lengthUnit);
    const sRangeDisp = sMaxDisp - sMinDisp || 1;

    const vs = data.v.filter((x): x is number => x !== null);
    const es = data.e_abs.filter((x): x is number => x !== null);
    const vMin = vs.length ? arrayMin(vs) : 0;
    const vMax = vs.length ? arrayMax(vs) : 1;
    const eMin = 0;
    const eMax = es.length ? arrayMax(es) : 1;
    const vRange = vMax - vMin || 1;
    const eRange = eMax - eMin || 1;

    scaleRef.current = { padL: PAD_L, plotW, sMinDisp, sMaxDisp };

    const xOf = (sDisp: number) => PAD_L + ((sDisp - sMinDisp) / sRangeDisp) * plotW;
    const yOfV = (v: number) => PAD_T + plotH - ((v - vMin) / vRange) * plotH;
    const yOfE = (e: number) => PAD_T + plotH - ((e - eMin) / eRange) * plotH;

    // 枠
    ctx.strokeStyle = "#363c48";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD_L, PAD_T, plotW, plotH);

    // 目盛り線とラベル (横軸 s、左縦軸 V、右縦軸 |E|)
    ctx.font = "10px system-ui, sans-serif";
    for (let i = 0; i <= TICKS; i++) {
      const sDisp = sMinDisp + (sRangeDisp * i) / TICKS;
      const x = xOf(sDisp);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.beginPath();
      ctx.moveTo(x, PAD_T);
      ctx.lineTo(x, PAD_T + plotH);
      ctx.stroke();
      ctx.fillStyle = "#8a919e";
      ctx.textAlign = i === 0 ? "left" : i === TICKS ? "right" : "center";
      ctx.textBaseline = "top";
      ctx.fillText(sDisp.toFixed(2), x, PAD_T + plotH + 4);

      const frac = (TICKS - i) / TICKS;
      const v = vMin + vRange * frac;
      const yV = PAD_T + plotH * (i / TICKS);
      ctx.fillStyle = V_COLOR;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(v.toPrecision(3), PAD_L - 6, yV);

      const e = eMin + eRange * frac;
      const yE = PAD_T + plotH * (i / TICKS);
      ctx.fillStyle = E_COLOR;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(e.toExponential(1), PAD_L + plotW + 6, yE);
    }

    // 軸ラベル
    ctx.fillStyle = "#8a919e";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`s [${LENGTH_UNIT_LABEL[lengthUnit]}]`, PAD_L + plotW / 2, rect.height - 2);

    // 曲線 (null で線を分断)
    const drawCurve = (values: (number | null)[], yOf: (v: number) => number, color: string) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < values.length; i++) {
        const val = values[i];
        if (val === null) {
          started = false;
          continue;
        }
        const x = xOf(mToUnit(data.s[i], lengthUnit));
        const y = yOf(val);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    };
    drawCurve(data.v, yOfV, V_COLOR);
    drawCurve(data.e_abs, yOfE, E_COLOR);

    // ホバーカーソル (縦線)
    if (hoverX !== null && hoverX >= PAD_L && hoverX <= PAD_L + plotW) {
      ctx.strokeStyle = "rgba(255,255,255,0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, PAD_T);
      ctx.lineTo(hoverX, PAD_T + plotH);
      ctx.stroke();
    }
  }, [data, hoverX, resizeTick, lengthUnit]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const el = canvasRef.current;
    const sc = scaleRef.current;
    if (!el || !sc || !data) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setHoverX(x);
    const sRangeDisp = sc.sMaxDisp - sc.sMinDisp || 1;
    const sDisp = sc.sMinDisp + ((x - sc.padL) / sc.plotW) * sRangeDisp;
    let idx = Math.round(((sDisp - sc.sMinDisp) / sRangeDisp) * (data.s.length - 1));
    idx = Math.max(0, Math.min(data.s.length - 1, idx));
    setHoverIdx(idx);
  };

  const handleMouseLeave = () => {
    setHoverX(null);
    setHoverIdx(null);
  };

  const downloadCsv = () => {
    if (!data) return;
    const lines = ["s,v,e_abs"];
    for (let i = 0; i < data.s.length; i++) {
      const v = data.v[i];
      const e = data.e_abs[i];
      lines.push(`${data.s[i]},${v ?? ""},${e ?? ""}`);
    }
    saveTextFile("profile.csv", lines.join("\n"), "CSV", ["csv"]).catch((err) => {
      setError(String(err));
    });
  };

  return (
    <div className="profile-panel">
      <div className="profile-panel-header">
        <span className="profile-title">ラインプロファイル</span>
        <span className="profile-legend">
          <span className="swatch" style={{ background: V_COLOR }} /> V
          <span className="swatch" style={{ background: E_COLOR }} /> |E|
        </span>
        <div className="spacer" />
        <button className="secondary" onClick={downloadCsv} disabled={!data}>
          CSV保存
        </button>
        <button className="secondary" onClick={onClose}>
          閉じる
        </button>
      </div>
      <div className="profile-panel-body">
        {loading && <div className="muted">計算中...</div>}
        {error && <div className="error">{error}</div>}
        {!loading && !error && (
          <canvas
            ref={canvasRef}
            className="profile-canvas"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
        )}
        {hoverIdx !== null && data && (
          <div className="profile-hover">
            s: {mToUnit(data.s[hoverIdx], lengthUnit).toFixed(2)} {LENGTH_UNIT_LABEL[lengthUnit]} &nbsp;
            V: {data.v[hoverIdx] !== null ? data.v[hoverIdx]!.toFixed(2) : "-"} V &nbsp;
            |E|: {data.e_abs[hoverIdx] !== null ? data.e_abs[hoverIdx]!.toExponential(2) : "-"} V/m
          </div>
        )}
      </div>
    </div>
  );
}
