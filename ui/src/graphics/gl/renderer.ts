// WebGL2 の描画。三角形メッシュ (場の配色・領域の塗り・メッシュの線)、線分、点を 1 フレームに描く。
// GPU の資源 (座標・値のテクスチャ、バッファ) はデータのオブジェクトごとに作って使い回し、そのフレームで
// 使わなかったものは消す。値 → 色はシェーダの中で配色テクスチャから引く (線形/対数、範囲は uniform)。

import type { Camera } from "../camera";
import { colormapRgba, LUT_SIZE, type ColormapKey } from "../colormaps";
import type { DisplayRange } from "../fieldScale";
import type { FieldLocation, ViewMesh } from "../scene";
import { COLOR_FS, POINT_FS, POINT_VS, SEG_VS, TRI_FS, TRI_VS } from "./shaders";

/** 色 (0〜1、アルファは乗算前) */
export type Rgba = [number, number, number, number];

/** データテクスチャの幅 (高さは要素数に応じて) */
const TEX_W = 2048;
/** 欠けた値 (NaN・無限大) の印 */
const MISSING = 3.0e38;

export interface MeshDraw {
  mesh: ViewMesh;
  mode: "none" | "field" | "regions";
  field?: { values: ArrayLike<number>; location: FieldLocation; range: DisplayRange; colormap: ColormapKey; alpha: number };
  /** 領域の番号 → 色 (mode = regions) */
  regionColors?: Rgba[];
  wire?: { color: Rgba; width: number } | null;
}

export interface SegmentDraw {
  /** x0, y0, x1, y1, … [m] (同じオブジェクトなら送り直さない) */
  data: ArrayLike<number>;
  color: Rgba;
  width: number;
}

export interface PointDraw {
  data: ArrayLike<number>;
  color: Rgba;
  size: number;
}

export interface Frame {
  /** CSS px */
  width: number;
  height: number;
  dpr: number;
  camera: Camera;
  /** 座標から引く原点 (float32 の桁落ちを避ける) */
  origin: [number, number];
  meshes: MeshDraw[];
  segments: SegmentDraw[];
  points: PointDraw[];
}

interface Program {
  prog: WebGLProgram;
  u: Record<string, WebGLUniformLocation | null>;
}

interface MeshRes {
  posTex: WebGLTexture;
  eregTex: WebGLTexture | null;
  buf: WebGLBuffer;
  vao: WebGLVertexArrayObject;
  count: number;
}

interface BufRes {
  buf: WebGLBuffer;
  vao: WebGLVertexArrayObject;
  count: number;
}

const UNIT = { pos: 0, nval: 1, eval: 2, ereg: 3, regionColors: 4, cmap: 5 } as const;

function compile(gl: WebGL2RenderingContext, vs: string, fs: string, attribs: string[], uniforms: string[]): Program {
  const shader = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) {
      throw new Error(`shader: ${gl.getShaderInfoLog(s)}`);
    }
    return s;
  };
  const prog = gl.createProgram()!;
  const v = shader(gl.VERTEX_SHADER, vs);
  const f = shader(gl.FRAGMENT_SHADER, fs);
  gl.attachShader(prog, v);
  gl.attachShader(prog, f);
  attribs.forEach((a, i) => gl.bindAttribLocation(prog, i, a));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS) && !gl.isContextLost()) {
    throw new Error(`program: ${gl.getProgramInfoLog(prog)}`);
  }
  gl.deleteShader(v);
  gl.deleteShader(f);
  const u: Program["u"] = {};
  for (const name of uniforms) u[name] = gl.getUniformLocation(prog, name);
  return { prog, u };
}

/** 数値の配列 → float32 (欠けた値は印に) */
function toFloat32(values: ArrayLike<number>): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    out[i] = Number.isFinite(v) ? v : MISSING;
  }
  return out;
}

export class GlRenderer {
  private tri!: Program;
  private seg!: Program;
  private pt!: Program;
  private dummy!: WebGLTexture;
  private lost = false;
  private origin: [number, number] = [0, 0];
  private maxTex = 4096;
  private meshes = new Map<ViewMesh, MeshRes>();
  private values = new Map<object, WebGLTexture>();
  private segs = new Map<object, BufRes>();
  private pts = new Map<object, BufRes>();
  private cmaps = new Map<ColormapKey, WebGLTexture>();
  private regionTex: WebGLTexture | null = null;
  private regionKey = "";
  /** 文脈が戻ったとき (呼び出し側で描き直す) */
  onRestored: (() => void) | null = null;

  static create(canvas: HTMLCanvasElement): GlRenderer | null {
    let gl: WebGL2RenderingContext | null = null;
    try {
      gl = canvas.getContext("webgl2", { antialias: true, alpha: true, premultipliedAlpha: true, preserveDrawingBuffer: true });
    } catch {
      gl = null;
    }
    return gl ? new GlRenderer(canvas, gl) : null;
  }

  private constructor(
    readonly canvas: HTMLCanvasElement,
    readonly gl: WebGL2RenderingContext,
  ) {
    canvas.addEventListener("webglcontextlost", this.handleLost);
    canvas.addEventListener("webglcontextrestored", this.handleRestored);
    this.init();
  }

  private handleLost = (e: Event) => {
    e.preventDefault();
    this.lost = true;
  };

  private handleRestored = () => {
    this.meshes.clear();
    this.values.clear();
    this.segs.clear();
    this.pts.clear();
    this.cmaps.clear();
    this.regionTex = null;
    this.regionKey = "";
    this.init();
    this.lost = false;
    this.onRestored?.();
  };

  private init(): void {
    const gl = this.gl;
    this.maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    this.tri = compile(gl, TRI_VS, TRI_FS, ["a_node"], [
      "u_pos", "u_nval", "u_eval", "u_ereg", "u_regionColors", "u_cmap", "u_texW", "u_mode", "u_log", "u_minPos",
      "u_a", "u_b", "u_range", "u_alpha", "u_wire", "u_wireColor", "u_wireHalf",
    ]);
    this.seg = compile(gl, SEG_VS, COLOR_FS, ["a_p0", "a_p1"], ["u_a", "u_b", "u_viewport", "u_width", "u_color"]);
    this.pt = compile(gl, POINT_VS, POINT_FS, ["a_p"], ["u_a", "u_b", "u_size", "u_color"]);
    gl.useProgram(this.tri.prog);
    for (const [name, unit] of Object.entries(UNIT)) gl.uniform1i(this.tri.u[`u_${name}`], unit);
    this.dummy = this.dataTexture(new Float32Array([0]), 1);
  }

  /** 使っている GPU の資源を消す (文脈はキャンバスと一緒に捨てられる) */
  dispose(): void {
    this.canvas.removeEventListener("webglcontextlost", this.handleLost);
    this.canvas.removeEventListener("webglcontextrestored", this.handleRestored);
    if (this.lost) return;
    this.sweep(new Set());
    const gl = this.gl;
    for (const t of this.cmaps.values()) gl.deleteTexture(t);
    this.cmaps.clear();
    if (this.regionTex) gl.deleteTexture(this.regionTex);
    this.regionTex = null;
    gl.deleteTexture(this.dummy);
    gl.deleteProgram(this.tri.prog);
    gl.deleteProgram(this.seg.prog);
    gl.deleteProgram(this.pt.prog);
  }

  private dataTexture(data: Float32Array, comps: 1 | 2): WebGLTexture {
    const gl = this.gl;
    const n = Math.max(1, Math.ceil(data.length / comps));
    const w = Math.min(TEX_W, this.maxTex);
    const h = Math.max(1, Math.ceil(n / w));
    if (h > this.maxTex) throw new Error(`too many elements for a texture: ${n}`);
    let buf = data;
    if (data.length !== w * h * comps) {
      buf = new Float32Array(w * h * comps);
      buf.set(data);
    }
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, comps === 1 ? gl.R32F : gl.RG32F, w, h, 0, comps === 1 ? gl.RED : gl.RG, gl.FLOAT, buf);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private meshRes(mesh: ViewMesh): MeshRes {
    const hit = this.meshes.get(mesh);
    if (hit) return hit;
    const gl = this.gl;
    const [ox, oy] = this.origin;
    const pos = new Float32Array(2 * mesh.nodes.length);
    mesh.nodes.forEach(([x, y], i) => {
      pos[2 * i] = x - ox;
      pos[2 * i + 1] = y - oy;
    });
    const idx = new Uint32Array(3 * mesh.triangles.length);
    mesh.triangles.forEach(([a, b, c], i) => {
      idx[3 * i] = a;
      idx[3 * i + 1] = b;
      idx[3 * i + 2] = c;
    });
    const buf = gl.createBuffer()!;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribIPointer(0, 1, gl.UNSIGNED_INT, 0, 0);
    gl.bindVertexArray(null);
    const ereg = mesh.regionOfTriangle ? this.dataTexture(Float32Array.from(mesh.regionOfTriangle), 1) : null;
    const res: MeshRes = { posTex: this.dataTexture(pos, 2), eregTex: ereg, buf, vao, count: idx.length };
    this.meshes.set(mesh, res);
    return res;
  }

  private valueTex(values: ArrayLike<number>): WebGLTexture {
    const key = values as object;
    let tex = this.values.get(key);
    if (!tex) {
      tex = this.dataTexture(toFloat32(values), 1);
      this.values.set(key, tex);
    }
    return tex;
  }

  private cmapTex(key: ColormapKey): WebGLTexture {
    let tex = this.cmaps.get(key);
    if (tex) return tex;
    const gl = this.gl;
    tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, LUT_SIZE, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colormapRgba(key));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.cmaps.set(key, tex);
    return tex;
  }

  /** 領域の色の表 (0 番 = 背景は透明、i + 1 番 = 領域 i) */
  private regionColorTex(colors: Rgba[]): WebGLTexture {
    const key = JSON.stringify(colors);
    const gl = this.gl;
    if (this.regionTex && key === this.regionKey) return this.regionTex;
    if (this.regionTex) gl.deleteTexture(this.regionTex);
    const w = colors.length + 1;
    const data = new Uint8Array(4 * w);
    colors.forEach((c, i) => c.forEach((v, k) => (data[4 * (i + 1) + k] = Math.round(Math.min(1, Math.max(0, v)) * 255))));
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.regionTex = tex;
    this.regionKey = key;
    return tex;
  }

  private bufRes(map: Map<object, BufRes>, data: ArrayLike<number>, instanced: boolean): BufRes {
    const key = data as object;
    const hit = map.get(key);
    if (hit) return hit;
    const gl = this.gl;
    const [ox, oy] = this.origin;
    const rel = new Float32Array(data.length);
    for (let i = 0; i < data.length; i += 2) {
      rel[i] = data[i] - ox;
      rel[i + 1] = data[i + 1] - oy;
    }
    const buf = gl.createBuffer()!;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, rel, gl.STATIC_DRAW);
    if (instanced) {
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
      gl.vertexAttribDivisor(0, 1);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
      gl.vertexAttribDivisor(1, 1);
    } else {
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    }
    gl.bindVertexArray(null);
    const res = { buf, vao, count: instanced ? Math.floor(data.length / 4) : Math.floor(data.length / 2) };
    map.set(key, res);
    return res;
  }

  /** このフレームで使わなかった資源を消す */
  private sweep(used: Set<object>): void {
    const gl = this.gl;
    for (const [k, r] of this.meshes) {
      if (used.has(k)) continue;
      gl.deleteTexture(r.posTex);
      if (r.eregTex) gl.deleteTexture(r.eregTex);
      gl.deleteBuffer(r.buf);
      gl.deleteVertexArray(r.vao);
      this.meshes.delete(k);
    }
    for (const [k, t] of this.values) {
      if (used.has(k)) continue;
      gl.deleteTexture(t);
      this.values.delete(k);
    }
    for (const map of [this.segs, this.pts]) {
      for (const [k, r] of map) {
        if (used.has(k)) continue;
        gl.deleteBuffer(r.buf);
        gl.deleteVertexArray(r.vao);
        map.delete(k);
      }
    }
  }

  render(f: Frame): void {
    if (this.lost || this.gl.isContextLost()) return;
    const gl = this.gl;
    const W = Math.max(1, Math.round(f.width * f.dpr));
    const H = Math.max(1, Math.round(f.height * f.dpr));
    if (this.canvas.width !== W) this.canvas.width = W;
    if (this.canvas.height !== H) this.canvas.height = H;
    if (f.origin[0] !== this.origin[0] || f.origin[1] !== this.origin[1]) {
      // 原点が変わったら座標を送り直す
      this.sweep(new Set());
      this.origin = [f.origin[0], f.origin[1]];
    }
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const c = f.camera;
    const s = c.scale;
    const a: [number, number] = [(2 * s) / f.width, (2 * s) / f.height];
    const b: [number, number] = [(2 * (c.ox + f.origin[0] * s)) / f.width - 1, 1 - (2 * (c.oy - f.origin[1] * s)) / f.height];
    const used = new Set<object>();
    for (const m of f.meshes) this.drawMesh(m, a, b, f.dpr, used);
    for (const sg of f.segments) {
      if (sg.data.length < 4) continue;
      const r = this.bufRes(this.segs, sg.data, true);
      used.add(sg.data as object);
      gl.useProgram(this.seg.prog);
      gl.bindVertexArray(r.vao);
      gl.uniform2f(this.seg.u.u_a, a[0], a[1]);
      gl.uniform2f(this.seg.u.u_b, b[0], b[1]);
      gl.uniform2f(this.seg.u.u_viewport, W, H);
      gl.uniform1f(this.seg.u.u_width, sg.width * f.dpr);
      gl.uniform4fv(this.seg.u.u_color, sg.color);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, r.count);
    }
    for (const p of f.points) {
      if (p.data.length < 2) continue;
      const r = this.bufRes(this.pts, p.data, false);
      used.add(p.data as object);
      gl.useProgram(this.pt.prog);
      gl.bindVertexArray(r.vao);
      gl.uniform2f(this.pt.u.u_a, a[0], a[1]);
      gl.uniform2f(this.pt.u.u_b, b[0], b[1]);
      gl.uniform1f(this.pt.u.u_size, Math.max(1, p.size * f.dpr));
      gl.uniform4fv(this.pt.u.u_color, p.color);
      gl.drawArrays(gl.POINTS, 0, r.count);
    }
    gl.bindVertexArray(null);
    this.sweep(used);
  }

  private bindTex(unit: number, tex: WebGLTexture): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  private drawMesh(m: MeshDraw, a: [number, number], b: [number, number], dpr: number, used: Set<object>): void {
    const gl = this.gl;
    if (m.mesh.triangles.length === 0) return;
    // テクスチャを作るとき今のユニットの結び付きが変わるので、全部そろえてから結び付ける
    const res = this.meshRes(m.mesh);
    used.add(m.mesh);
    const field = m.mode === "field" && m.field ? m.field : null;
    const valTex = field ? this.valueTex(field.values) : null;
    if (field) used.add(field.values as object);
    const regionTex = this.regionColorTex(m.regionColors ?? []);
    const cmapTex = this.cmapTex(field?.colormap ?? "viridis");
    const u = this.tri.u;
    gl.useProgram(this.tri.prog);
    gl.bindVertexArray(res.vao);
    gl.uniform2f(u.u_a, a[0], a[1]);
    gl.uniform2f(u.u_b, b[0], b[1]);
    gl.uniform1i(u.u_texW, Math.min(TEX_W, this.maxTex));
    this.bindTex(UNIT.pos, res.posTex);
    this.bindTex(UNIT.nval, field?.location === "node" && valTex ? valTex : this.dummy);
    this.bindTex(UNIT.eval, field?.location === "element" && valTex ? valTex : this.dummy);
    this.bindTex(UNIT.ereg, res.eregTex ?? this.dummy);
    this.bindTex(UNIT.regionColors, regionTex);
    this.bindTex(UNIT.cmap, cmapTex);
    let mode = 0;
    if (field) {
      mode = field.location === "node" ? 1 : 2;
      const r = field.range;
      gl.uniform2f(u.u_range, r.lo, r.hi);
      gl.uniform1i(u.u_log, r.log ? 1 : 0);
      gl.uniform1f(u.u_minPos, Number.isFinite(r.minPositive) ? r.minPositive : 1e-30);
      gl.uniform1f(u.u_alpha, field.alpha);
    } else if (m.mode === "regions" && res.eregTex) {
      mode = 3;
    }
    gl.uniform1i(u.u_mode, mode);
    gl.uniform1i(u.u_wire, m.wire ? 1 : 0);
    if (m.wire) {
      gl.uniform4fv(u.u_wireColor, m.wire.color);
      gl.uniform1f(u.u_wireHalf, 0.5 * m.wire.width * dpr);
    }
    if (mode === 0 && !m.wire) return;
    gl.drawArrays(gl.TRIANGLES, 0, res.count);
  }
}
