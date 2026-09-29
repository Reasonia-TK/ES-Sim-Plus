// 三角形メッシュの点の所属 (プローブ・ホバー用)。三角形を格子のバケツに入れておき、点の入るバケツの
// 三角形だけを調べる (v1 は全三角形を総当たりで、約 2 万要素までが目安だった)。

export interface TriMesh {
  nodes: [number, number][];
  triangles: [number, number, number][];
}

export interface Hit {
  tri: number;
  /** 重心座標 (節点の値の補間に使う) */
  w: [number, number, number];
}

export class MeshIndex {
  private readonly x0: number;
  private readonly y0: number;
  private readonly dx: number;
  private readonly dy: number;
  private readonly nx: number;
  private readonly ny: number;
  private readonly start: Int32Array;
  private readonly items: Int32Array;

  constructor(private readonly mesh: TriMesh) {
    const { nodes, triangles } = mesh;
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const [x, y] of nodes) {
      x0 = Math.min(x0, x);
      y0 = Math.min(y0, y);
      x1 = Math.max(x1, x);
      y1 = Math.max(y1, y);
    }
    const n = Math.max(1, triangles.length);
    const side = Math.max(1, Math.round(Math.sqrt(n / 2)));
    const w = Math.max(x1 - x0, 1e-300);
    const h = Math.max(y1 - y0, 1e-300);
    this.nx = Math.max(1, Math.round(side * Math.sqrt(w / h)));
    this.ny = Math.max(1, Math.round(side * Math.sqrt(h / w)));
    this.x0 = x0;
    this.y0 = y0;
    this.dx = w / this.nx;
    this.dy = h / this.ny;
    // 2 回走査: 数えてから詰める (CSR)
    const cells = this.nx * this.ny;
    const counts = new Int32Array(cells + 1);
    const range = (t: [number, number, number]) => {
      const xs = t.map((k) => nodes[k][0]);
      const ys = t.map((k) => nodes[k][1]);
      return [this.cx(Math.min(...xs)), this.cx(Math.max(...xs)), this.cy(Math.min(...ys)), this.cy(Math.max(...ys))];
    };
    for (const t of triangles) {
      const [i0, i1, j0, j1] = range(t);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) counts[j * this.nx + i + 1]++;
    }
    for (let c = 0; c < cells; c++) counts[c + 1] += counts[c];
    this.start = counts;
    this.items = new Int32Array(counts[cells]);
    const fill = counts.slice(0, cells);
    triangles.forEach((t, k) => {
      const [i0, i1, j0, j1] = range(t);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) this.items[fill[j * this.nx + i]++] = k;
    });
  }

  private cx(x: number): number {
    return Math.min(this.nx - 1, Math.max(0, Math.floor((x - this.x0) / this.dx)));
  }

  private cy(y: number): number {
    return Math.min(this.ny - 1, Math.max(0, Math.floor((y - this.y0) / this.dy)));
  }

  /** 点を含む三角形と重心座標 (外なら null) */
  locate(x: number, y: number): Hit | null {
    const { nodes, triangles } = this.mesh;
    if (x < this.x0 - this.dx || y < this.y0 - this.dy || x > this.x0 + (this.nx + 1) * this.dx || y > this.y0 + (this.ny + 1) * this.dy) return null;
    const c = this.cy(y) * this.nx + this.cx(x);
    const eps = 1e-12;
    for (let m = this.start[c]; m < this.start[c + 1]; m++) {
      const k = this.items[m];
      const [a, b, d] = triangles[k];
      const [ax, ay] = nodes[a];
      const [bx, by] = nodes[b];
      const [dx, dy] = nodes[d];
      const det = (by - dy) * (ax - dx) + (dx - bx) * (ay - dy);
      if (det === 0) continue;
      const w0 = ((by - dy) * (x - dx) + (dx - bx) * (y - dy)) / det;
      const w1 = ((dy - ay) * (x - dx) + (ax - dx) * (y - dy)) / det;
      const w2 = 1 - w0 - w1;
      if (w0 >= -eps && w1 >= -eps && w2 >= -eps) return { tri: k, w: [w0, w1, w2] };
    }
    return null;
  }

  /** 節点の値を点で補間 (外なら null) */
  interpolate(values: ArrayLike<number>, x: number, y: number): number | null {
    const h = this.locate(x, y);
    if (!h) return null;
    const [a, b, c] = this.mesh.triangles[h.tri];
    return h.w[0] * values[a] + h.w[1] * values[b] + h.w[2] * values[c];
  }
}
