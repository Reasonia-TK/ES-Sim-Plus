// WebGL2 のシェーダ。三角形 (場の塗り・領域の塗り・メッシュの線)、太さのある線分 (等値線・軌道など)、点 (粒子)。
// 座標はワールド [m] から原点 (ドメインの中心) を引いた値を float32 で持ち、クリップ座標へは a·p + b で写す。

/** 三角形: 頂点は節点番号だけ (三角形ごとに 3 個並べる)。座標と値はテクスチャから読む */
export const TRI_VS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in uint a_node;
uniform sampler2D u_pos;
uniform sampler2D u_nval;
uniform sampler2D u_eval;
uniform sampler2D u_ereg;
uniform sampler2D u_regionColors;
uniform int u_texW;
uniform int u_mode;
uniform int u_log;
uniform float u_minPos;
uniform vec2 u_a;
uniform vec2 u_b;
out float v_val;
out float v_missing;
out vec3 v_bary;
out vec4 v_color;

ivec2 tc(int i) { return ivec2(i % u_texW, i / u_texW); }

void val(float v) {
  // 欠けた値 (NaN・無限大) はアップロード時に 3e38 にしてある
  if (v > 1.0e38) {
    v_missing = 1.0;
    v_val = 0.0;
  } else {
    v_missing = 0.0;
    v_val = u_log == 1 ? log(max(v, u_minPos)) * 0.43429448190325176 : v;
  }
}

void main() {
  int node = int(a_node);
  vec2 p = texelFetch(u_pos, tc(node), 0).xy;
  gl_Position = vec4(u_a * p + u_b, 0.0, 1.0);
  int k = gl_VertexID % 3;
  v_bary = vec3(k == 0 ? 1.0 : 0.0, k == 1 ? 1.0 : 0.0, k == 2 ? 1.0 : 0.0);
  int elem = gl_VertexID / 3;
  v_val = 0.0;
  v_missing = 0.0;
  v_color = vec4(0.0);
  if (u_mode == 1) {
    val(texelFetch(u_nval, tc(node), 0).r);
  } else if (u_mode == 2) {
    val(texelFetch(u_eval, tc(elem), 0).r);
  } else if (u_mode == 3) {
    int r = int(texelFetch(u_ereg, tc(elem), 0).r + 1.5);
    v_color = texelFetch(u_regionColors, ivec2(r, 0), 0);
  }
}
`;

export const TRI_FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_cmap;
uniform int u_mode;
uniform vec2 u_range;
uniform float u_alpha;
uniform int u_wire;
uniform vec4 u_wireColor;
uniform float u_wireHalf;
in float v_val;
in float v_missing;
in vec3 v_bary;
in vec4 v_color;
out vec4 o;

void main() {
  vec4 fill = vec4(0.0);
  if (u_mode == 1 || u_mode == 2) {
    if (v_missing < 0.001) {
      float span = u_range.y - u_range.x;
      float t = span == 0.0 ? 0.5 : clamp((v_val - u_range.x) / span, 0.0, 1.0);
      vec3 c = texture(u_cmap, vec2(t * (255.0 / 256.0) + 0.5 / 256.0, 0.5)).rgb;
      fill = vec4(c * u_alpha, u_alpha);
    }
  } else if (u_mode == 3) {
    fill = vec4(v_color.rgb * v_color.a, v_color.a);
  }
  if (u_wire == 1) {
    // 三角形の辺までの距離 [px] (重心座標とその微分から)。隣の三角形が反対側の半分を描く
    vec3 d = v_bary / max(fwidth(v_bary), vec3(1.0e-6));
    float dist = min(min(d.x, d.y), d.z);
    float a = u_wireColor.a * (1.0 - smoothstep(u_wireHalf - 0.5, u_wireHalf + 0.5, dist));
    fill = vec4(u_wireColor.rgb * a, a) + fill * (1.0 - a);
  }
  if (fill.a <= 0.0) discard;
  o = fill;
}
`;

/** 線分: 1 本 = 1 インスタンス (x0, y0, x1, y1)、画面上で幅 u_width [px] の四角形に広げる */
export const SEG_VS = `#version 300 es
precision highp float;
precision highp int;
in vec2 a_p0;
in vec2 a_p1;
uniform vec2 u_a;
uniform vec2 u_b;
uniform vec2 u_viewport;
uniform float u_width;

void main() {
  vec2 h = 0.5 * u_viewport;
  vec2 s0 = (u_a * a_p0 + u_b) * h;
  vec2 s1 = (u_a * a_p1 + u_b) * h;
  vec2 d = s1 - s0;
  float len = length(d);
  vec2 n = len > 0.0 ? vec2(-d.y, d.x) / len : vec2(0.0, 1.0);
  vec2 t = len > 0.0 ? d / len : vec2(1.0, 0.0);
  int id = gl_VertexID;
  float side = (id == 0 || id == 2) ? -1.0 : 1.0;
  bool end = id >= 2;
  // 端を線幅の半分だけ伸ばして、折れ線のつなぎ目の隙間を目立たなくする
  vec2 s = (end ? s1 + t * 0.5 * u_width : s0 - t * 0.5 * u_width) + n * side * 0.5 * u_width;
  gl_Position = vec4(s / h, 0.0, 1.0);
}
`;

export const COLOR_FS = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 o;
void main() {
  o = vec4(u_color.rgb * u_color.a, u_color.a);
}
`;

/** 点 (粒子): 丸く塗る */
export const POINT_VS = `#version 300 es
precision highp float;
in vec2 a_p;
uniform vec2 u_a;
uniform vec2 u_b;
uniform float u_size;
void main() {
  gl_Position = vec4(u_a * a_p + u_b, 0.0, 1.0);
  gl_PointSize = u_size;
}
`;

export const POINT_FS = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 o;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  o = vec4(u_color.rgb * u_color.a, u_color.a);
}
`;
