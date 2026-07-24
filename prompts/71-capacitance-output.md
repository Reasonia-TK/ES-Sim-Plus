# 71: 静電場ソルブの静電容量出力

## 背景 (ユーザー要望)

「静電場計算時に静電容量を出力してほしい」

## 方針 (物理)

- 各電極の誘起電荷 Q を **FEM の反力 (残差) 法**で計算する:
  組み立て済みの全体剛性 K (BC適用前) と右辺 f に対し r = Kφ − f。
  Dirichlet 節点上の r がその節点の集中電荷 (弱形式の ∮ε∂φ/∂n) に等しい。
  電極ごとに節点残差を合計して Q とする。フラックス積分より精度が高い標準手法。
- 単位: 平面2D (xy) は [C/m] (奥行き1m あたり)。軸対称 (rz/rz_x0) は [C] —
  ただし fem.py の軸対称剛性は 2π を含まない r̄ 重み (energy 計算では 2π を掛けている)
  ので、**残差にも 2π を掛けて**物理電荷にすること。
- 周期境界: mesh.periodic_map でスレーブ行がマスターへ寄っているため、
  残差はマスター節点で評価すれば良い (スレーブ側は 0)。実装時に確認。
- **静電容量 C**: 電極の電位が**ちょうど2水準** (V_hi > V_lo) で、かつ
  空間電荷 ρ が全域 0 の場合のみ定義し、C = Q_hi / (V_hi − V_lo)
  (Q_hi は高電位側電極群の合計電荷)。それ以外 (3水準以上、ρ≠0) は null。
  単位: xy → [F/m]、軸対称 → [F]。

## 電極のグルーピング

「電極」は次の2種の集まり:
1. Dirichlet 境界条件エントリ (domain 外周のエッジ)。ラベルは
   「下辺」「右辺」等 (フロント表示は英語キーでなく edge index を返し
   フロント側で辺名にするのでも良い — 実装しやすい方で。ただし schema には
   label: str を持たせ、backend 側で "edge0" 等を入れておけばフロントで変換可能)
2. conductor 領域 (voltage 指定)。ラベルは region id。

mesh (meshing.py の Mesh) の dirichlet が node→voltage の dict しか持たない場合、
電極への帰属は**幾何で再構成**する: domain エッジ上の節点 (線分上判定、tol は
メッシュサイズ比例)、conductor 領域内/境界上の節点 (polygon/circle 判定)。
meshing.py 側に既に帰属情報 (どのBC/領域から来た節点か) があればそれを使う方が
確実なので、まず meshing.py を読んで確認すること。両方に載る節点が出ないよう
優先順位 (領域 > エッジ) を決めて重複を除く。

## backend 変更

- `es_sim/fem.py`: solve() (または新ヘルパ) で電極ごとの電荷を計算し
  Solution に追加 (`charges: list[(label, voltage, q)]` 相当)。
- `es_sim/schema.py`:
  ```py
  class ElectrodeCharge(BaseModel):
      label: str      # "edge0".."edge3" または region id
      voltage: float  # 電極電位 [V]
      q: float        # 誘起電荷 [C/m] (xy) / [C] (軸対称)
  class SolveResult(...):
      charges: list[ElectrodeCharge] = []
      capacitance: float | None = None  # 2電極系のみ。[F/m] (xy) / [F] (軸対称)
  ```
- `es_sim/server.py` (または postprocess): SolveResult 組み立てに追従。

## frontend 変更

- `types.ts`: SolveResult に charges / capacitance を追加 (optional で後方互換)。
- `App.tsx` の ResultSummary: エネルギー表示の並びに追加:
  - 静電容量: `C = 1.234e-12 F/m` (軸対称なら F)。null なら
    「静電容量: - (2電極系のみ)」等の表示。
  - 電極電荷: 電極ごとに `ラベル (V値V): Q値 C/m` の kv 行。ラベル "edge0".."edge3" は
    FieldPanel の EDGE_LABELS (座標系対応) で辺名に変換して表示。
  - 表示は formatNumber / toExponential(3) 等既存の流儀に合わせる。

## テスト (backend/tests/)

1. **平行平板 (xy)**: 上下辺 Dirichlet (0/100V)、左右 Neumann、真空。
   解は線形場なので P1 で厳密 → C = ε0·W/d に相対誤差 1e-10 以下で一致、
   上下電極の Q が ±で相殺 (|Q_hi + Q_lo| ≪ |Q_hi|)。
2. **誘電体充填**: 同上で全域 eps_r=4 の誘電体領域 → C が4倍。
3. **同軸円筒 (rz_x0 等)**: 内導体 r=a (conductor 領域)、外周 r=b Dirichlet →
   C = 2πε0 L / ln(b/a) に相対誤差 3% 以内。
4. **3電極 or ρ≠0**: capacitance が null になること。

`python -m pytest tests/ -q` 全件パス (現在 138)。

## 検証

- backend: pytest 全件。frontend: `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル (残差法・2π・単位の理由を書く)。
- git commit はしない。
