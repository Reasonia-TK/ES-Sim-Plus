# 98: 2D PIC — シースエッジの検出+可視化 (準中性度等値線 + ライン Brinkmann)

## 背景 (ユーザー要望)

「1Dと2Dの両方にsheath edgeを検出および可視化する機能を追加したい」
1D は prompts/97 で Brinkmann 実装済み。2D は本プロンプトで **両方**:

1. **準中性度の等値線**: n_e/n_i = α (既定 0.5、スライダ可変) の等値線を
   メッシュ全体で抽出してキャンバスに重ね描き (面全体で一意に定義できる)。
2. **ライン指定 Brinkmann**: 2点クリックの評価ライン (最大4本) 上で 1D と同じ
   Brinkmann 積分判定を行い、ライン上にシースエッジ位置 s のマーカーを表示。

どちらも**フロントエンド計算** (時間平均の n_e/n_i 節点配列も位相分解 cycle の
節点配列も既に done メッセージでフロントに届いているため。α スライダの即時反映と
保存済み結果ファイルでの動作のためにもフロント計算が適切 — 理由コメントに書く)。

## backend (schema のみ)

### schema.py

```py
class SheathLine(BaseModel):
    """2D シースエッジの Brinkmann 評価ライン (prompts/98)。計算はフロント側。
    p1 = 電極側、p2 = バルク側 (x_b = p2)。"""
    p1: Point
    p2: Point
    label: str = ""  # 空ならフロントが S1, S2... を振る

class PicSettings(...):
    # シースエッジ評価ライン (最大4)。可視化専用 (backend は永続化のみ)
    sheath_lines: list[SheathLine] = []
```

validator で最大4本。テスト: validator (5本で ValidationError)、既存プロジェクト
読込不変。`python -m pytest tests/ -q` 全件パス (現在 265 + 新規)。

## frontend

### 共通: Brinkmann 判定の TS 実装

`frontend/src/sheath.ts` (新規):
- `brinkmannSheathEdge(dist: number[], nE: number[], nI: number[]): number | null`
  — backend/es_sim/pic1d.py の `brinkmann_sheath_edge` と同じ式
  (G(s) = ∫₀ˢ n_e − ∫ₛ^{end} (n_i − n_e)、台形累積 + 線形補間根)。
  dist は p1 からの距離列 (末尾が x_b)。実装は pic1d.py を読んで忠実に移植。
- `marchingTrianglesContour(nodes, tris, values: number[], level: number,
  mask: (tri) => boolean): Array<[Point, Point]>` — 三角形ごとの等値線分抽出
  (辺上の線形補間。マスク false の三角形はスキップ)。
- ライン上のフィールド補間 `sampleAlongLine(nodes, tris, values, p1, p2, n=200)`
  — 各サンプル点の所属三角形を総当たりで探し重心座標補間 (20k要素×200点程度
  なので総当たりで十分。既存に点位置特定ヘルパがあれば流用)。三角形外
  (ドメイン外) のサンプルは NaN → Brinkmann では NaN 区間を除外して評価。

### 等値線オーバーレイ (CadCanvas)

- PIC 結果 (時間平均 n_e/n_i があるとき) と位相アニメーション中 (cycle の
  現在ビンの n_e/n_i) に、比 R = n_e / n_i の等値線 (level α) を描く。
  - ノイズマスク: 三角形の3節点すべてで n_i < 0.02 × max(n_i) の三角形は除外
    (ほぼ真空の領域で比が乱れて偽の等値線が出るため。閾値の理由コメント)。
  - 線色 #ffb454、実線 1.5px。表示トグル「シースエッジ」を既存の表示トグル群
    (メッシュ/エミッタ/コレクタ/ガス境界/メッシュ細分) に追加 (既定オン)。
  - α スライダ (0.05〜0.95、step 0.05、既定 0.5) — PicPanel の結果セクション
    (または既存トグル群の近く。UI 的に自然な方) に「準中性度 α」として配置。
    値は UI state で良い (project へ永続化しない)。
- 位相アニメ再生中はビン切替に追従して等値線を再計算 (200ms 毎程度の再生速度
  なら marching triangles は十分軽い。重ければ useMemo でビン別キャッシュ)。

### 評価ライン (Brinkmann)

- Tool "sheathline" 「シース評価線」ボタン (コレクタ/メッシュ細分の並び)。
  2点クリックで pic.sheath_lines に追加 (最大4、超過時はステータスヒント)。
  commitProject 経由 (Undo/Redo は既存の設定変更と同じ)。
  hint: 「1点目を電極側、2点目をバルク側に取ってください (Brinkmann 積分の
  参照点はライン終点)」。
- オーバーレイ: #ffb454 の破線 + ラベル S1, S2, ...。シースエッジ位置 s に
  ●マーカー (時間平均)。位相アニメ中は現在ビンの s に追従。
- PicPanel (結果セクション): ライン一覧 (ラベル / p1→p2 要約 (表示単位) /
  s の値 [表示単位] / 削除ボタン)。cycle があればライン毎の s(φ) 折れ線チャート
  (1D の Pic1dSheathPhaseChart の流儀。null ビンは線を切る)。
- 設定変更 (ライン追加/削除) は結果を破棄しない (可視化専用のため。既存の
  「結果を破棄する mesh 変更」とは区別されることを確認 — commitProject の
  結果破棄条件を読んで、sheath_lines 変更では破棄しないようにする)。

### types.ts

- `SheathLine`、`PicSettings.sheath_lines?: SheathLine[]`。

### 検証

- `cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。
- backend: `cd /home/claude/ES-Sim/backend && python -m pytest tests/ -q` 全件パス。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- rz / rz_x0 座標系でも密度は節点配列なのでそのまま動く (等値線・ラインとも
  幾何座標で評価。特別対応不要だが動作を妨げない)。
- Math.min/max スプレッド禁止 (arrayMin/arrayMax)。CommitNumberInput 系。
- 結果無し・n_i 全ゼロなどの縮退で例外を出さない (等値線ゼロ本・s=null 表示)。
