# 90: 任意の辺 (線分) に沿ったローカルメッシュ細分化

## 背景 (ユーザー要望)

「ローカルメッシュの設定を任意の辺に与えられるようにしてほしい」
現状のローカルメッシュは領域単位 (mesh.local_sizes の region 指定) のみ。
電極エッジやドメイン辺の近傍だけ細かくしたいニーズに対し、**2点クリックで
置いた線分の近傍を指定サイズに細分化**できるようにする (シース解像などに有効)。

## データモデル

### backend/es_sim/schema.py

```py
class EdgeMeshSize(BaseModel):
    """線分近傍のローカルメッシュサイズ (prompts/90)。gmsh の距離場で
    線分から dist_in までは size、dist_out で全体サイズへ線形に戻す。"""
    p1: Point
    p2: Point
    size: float = Field(..., gt=0)
    # 遷移距離 (None は自動: dist_in = 2·size、dist_out = 8·size 程度を実装側で)
    dist_in: float | None = Field(None, gt=0)
    dist_out: float | None = Field(None, gt=0)
class MeshSettings(...):
    local_edge_sizes: list[EdgeMeshSize] = []
```

## メッシュ生成 (backend/es_sim/meshing.py)

- 非構造 (gmsh) モードのみ対応。gmsh の **Distance + Threshold フィールド**で実装:
  - 線分ごとに: 補助点列 (線分上に size 間隔でサンプル) を Distance フィールドの
    PointsList にする方式か、gmsh の曲線を作らず `Distance` の `SamplingPoints`
    を使う方式 — gmsh API で最も安定な方法を選ぶ (幾何エンティティとして
    線分を fragment に混ぜると領域トポロジを壊しうるので、**フィールドのみ**で
    実現し、既存のジオメトリ構築には一切手を入れない)。
  - Threshold: SizeMin = entry.size (dist ≤ dist_in)、SizeMax = 全体サイズ
    (dist ≥ dist_out)。dist_in/out の自動値は 2·size / 8·size。
  - 複数エントリ + 既存の領域ローカルサイズと共存: 最終的に
    `Min` フィールドで合成して Background Mesh に設定する。
    既存の領域ローカルサイズの実装方式 (おそらく点のサイズ指定 or フィールド) を
    読み、**既存の挙動を変えずに** Min 合成へ統合する
    (local_edge_sizes が空なら従来経路と完全一致 = ビット不変)。
- 構造格子モードでは無視し、警告を warnings 的な仕組みがあれば出す
  (無ければ docstring/hint 記載のみで可)。

## frontend

### ツール (CadCanvas + App)

- Tool "meshref" 「メッシュ細分」ボタン (コレクタ/ガス境界の並び)。
  2点クリックで mesh.local_edge_sizes に追加 (size 既定 = mesh.size / 4)。
  commitProject 経由 (Undo/Redo・結果破棄は既存の mesh 変更と同じ)。
- オーバーレイ: 水色系 (#59c2ff) の破線線分+ラベル M1, M2, ...。
  表示トグル群に「メッシュ細分」を追加。

### FieldPanel (mesh セクション)

- 「辺ローカルサイズ」一覧: 行 = ラベル / p1→p2 要約 (表示単位) / size 入力
  (CommitNumberInput、表示単位変換) / 削除ボタン。
  hint: 「キャンバスの「メッシュ細分」ツールで2点クリックでも追加できます。
  線分近傍が指定サイズに細分化されます (非構造メッシュのみ)」。
- dist_in/dist_out の UI は出さない (自動値。上級者は JSON 直編集で可、hint 記載不要)。

### types.ts

- EdgeMeshSize / MeshSettings.local_edge_sizes。

## テスト (backend/tests/)

1. 細分化の実効性: 一様 domain に線分エントリ (size = 全体の 1/5) を置いて
   メッシュ生成し、線分近傍 (dist_in 内) の要素の代表寸法 sqrt(2A) の中央値が
   全体領域の中央値より十分小さい (≦ 1/2.5 程度) こと。
2. 空リストで従来とビット不変 (既存メッシュ依存テストが全パスすることでも担保)。
3. 領域ローカルサイズとの共存: 両方指定して両方効くこと (それぞれの近傍で細かい)。
4. validator: size ≤ 0 が ValidationError。

`python -m pytest tests/ -q` 全件パス (現在 222)。

## 検証

- backend: pytest 全件。frontend: `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- 保存/読込は project.mesh に自然に含まれる (確認のみ)。
- 単位表示は lengthUnit (mm/µm) に追従させる。
