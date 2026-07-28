# 85: 指定範囲の EEDF/EEPF 取得

## 背景 (ユーザー要望)

「指定した範囲のeepf/eedfを取得する機能を追加してほしい」
プラズマ診断の定番で、指定した空間領域内の電子のエネルギー分布関数
(EEDF: f(E), ∫f dE = 1) と確率関数 (EEPF: f(E)/√E、Maxwellian なら
片対数で直線) を、PIC の時間平均区間で集計する。

## データモデル

### backend/es_sim/schema.py

```py
class EedfRegion(BaseModel):
    """EEDF/EEPF の集計領域 (軸平行矩形)。キャンバスの2点クリックで指定する。"""
    p1: Point
    p2: Point                     # 対角の2点 (順不同)
    label: str = ""
    bins: int = Field(100, ge=10, le=1000)
    e_max_ev: float | None = None  # None = 平均区間の最初の集計時に自動決定
class PicSettings(...):
    eedf_regions: list[EedfRegion] = []   # 最大4個 (validator)
```

## backend 集計 (es_sim/pic.py)

- コレクタ (IEDF/IADF) の集計と同じライフサイクル: **時間平均区間内の毎ステップ**、
  各領域について矩形内の電子を選び、重み付きエネルギーヒストグラムへ加算する。
  - 矩形内判定: min(p1,p2) ≦ x ≦ max(p1,p2) の軸平行判定 (ベクトル化、数行)。
  - エネルギー: E = ½ m_e |v|² / e [eV] (3成分)。
  - e_max_ev が None の場合、その領域の最初の集計ステップで
    「矩形内電子の最大エネルギー × 1.2」(電子がいなければ 30 eV) に確定する。
    グリッド確定後は範囲外 (E > e_max) は最終ビンに畳まず**オーバーフロー計数**
    に足す (分布の形を歪めないため)。
  - 蓄積: hist[bins] (Σw)、n_samples (ステップ数)、Σw、ΣwE (平均エネルギー用)、
    overflow_w。
- prepare_continue でコレクタ同様にリセット。
- done 出力 (server.py の done 組み立てに追加):
  ```
  eedf: [{label, e_centers: [...], f: [...],   # EEDF [eV^-1]、∫f dE = 1 に正規化
          mean_energy_ev, t_eff_ev,            # T_eff = (2/3)⟨E⟩
          total_weight, overflow_frac, n_samples}]
  ```
  Σw = 0 (電子が一度も入らなかった) 領域は f 全ゼロ+フラグで返す。
- 位相分解は対象外 (時間平均のみ)。

## 出力経路の追従

- **ResultsBundle** (frontend types.ts / App の保存・復元) の pic に `eedf` を追加。
- **batch.py / sweep.py** の結果組み立てにも同様に含める (server done と同じ変換を
  流用している構造なら自然に載るはず — 確認し、漏れるなら追加)。

## frontend

### ツール (CadCanvas)

- Tool "eedfbox" を追加。**矩形ツールと同じ2点クリック** (1点目→ラバーバンド矩形
  →2点目確定) で pic.eedf_regions に追加 (最大4、ラベル E1..E4 自動採番)。
  ツールバーのコレクタ/ガス境界の並びに「EEDF領域」ボタン。
  配置済み領域は紫系 (#c792ea) の破線矩形+ラベルで常時オーバーレイ
  (表示トグル群に「EEDF領域」を追加)。
  配置後は PIC-MCC スタディへ切替 (コレクタと同じ誘導)。

### PicPanel

- setup: 「PIC: EEDF/EEPF 領域 (最大4個)」セクション — 領域一覧
  (ラベル・座標要約・bins・e_max の編集、削除)。コレクタ設定 UI の流儀に合わせる。
- results: EEDF/EEPF のプロット (コレクタの HistogramChart の流儀):
  - 領域選択 select (複数領域があれば切替。または全領域を色分け重ね描き —
    実装しやすい方。重ね描き推奨: 領域比較が本機能の主目的のため)
  - 表示切替: EEDF / EEPF (EEPF = f/√E、E=0 ビンは除外)
  - 縦軸 log トグル (既定 ON — EEPF は片対数で見るのが定番)
  - T_eff / 平均エネルギー / オーバーフロー率の kv 表示
  - CSV保存 (E, f_EEDF, f_EEPF の列、領域ごと)
- 未実行時の文言等は既存の結果ページに合わせる。

## テスト (backend/tests/)

1. **Maxwellian 検証**: 温度 T の初期プラズマ (衝突・電場なし、全反射壁 or 周期) で
   domain 全体を覆う EEDF 領域を置き、得られた EEDF が Maxwellian
   f(E) = 2√(E/π) T^{-3/2} exp(−E/T) と数%で一致、t_eff_ev ≈ T (5%以内)。
2. 矩形選択: 電子を左右半分に別温度で置き、左半分のみの領域の T_eff が
   左側温度に一致 (右側に引っ張られない) こと。
3. 正規化: ∫f dE = 1 (数値積分で 1e-6 以内)。
4. 電子が入らない領域が安全に全ゼロで返ること。

`python -m pytest tests/ -q` 全件パス (現在 211)。

## 検証

- backend: pytest 全件。frontend: `npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」を書く既存スタイル。git commit はしない。
- 毎ステップの集計コストは矩形マスク+bincount で O(N_e)。ベンチへの影響が
  1〜2%を超えるなら報告 (領域未設定時はゼロコストであること)。
- ユーザーの perf 改修 (バッファ再利用等) と衝突しないよう pic.py は最小 diff。
