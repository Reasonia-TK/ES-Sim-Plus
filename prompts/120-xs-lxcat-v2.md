# 120: LXCat 完全対応 — 断面積パッケージ `es_sim/xs/` (backend、v2 フェーズ P1)

計画書: `prompts/119-v2-rebuild-plan.md` (P1)。参照: https://github.com/Reasonia-TK/boltzpmp
(ユーザー自作の Boltzmann ソルバー。PyPI `boltzpmp` 0.5.0 が `backend/.venv` に導入済み。
Python 層のソースは `backend/.venv/Lib/site-packages/boltzpmp/*.py` で読める。パーサー本体は
Rust (`rust/boltzpmp-core/src/lxcat.rs`) で、`bp.parse_lxcat(text_or_path)` として呼べる)。

## 目的

LXCat 形式の断面積ファイルを **boltzpmp のパーサーの上位互換**で読み、ES-Sim の全ソルバー
(v1 MCC・流体・Boltzmann、将来の GPU MCC) が共通に使えるリッチな断面積モデルを作る。
現行 `es_sim/lxcat.py` (v1) は最小限 (パラメータ行の先頭トークンのみ・ATTACHMENT 破棄・
複数ガスを 1 種に合算・EFFECTIVE を無変換で elastic 扱い・COLUMNS 単位無視) なので置き換える。

## 作るもの

```
backend/es_sim/xs/
├── __init__.py   公開 API の再エクスポート
├── model.py      CrossSection / LxcatDatabase / LxcatDocument / GasComponent / GasMixture
├── lxcat.py      parse_lxcat_document(text) / load_lxcat(path)
├── convert.py    EFFECTIVE→ELASTIC 変換・電子断面積セットの整理・v1 XsProcess 変換・boltzpmp 変換
└── masses.py     代表的なガスの分子量テーブル (amu)
backend/tests/test_xs_lxcat.py, test_xs_convert.py
backend/tests/data/synthetic_lxcat_full.txt   (新規合成フィクスチャ。実データは入れない)
```

既存ファイルの変更は次の 3 つだけ:

1. `es_sim/lxcat.py`: 関数 `parse_lxcat(text, species)` の**シグネチャと戻り値型を保ったまま**、
   中身を新パーサー + `to_v1_processes` のラッパーに置き換える (モジュール docstring も更新)。
2. `es_sim/server.py`: `POST /v2/xs/parse` を追加 (既存 `/lxcat/parse` は 1. により自動で新パーサー経由になる)。
3. `tests/test_mcc.py`: 1. による**意図的な挙動変更** (下記) に合わせて期待値を更新。変更理由をコメントに書く。

`schema.py`・v1 の MCC/流体/boltz.py・フロントエンドは変更しない。

## 1. データモデル (`model.py`)

`@dataclass` で定義 (pydantic ではない。JSON 化は `to_dict()` を持たせる)。

```python
Kind = Literal["ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION",
               "ISOTROPIC", "BACKSCAT"]   # 最後の 2 つはイオン-中性 (Phelps 形式)

@dataclass
class CrossSection:
    kind: Kind
    projectile: str          # "e" (電子) またはイオン名 ("Ar^+" 等。SPECIES 行の "/" の左)
    target: str              # 標的ガス名 (2 行目の矢印の左、または SPECIES 行の "/" の右)
    product: str | None      # 2 行目の矢印の右 ("Ar*(11.55eV)" 等)。無ければ None
    reversible: bool         # 2 行目が "<->" (逆過程 = 超弾性の対象)
    threshold_ev: float      # 閾値 [eV] (ELASTIC/EFFECTIVE/ISOTROPIC/BACKSCAT は 0)
    mass_ratio: float | None # m/M (ELASTIC/EFFECTIVE のみ。0 < m/M < 1)
    weight_ratio: float | None   # g_upper/g_lower (EXCITATION の 3 行目 2 番目、ROTATION)
    lower_state: tuple[float, float] | None  # ROTATION の (E_low[eV], g_low)
    upper_state: tuple[float, float] | None  # ROTATION の (E_up[eV], g_up)
    energy_ev: np.ndarray    # float64、非減少 (重複は段差として許容)
    sigma_m2: np.ndarray     # float64、≥0、SI (m^2) に換算済み
    sigma_mt_m2: np.ndarray | None  # 3 列目 (運動量移行断面積)。無ければ None
    process: str             # PROCESS: 行の値 ("" 可)
    species: str             # SPECIES: 行の値 ("" 可)
    param: dict[str, float | str | bool]  # PARAM.: を解釈した辞書
    comment: str             # COMMENT: 行 (複数行は改行で連結)
    updated: str | None
    columns: str | None      # COLUMNS: 行の生値
    database: str | None     # 所属 DATABASE 名
    line: int                # キーワード行 (タイプ行なし形式は SPECIES 行) の 1 始まり行番号
    meta: dict[str, str]     # 上記以外の "KEY: value" 行
```

メソッド/プロパティ:
- `label` — `process` が空でなければそれ、無ければ `f"{kind} {target}" + (f" -> {product}" if product else "")`。
- `sigma(eps, *, below="zero", above="clamp")` — 線形補間。`below="zero"`: 表の最初の点より下と
  閾値未満 (inelastic) は 0 (boltzpmp と同じ)。`below="clamp"`: np.interp と同じ (v1 MCC 互換)。
  `above="clamp"`: 最後の値で一定 (boltzpmp・np.interp と同じ)、`above="zero"`: 0。
- `to_dict()` / `from_dict()` (配列はリスト、tuple はリスト)。

`LxcatDatabase(name: str | None, meta: dict[str, str], cross_sections: list[CrossSection])`、
`LxcatDocument(databases: list[LxcatDatabase], warnings: list[str])` (+ `all_cross_sections()`、
`targets(projectile="e")` = 出現順・重複なしの標的名リスト、`to_dict()`)。

`GasComponent(name, fraction, mass_amu: float | None, cross_sections: list[CrossSection])`、
`GasMixture(components, pressure_pa, temperature_k)` (+ `number_density` = p/(k_B T)、
`validate()` = 分率の和が 1±1e-6・各成分に電子の運動量移行断面積があること)。

## 2. パーサー (`lxcat.py`)

`parse_lxcat_document(text: str) -> LxcatDocument`、`load_lxcat(path) -> LxcatDocument`
(UTF-8 (BOM 除去) → 失敗時 Latin-1 にフォールバック、CRLF 可)。

**boltzpmp 0.5.0 の lxcat.rs と同じ解釈** (下記) を必ず満たし、そのうえで拡張する。

### ブロック構造 (標準形式)

- キーワード行: 前後空白を除いて**完全一致・大文字**の `ELASTIC` `EFFECTIVE` `EXCITATION`
  `IONIZATION` `ATTACHMENT` `ROTATION`。拡張: `MOMENTUM` を `EFFECTIVE` の別名として受理
  (旧 BOLSIG 形式。param["keyword_alias"]="MOMENTUM" を残し警告なし)。
- 2 行目: 反応式。`A`、`A -> B`、`A <-> B` (`<->` なら reversible=True)。矢印の前後で
  target/product に分け strip。
- パラメータ行 (2 行目の直後):
  - ELASTIC/EFFECTIVE: 3 行目の先頭トークンが数値なら m/M (**任意**。数値でなければ行を
    消費しない)。0 < m/M < 1 以外は ValueError。
  - EXCITATION: 3 行目 = 閾値 (必須) と任意の g_up/g_low。`!` または `#` 以降は無視。
    閾値 < 0 は ValueError。
  - IONIZATION: 3 行目 = 閾値 (必須)。余分なトークンは param["extra_tokens"] に文字列で残す。
  - ATTACHMENT: パラメータ行なし (LXCat 仕様)。拡張: 3 行目が「コロンを含まず数値トークン
    のみ」の行なら閾値として受理 (param["threshold_line"]=True)。
  - ROTATION: 3・4 行目 = 下準位 "E g"、上準位 "E g"。threshold = E_up − E_low (> 0 で
    なければ ValueError)、weight_ratio = g_up/g_low、lower_state/upper_state に保存。
  - 拡張: 必須の閾値行が無い (数値でない) が PARAM.: に `E = x eV` があればそれを使い警告。
    どちらも無ければ ValueError (行番号付き)。
- テーブル前の "KEY: value" 行 (キーは大文字小文字を区別しない、最初の ":" で分割):
  `SPECIES` `PROCESS` `PARAM.` `COMMENT` (複数可→改行連結) `UPDATED` `COLUMNS`、
  それ以外は `meta`。キーを持たない自由記述行は comment に追記。
- `PARAM.:` の解釈: "," 区切りの各要素を `key = value [unit]` として数値化 (単位 eV 等は
  除去、数値化できなければ文字列)、`=` の無い要素 (例 "complete set") は True。
  パラメータ行の値と PARAM の値 (m/M・E) が相対 1e-6 を超えて食い違えば警告
  (パラメータ行を優先)。ELASTIC/EFFECTIVE でパラメータ行が無く PARAM に m/M があれば採用。
- `COLUMNS:` の単位換算: エネルギー `eV`/`meV`/`keV`、断面積 `m2`/`m^2`/`cm2`/`cm^2`/
  `1e-20 m2`/`10^-20 m2`/`Å2`/`A2`/`1e-16 cm2`。SI (eV, m²) に換算して格納。
  解釈できない単位は警告して eV・m² とみなす。COLUMNS 行が無ければ eV・m²。
- テーブル: 前後を「`-` のみで 5 文字以上」の行で挟む。空行は読み飛ばす。各行は 2 列以上の
  数値で、全行同じ列数。3 列目があれば sigma_mt (運動量移行)。4 列以上は警告して 3 列目まで。
  エネルギーは非減少 (重複可)、全値有限・≥0。違反は ValueError (行番号付き)。
  終了区切りが無い・表が空も ValueError。

### タイプ行なし形式 (Phelps イオンデータ)

キーワードブロックの外で `SPECIES:` 行が現れたらタイプ行なしブロックの開始。テーブルまでの
"KEY: value" 行を集め、`PROCESS:` の最後の語 (`,` を空白扱い、末尾 `.` 除去、小文字化) で
種別判定: `isotropic`→ISOTROPIC、`backscat`/`backscattering`→BACKSCAT。未知なら警告して
スキップ (テーブルは読み飛ばす)。SPECIES 値 "Ar^+ / Ar" → projectile="Ar^+"、target="Ar"。
PROCESS が無ければ ValueError。

### ファイル全体

- `DATABASE:` 行でデータベースの切替 (以降のブロックの所属)。`PERMLINK:` `DESCRIPTION:`
  `CONTACT:` `HOW TO REFERENCE:` 等、ブロック外の "KEY: value" はその DB の meta に入れる。
  DATABASE 行が 1 つも無ければ name=None の DB 1 つ。
- `xxxx…`・`****…`・`******** Ar ********` のような区切り/見出し行、説明文はスキップ。
- SPECIES 行から projectile を決める ("e / Ar" → "e")。SPECIES 行が無い標準ブロックは
  projectile="e"。SPECIES の target が 2 行目の target と違えば警告 (2 行目を優先)。
- ブロックが 1 つも無ければ ValueError (v1 と同じ)。
- 回復可能な問題は `LxcatDocument.warnings` に日本語で積む (行番号を含める)。

## 3. 変換 (`convert.py`)

- `effective_to_elastic(effective, inelastic) -> tuple[CrossSection, list[str]]`:
  LXCat/BOLSIG+ の定義 (EFFECTIVE = 弾性運動量移行 + 全 inelastic の積分断面積) に従い、
  σ_el(ε) = max(σ_eff(ε) − Σ σ_inel(ε), 0)。評価グリッドは effective と inelastic 全表の
  エネルギーの和集合 (effective の範囲内、閾値の直前直後で段差が表現されるよう閾値自体も
  含める)。σ_inel は `sigma(below="zero", above="clamp")`。0 でクリップした点が σ_eff の
  1% を超える量なら警告。kind="ELASTIC"、mass_ratio は effective のもの、
  param["converted_from"]="EFFECTIVE"。inelastic = 同じ projectile・target の
  EXCITATION/IONIZATION/ATTACHMENT/ROTATION。
- `resolve_momentum_transfer(cross_sections) -> tuple[list[CrossSection], list[str]]`:
  電子の標的ごとに、ELASTIC があれば EFFECTIVE を除外 (警告)、EFFECTIVE だけなら変換
  (警告で変換した旨を明記)、どちらも無ければ警告。順序は元の出現順を保つ。
- `build_mixture(doc, fractions: dict[str, float], pressure_pa, temperature_k,
  masses: dict[str, float] | None = None) -> tuple[GasMixture, list[str]]`:
  電子断面積を標的ごとに GasComponent にまとめる (resolve_momentum_transfer 適用後)。
  質量は masses 引数 → m/M から M=m_e/(m/M) [amu] → masses.py の表、の順。
- `to_v1_processes(doc, species: "electron"|"ion") -> tuple[list[XsProcess], list[str]]`
  (v1 の `XsProcess` を返す。**v1 MCC が扱える種別だけ**):
  - electron: resolve_momentum_transfer 後、ELASTIC→"elastic" (mass_ratio が None なら
    質量表から、それも無ければ 0.0 で警告)、EXCITATION→"excitation"、ROTATION→"excitation"
    (警告: 超弾性は v1 MCC 非対応)、IONIZATION→"ionization"、ATTACHMENT→スキップ
    (警告文に "ATTACHMENT" を含める)。電子の標的が 2 種以上なら「v1 は単一ガスとして
    合算する」旨を警告。label は `label` プロパティ。
  - ion: ISOTROPIC→"isotropic"、BACKSCAT→"backscat" (threshold 0・mass_ratio 0)。
    電子の種別や projectile="e" のブロックは species フィルタとしてスキップ警告
    (v1 と同じく 1 ブロックにつき 1 警告)。
  - 出現順を保つ。energy/sigma はリストで (sigma_mt は v1 に無いので捨てる)。
- `to_boltzpmp_mixture(mixture) -> bp.Mixture` (boltzpmp を関数内で import):
  各成分を `bp.Gas(name, fraction, cross_sections=[bp.CrossSection(...)], mass_amu)` に。
  CrossSection の引数 (kind・species・name・threshold・mass_ratio・data・weight_ratio・
  lower_state・upper_state・mt_data・comment) の意味は **boltzpmp の Python ソース
  (`crosssections.py`) を読んで正確に**合わせる (target/product/reversible を boltzpmp が
  どこから得るかも確認)。boltzpmp は `<->` の生成物が混合ガスに無いとエラーにするので、
  その場合は reversible を落として (二準位系の扱い) 警告する。
- `to_boltzpmp_cross_sections(...)` 等の小関数は自由に切ってよい。

## 4. v1 互換ラッパー (`es_sim/lxcat.py`) と意図的な挙動変更

`parse_lxcat(text, species)` = `to_v1_processes(parse_lxcat_document(text), species)`。
構造エラーは従来どおり ValueError (server が 422 に変換)。

**意図的な挙動変更 (物理の修正)**: 合成フィクスチャ `tests/data/synthetic_electron.txt` は
Ar に ELASTIC と EFFECTIVE の両方を持つ。v1 は両方を elastic として取り込み運動量移行を
二重計上していた。新実装は ELASTIC を採用し EFFECTIVE を除外する → species="electron" の
結果は `["elastic", "excitation", "ionization"]`、警告は EFFECTIVE 除外 1 件 + ATTACHMENT
スキップ 1 件 = 2 件 (それぞれ "EFFECTIVE"/"ATTACHMENT" を含む)。`test_mcc.py` の
`test_parse_synthetic_electron` と `test_lxcat_endpoint` をこれに合わせて更新する。
他の既存テスト (species フィルタ・無効テキスト・イオン・実ファイル (無ければ skip)・
合成断面積で MCC を回すテスト) は**変更せずに通る**こと。

## 5. API (`server.py`)

`POST /v2/xs/parse` — リクエスト `{"text": str}`、レスポンス:
```jsonc
{
  "databases": [ { "name": "...", "meta": {...}, "cross_sections": [CrossSection.to_dict(), ...] } ],
  "targets": [ { "name": "Ar", "projectile": "e", "counts": {"ELASTIC": 1, ...},
                 "momentum_transfer": "elastic" | "effective" | "none",
                 "mass_amu": 39.948 | null } ],
  "warnings": [...]
}
```
パース失敗は 422 (既存 `/lxcat/parse` と同じ流儀)。pydantic のレスポンスモデルは
`es_sim/xs/` 側に置いてよい (schema.py は触らない)。

## 6. テスト

新規合成フィクスチャ `synthetic_lxcat_full.txt` (実データではない旨をヘッダに明記) に最低限
次を含める: DATABASE 2 つ・標的 2 種 (Ar と N2 など)・全キーワード (ELASTIC/EFFECTIVE/
EXCITATION/IONIZATION/ATTACHMENT/ROTATION/MOMENTUM)・`<->` + 統計重み比・3 列テーブル・
PARAM.: のみで閾値を与えるブロック・COLUMNS が cm² のブロック・段差 (エネルギー重複)・
`!` コメント付きパラメータ行・タイプ行なしイオンブロック。CRLF と BOM 付きの変種は
テスト内でテキストを加工して作る。

テスト項目:
1. 全ブロック種の各フィールド (kind/projectile/target/product/reversible/閾値/m/M/重み比/
   準位/単位換算後の値/sigma_mt/database/line) の検証。
2. 構造エラー (区切りなし・表が空・列数不一致・エネルギー減少・負の σ・m/M 範囲外・
   閾値欠落・ROTATION の逆転) がそれぞれ ValueError になり、メッセージに行番号を含む。
3. **boltzpmp との一致**: 電子の標準ブロックについて `bp.parse_lxcat` の結果と
   kind・閾値・m/M・重み比・(E, σ) 表・mt_data が一致する (boltzpmp が扱わない拡張部分
   (MOMENTUM・PARAM 由来閾値・cm² 換算・タイプ行なし) は比較から除外するか、boltzpmp が
   読める形の別テキストで比較)。
4. EFFECTIVE→ELASTIC 変換: 解析的に作った σ_eff・σ_exc (閾値付き) で差分が正しいこと、
   クリップ時の警告、閾値前後の段差。
5. `sigma()` の below/above 各モード。
6. `build_mixture` (分率・質量の決定順・検証エラー)、`to_boltzpmp_mixture` で得た
   Mixture を `bp.PMSolver(mix, eps_max_eV=60, d_eps_eV=0.2, n_theta=16).solve_dc(100)` が
   収束すること (Ar 単体と Ar/N2 混合の 2 例)。
7. v1 互換: 既存 `test_mcc.py` の全テストが (上記 2 件の期待値更新のみで) 通る。
   `/lxcat/parse` と `/v2/xs/parse` のエンドポイントテスト (TestClient)。
8. `to_dict()`/`from_dict()` の往復。

## 実行環境 (この PC)

- venv: `backend/.venv` (uv 管理)。テスト: `cd backend && .venv/Scripts/python.exe -m pytest tests/test_xs_lxcat.py -q -p no:cacheprovider`。
- 依存の追加は不要 (numpy・boltzpmp は導入済み)。**pyproject.toml / uv.lock は変更しない**。
- Windows の Smart App Control が有効。新しいバイナリ依存は入れないこと。
- 並行して別の作業者が `es_sim/device/` `es_sim/kernels/` `es_sim/geom/` `es_sim/eb/`
  `es_sim/field/` と `tests/test_v2_*.py` を編集している。これらには触れないこと。
- git の commit/push はしない (変更は作業ツリーに残す)。

## 完了条件

- 上記テストが全て通り、`pytest tests/ -q` の全体 (既存 380 件超) も通る (boltz を含む)。
- コードは既存モジュールの流儀 (日本語 docstring・コメント、型ヒント、`from __future__ import annotations`) に合わせる。
- 最後に、実装した内容・boltzpmp との差異 (上位互換の拡張点)・意図的な挙動変更・既知の
  制約を簡潔に報告する。
