# 117: boltzpm 連携 (backend) — Boltzmann ソルバーによる流体係数生成 (LMEA)

## 背景 (ユーザー要望)

「EEDF、電子移動度、速度定数などを boltzpm (https://github.com/Syb04/boltzpm、
ユーザー自作の伝播演算子法 Boltzmann ソルバー、純 numpy/scipy) を使って計算して
流体モデルに適用できるようにしたい」

現在の流体係数は Maxwell 平均 (fluid_coeffs.py)。boltzpm は非 Maxwell な EEDF を
解くので、**BOLSIG+ 流の局所平均エネルギー近似 (LMEA)** で置き換え可能にする:
E/N を掃引して定常解を求め、**平均エネルギー ε̄ をキー**に μ_e·N・レート係数を
テーブル化 → 流体は各点の T_e から ε̄ = (3/2)T_e でテーブル参照する。

boltzpm はローカルに clone 済み: /home/claude/boltzpm (API は README 参照。
`bp.Gas / bp.Mixture / bp.PMSolver(mixture, eps_max_eV, d_eps_eV, n_theta)`、
`solver.solve_dc(EN_Td)` → SwarmResult {mean_energy, drift_velocity,
reduced_ionization_frequency, rate_coefficients (dict "gas:process" → k),
eedf 系属性は output.py を読んで確認}。まず README/SPEC/ソースを読むこと)。

## 依存の追加

- サンドボックス: `pip install /home/claude/boltzpm --break-system-packages`。
- `backend/pyproject.toml` dependencies に
  `"boltzpm @ git+https://github.com/Syb04/boltzpm.git"` を追加 (リリース CI の
  `pip install -e .` で取得される)。
- `backend/es_sim_server.spec` (PyInstaller): boltzpm のパッケージデータ
  (`boltzpm/data/*.txt`) が同梱されるよう `collect_data_files("boltzpm")` 等を
  追加 (spec の既存構造を読んで流儀に合わせる)。
- boltzpm の import はモジュールレベルで try/except し、無い環境では
  /ws/boltz がエラーメッセージ (「boltzpm がインストールされていません」) を
  返す (既存テストが boltzpm 無し環境でも落ちないように — ただし CI/サンドボックス
  にはインストールするのでテストはフルに走る)。

## backend/es_sim/boltz.py (新規)

### 変換

- `xsprocess_to_mixture(processes: list[XsProcess], mass_amu: float,
  p_pa: float, t_k: float) -> bp.Mixture`
  — kind の対応 (elastic/excitation/ionization ← 既存 XsProcess)、
  threshold_ev、energy_ev/sigma_m2 列。boltzpm の CrossSection/Gas/Mixture の
  コンストラクタをソースで確認して正確に。attachment は未使用 (存在すれば
  ValueError で明示)。

### E/N 掃引 → テーブル

- `run_boltz_sweep(processes, mass_amu, p_pa, t_k, opts, progress_cb=None,
  should_stop=None) -> dict` (= BoltzTable の dict):
  - `opts`: en_min_td=0.5, en_max_td=1000, n_points=32 (geomspace)、
    eps_max_ev=None (None なら最大閾値の8倍と 40 eV の大きい方)、
    d_eps_ev=0.25、n_theta=16。
  - 各 E/N で solve_dc → 記録: en_td、mean_energy_ev ⟨ε⟩、
    mobility_n = W / (EN_Td·1e-21) [1/(m·V·s)] (μN = W/E ÷ N の同値式。
    単位検算コメント必須)、プロセス別 k [m^3/s] (kind 別に電離和 k_ion・励起和
    k_exc へ集約 + 閾値加重 e_ion_ev/e_exc_ev を**点ごと**に)、EEDF
    (eps_ev グリッド + 規格化 f。全点保存 — 表示用)。
  - 収束フラグ false の点は warnings に記録して**除外** (テーブルに入れない)。
  - ε̄ 昇順に整列し、単調でない場合は警告 (LMEA の前提が崩れるため)。
  - meta: 使用 opts、processes のハッシュ (sha256 of JSON — フロントの
    「断面積変更後は再生成」判定用)、boltzpm バージョン。

### schema.py

```py
class BoltzTable(BaseModel):
    """boltzpm による LMEA 係数テーブル (prompts/117)。ε̄ = (3/2)Te をキーに参照。"""
    en_td: list[float]; mean_energy_ev: list[float]
    mobility_n: list[float]          # μ_e·N [1/(m·V·s)]
    k_ion: list[float]; k_exc: list[float]
    e_ion_ev: list[float]; e_exc_ev: list[float]
    eedf_eps_ev: list[float]; eedf: list[list[float]]
    source_hash: str; opts: dict; warnings: list[str] = []
# Fluid1dSettings / Fluid2dSettings:
    electron_model: Literal["maxwell", "boltzmann"] = "maxwell"  # 既定は従来
    boltz_table: BoltzTable | None = None
# validator: electron_model="boltzmann" で boltz_table が None なら ValueError
```

### fluid1d.py / fluid2d.py への組み込み

- electron_model=="boltzmann" のとき、Te → ε̄=(3/2)Te でテーブル参照
  (log-log 補間、fluid_coeffs.interp_loglog 流用。範囲外クランプ + 初回警告):
  - μ_e = mobility_n(ε̄)/n_g、D_e = μ_e·(2/3)ε̄ (**一般化 Einstein**。boltzpm は
    拡散係数を出さないため。近似である旨コメント)
  - k_ion/k_exc と損失エネルギー e_ion_ev(ε̄)/e_exc_ev(ε̄)
  - 弾性エネルギー損失: ν_m_eff = e/(m_e μ_e) から 3(m/M)ν_m_eff(Te−Tg)
    (二項近似の実効運動量移行周波数と整合する扱い。理由コメント)
- "maxwell" (既定) は従来経路と**ビット不変**。

### server.py — /ws/boltz

- {cmd:"start", processes | project+module ("fluid1d"/"fluid2d" — 設定から
  processes/質量/ガス条件を取る)、opts (省略可)} →
  progress {i, n_points, en_td, elapsed_s} / done {table} / error {detail}。
  cancel 対応、anyio オフロード (既存 WS の流儀)。テーブルはフロントが受け取り
  settings に格納する (サーバー側では保持しない)。

## テスト (tests/test_boltz.py — 新規)

1. 変換: eduPIC Ar 解析式 XsProcess → Mixture が例外なく構築、プロセス数一致。
2. 小規模掃引 (粗メッシュ: d_eps 0.5・n_theta 8・n_points 5、en 1〜300 Td) で:
   - ε̄ が E/N とともに単調増加
   - mobility_n のオーダー 1e23〜1e25、E/N とともに減少傾向
   - k_ion が ε̄ とともに増加、低 ε̄ で ≪ 高 ε̄
   - EEDF が非負・規格化 (∫f√ε dε ≈ 1 など boltzpm の規格を確認して検証)
3. Maxwell との比較 (物理サニティ): 同じ断面積で ε̄ 3 eV 相当の k_ion が
   Maxwell 平均値とオーダー一致 (factor 10 以内。非 Maxwell 分布で下がる方向の
   コメント)。
4. fluid1d: electron_model="boltzmann" + 生成テーブルで CCP スモークが収束・有限。
   "maxwell" 既定の既存テストがビット不変。validator。
5. CI 時間: 掃引テストは 1〜2 分以内に収める (粗パラメータ)。

`python -m pytest tests/ -q` 全件パス (現在 363 + 新規)。

## 注意

- コメントは日本語で「なぜ」(LMEA の考え方、μN の単位検算、D_e 一般化 Einstein、
  ν_m_eff、収束失敗点の除外)。
- git commit はしない。boltzpm 本体のコードは変更しない (バグを見つけたら
  レポートに記載のみ)。
- release.yml は変更不要のはず (pip install -e . が git 依存を解決する) だが、
  spec の datas 追加は必要性を確認して対応。
