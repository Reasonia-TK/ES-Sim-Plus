# 95: 1D PIC/MCC に FN 電界放出 (Fowler–Nordheim) を追加

## 背景 (ユーザー要望)

「FN emissionも追加して」
2D PIC には FN 電界放出 (prompts/46、fn.py: Murphy-Good 式 + Forbes 近似の
`fn_current_density`) がある。1D PIC/MCC の電極 (左/右) にも同じ物理で
FN 放出を追加する。µm ギャップ + 高電界の FN 検証は 1D が最適な用途。

## backend

### schema.py

```py
class Fn1dEmission(BaseModel):
    """1D 電極の FN 電界放出 (prompts/95)。fn.py の fn_current_density を流用。"""
    phi_ev: float = Field(4.5, gt=0)      # 仕事関数 [eV]
    beta: float = Field(1.0, gt=0)        # 電界増倍係数
    init_energy_ev: float = Field(0.1, ge=0)
    # マクロ重み [実電子数/m^2 / マクロ粒子]。None なら初期プラズマの w0 を使う
    macro_weight: float | None = Field(None, gt=0)

class Pic1dElectrode(BaseModel):
    ...
    fn: Fn1dEmission | None = None  # None = 放出なし (従来とビット不変)
```

### pic1d.py — 毎ステップの放出

- 表面電界: 電極節点の E (節点配列の端値)。放出方向 n̂ は左電極 +x、右電極 −x。
  FN に渡す幾何電界 F は「電子を真空側へ引き出す向き」のみ:
  **F = −E·n̂ が正のときその値、そうでなければ 0** (2D の fn.py docstring と
  同じ規約。左: F = −E[0] (E[0]<0 のとき正)、右: F = +E[-1])。
- J = `fn_current_density(F, phi_ev, beta)` [A/m²] (fn.py から import。1D の
  マクロ重みは [m^-2] 単位なので面積換算不要 — J·dt/e がそのまま実電子数/m²)。
- 放出数: N_real = J·dt/e、n_macro = N_real / w_fn (w_fn = macro_weight または w0)。
  **小数部は電極ごとの累積キャリー** (決定論的。乱数不使用) — 2D の PIC 側 FN
  実装 (pic.py) の端数処理を読んで同じ方式にする。
- 生成粒子: 位置は壁からわずかに内側 (2D の delta 規約を読んで踏襲。例: 1e-6·gap
  や dx の小さな係数 — 2D と整合する方が良い)、速度は init_energy_ev を放出方向
  n̂ に与える (2D の初速規約を読んで一致させる。等方半球なら半球で)。
  リープフロッグの半ステップ規約 (新粒子の v の扱い) は既存の SEE/電離粒子の
  流儀に合わせる。
- 続きから実行: キャリーも状態に含めてビット一致を保つ。

### 診断

- history に `fn_left` / `fn_right` (そのステップの放出重み [m^-2]) を追加。
- done result に `fn: {left: {j_avg, total_w}, right: {...}} | null` を追加
  (j_avg = 平均区間中の平均放出電流密度 [A/m²]、total_w = 累計放出重み。
  fn 未設定なら null)。frame の counts は history 最終行なので自然に載る。

### テスト (tests/test_pic1d.py に追加)

1. 放出数の整合: MCC なし・強い DC 逆バイアス (例: gap 1 µm、V_L=0, V_R=−100 V
   → F ≈ 1e8 V/m、beta=50 等で J が有意に出る条件) で、総放出重み ≈
   Σ J(F_t)·dt/e (rtol 1e-6。キャリー誤差はマクロ1個未満)。
2. 向きの検査: 引き出し電界が逆向きの電極からは放出ゼロ。
3. fn=None で従来とビット不変 (既存テスト全パス + 明示は不要)。
4. 続きから: fn ありで run(n)+continue(m) == run(n+m) ビット一致 (キャリー含む)。
5. validator: phi_ev<=0 等で ValidationError。

`python -m pytest tests/ -q` 全件パス (現在 243 + 新規)。

## frontend

### types.ts

- `Fn1dEmission`、`Pic1dElectrode.fn?: Fn1dEmission | null`。

### Pic1dPanel — 電極 (左/右) セクション

- 「FN電界放出」Toggle (RF重畳の下)。オンで phi_ev (仕事関数 [eV])、β (電界増倍)、
  初期エネルギー [eV]、マクロ重み (CommitNullableNumberInput、null=自動) を表示。
  2D の FN 設定 UI (FnPanel / ParticlePanel の FN 部分) の文言・hint を参照して
  合わせる。hint: 「表面電界が電子を引き出す向きのときのみ放出します
  (Murphy-Good FN 式)」。
- トグルオフ→オンで直前値を復元 (MCC 設定と同じ ref 保持の流儀)。

### Plot1dView — 結果サマリ

- fn 結果があれば数値サマリに「FN放出: 左 J_avg / 右 J_avg [A/m²]」行を追加
  (formatNumber)。
- history チャートに fn 放出を足す必要はない (サマリのみで可)。

### 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- fn.py の `fn_current_density` は**変更せず import** して使う。
- 2D の pic.py FN 実装 (端数キャリー・delta オフセット・初速・診断) を必ず読んで
  規約を一致させること。
- fn 未指定の既存プロジェクト・保存ファイルは挙動不変 (ビット一致)。
