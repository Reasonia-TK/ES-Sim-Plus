# 118: boltzpm 連携 (frontend) — 係数生成 UI・スウォーム/EEDF チャート

## 背景

prompts/117 で backend 実装済み。フロントから boltzpm 係数生成を実行し、
生成テーブルを流体設定に格納・可視化する。

## backend 仕様 (実装済み、正確な形は schema.py / server.py を読んで写す)

- `/ws/boltz`: {cmd:"start", project, module:"fluid1d"|"fluid2d", opts?} →
  started {n_points} → progress {i, n_points, en_td, elapsed_s} →
  done {table: BoltzTable} / error {detail}。{cmd:"stop"} で中断。
- BoltzTable: en_td[], mean_energy_ev[], mobility_n[], k_ion[], k_exc[],
  e_ion_ev[], e_exc_ev[], eedf_eps_ev[], eedf[][], source_hash, opts, warnings[]。
- Fluid1dSettings/Fluid2dSettings: electron_model ("maxwell"|"boltzmann"、
  既定 maxwell)、boltz_table。
- opts 既定: en_min_td 0.5, en_max_td 1000, n_points 32, d_eps_ev 0.25,
  n_theta 16, eps_max_ev null (自動)。
- **注意: 既定パラメータの掃引は数分かかる** (低 E/N 点が支配的)。進捗表示と
  中断が必須。

## 作業内容 (frontend)

### types.ts / boltzClient.ts

- BoltzTable・BoltzOpts・WS メッセージ型、Fluid1d/2dSettings への
  electron_model/boltz_table 追加。
- boltzClient.ts (新規): /ws/boltz クライアント (既存クライアントの流儀)。

### Fluid1dPanel / Fluid2dPanel — 「電子係数 (boltzpm)」セクション

- 「電子係数モデル」select: 「Maxwell 平均 (既定)」/「Boltzmann (boltzpm)」。
  boltzmann 選択時にテーブル未生成なら実行ボタンを無効化しヒント表示
  (backend validator と整合)。
- 「boltzpm で係数生成」ボタン + 進捗 (i/n_points、現在の E/N、経過秒) +
  中断ボタン。実行中は anyRunning に統合 (他ソルバーと排他)。
  完了で settings.boltz_table へ commit (結果付き保存に自然に同梱される)。
  hint: 「E/N を掃引して定常 Boltzmann 解を求めます (既定設定で数分)」。
- 生成パラメータ (折りたたみ or 小さめの節): en_min/max_td、n_points、
  d_eps_ev、n_theta、eps_max_ev (null=自動)。CommitNumberInput 系。
- **鮮度警告**: electron_processes の JSON を sha256 して boltz_table.source_hash
  と比較 (Web Crypto の crypto.subtle.digest。backend のハッシュ対象 JSON の
  正確な形を boltz.py で確認して一致させる。非同期なので useEffect + state)。
  不一致なら「断面積が変更されています。係数を再生成してください」警告。
- テーブル情報表示: 点数、ε̄ 範囲、生成時 opts、warnings (収束失敗点の除外等)。
  「テーブルを削除」ボタン (boltz_table=null、electron_model を maxwell に戻す)。

### チャート (パネル内、既存チャート部品を流用)

1. **スウォームパラメータ**: ε̄ (横軸、log) vs μ_e·N と k_ion/k_exc (縦軸 log)。
   2軸が難しければチャート2枚 (μN / k)。
2. **EEDF ビューア**: E/N 選択 (select: テーブルの en_td 一覧) → その点の
   EEDF (ε vs f、log 縦軸トグル)。EEPF (f/√ε) 切替トグルもあると良い。
   CSV 書き出し (既存 EEDF チャートの流儀)。

### 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。
backend は変更しない (プロトコル齟齬があれば報告のみ)。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- 1D/2D パネルで UI を共通部品化する (BoltzSection コンポーネント等。二重管理を
  避ける)。
- Math.min/max スプレッド禁止 (arrayMin/arrayMax)、import type ブロックに
  割り込まない。対数軸は 0/負値を除外。
