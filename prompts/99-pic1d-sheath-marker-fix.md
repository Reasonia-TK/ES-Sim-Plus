# 99: 1D シースエッジマーカーの右電極側が左に描かれるバグ修正

## 背景 (ユーザー報告)

「1Dのシースエッジの位相分解アニメーションなんですが、lhsに2つ表示されていて
rhsには表示されてないです」

原因: backend (pic1d.py `brinkmann_sheath_edge`) の from_left=False は右電極からの
距離座標 d = gap − x へ鏡映して評価するため、**s_right は「右電極からの距離」**
として返る (シース厚として自然な量。s(φ) チャートやサマリにはこの方が適切)。
一方 frontend の Plot1dView は左右どちらのマーカーも値をそのまま x 座標として
描いているため、右側マーカーが x = s_right (左端近傍) に出て LHS に2本重なる。
時間平均ビュー (Pic1dResultView の sheathMarkers) にも同じバグがある。

## 修正 (frontend/src/canvas/Plot1dView.tsx のみ)

1. **位相アニメ (Pic1dCyclePlayer) のマーカー**: 右側は
   `x = xDisp[xDisp.length − 1] − mToUnit(sr, lengthUnit)` (表示単位の gap から引く)。
   左はそのまま。コメントに「s_right は右電極からの距離 (backend の鏡映座標) で
   あるため x 座標へ変換する」と理由を書く。
2. **時間平均ビュー (Pic1dResultView) のマーカー**: 同様に右側は gap − right_s
   (profiles.x の末尾を gap として使う)。
3. **ラベルの明確化**:
   - 数値サマリの行を「シースエッジ (電極からの距離): 左 … / 右 …」に変更
     (値自体は従来どおり距離 = シース厚で正しい)。
   - s(φ) チャート (Pic1dSheathPhaseChart) の軸/凡例に「電極からの距離」で
     ある旨が伝わる表記があるか確認し、無ければタイトルか凡例に補う。
4. backend は変更しない (規約: s = 各電極からの距離。brinkmann_sheath_edge の
   docstring に既記載の「電極からの距離」を、build_pic1d_result 側の
   sheath/cycle.sheath を組み立てる箇所のコメントでも明記 — コメント追記のみ可)。

## 検証

- `cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。
- backend に挙動変更なし (コメント追記のみなら pytest 再実行は全件でなくても可だが
  念のため `python -m pytest tests/test_pic1d.py -q`)。

## 注意

- コメントは日本語で「なぜ」。git commit はしない。
- 2D のライン Brinkmann (sheath.ts) は p1 からの距離を p1+s·方向 に描いており
  影響なし (確認のみ)。
