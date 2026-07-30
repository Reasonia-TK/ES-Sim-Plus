# 94: 1D プリセット取得の「Failed to fetch」修正 (自動再試行)

## 背景 (ユーザー報告)

「プリセット取得に失敗しました: TypeError: Failed to fetch というエラーが出ている」

原因: Pic1dPanel はパネル初回マウント時に `GET /pic1d/presets` を**1回だけ**呼ぶ
(useEffect [], 再試行なし)。アプリ起動直後はバックエンドのサイドカー起動が
パネル表示より遅れることがあり、その1回が TypeError: Failed to fetch で失敗すると
エラーが表示されたまま二度と取得されない。
また旧バックエンド (v0.1.33 未満) には endpoint が無く 404 になるが、
api.pic1dPresets は res.ok を見ずに r.json() しており、エラー系 JSON を
presets として扱ってしまう。

## 修正

### frontend/src/api.ts — pic1dPresets

- `res.ok` チェックを追加 (sweepResult と同じ流儀)。
- 404 のときは「バックエンドが古い可能性があります (上部バーの backend
  バージョンを確認してください)」という日本語メッセージで throw
  (旧サイドカー残留の既知問題の切り分け用)。

### frontend/src/panels/Pic1dPanel.tsx — プリセット取得

- 失敗したら **3秒間隔で成功するまで自動再試行** (useEffect 内で setTimeout、
  クリーンアップで cancelled フラグ + clearTimeout。presets 取得成功後は再試行しない)。
- 再試行中のエラー表示は文言を「プリセット取得に失敗しました (自動再試行中):
  {エラー}」に変更 (`.error` のままで可)。

## 検証

`cd /home/claude/ES-Sim/frontend && npx tsc --noEmit && npx vite build`。

## 注意

- コメントは日本語で「なぜ」(バックエンド起動レースの説明を書く)。
- backend は変更しない。git commit はしない。
