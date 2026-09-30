#!/usr/bin/env bash
# ES-Sim バックエンドの配布ビルド (Linux / macOS 検証用、prompts/44)。
#
# venv (または pyinstaller が入った Python 環境) で実行する:
#   bash scripts/build_backend.sh
#
# PyInstaller で backend/dist/es-sim-backend を生成し、Tauri の externalBin が
# 要求する target triple 付きファイル名で ui/src-tauri/binaries/ へ配置する (P6f で v1 の frontend/ から移した)。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/ui/src-tauri/binaries"

# ---- target triple の決定 (rustc があればホスト triple、無ければ uname から推定) ----
if command -v rustc >/dev/null 2>&1; then
    TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
else
    case "$(uname -sm)" in
        "Linux x86_64")  TRIPLE="x86_64-unknown-linux-gnu" ;;
        "Linux aarch64") TRIPLE="aarch64-unknown-linux-gnu" ;;
        "Darwin arm64")  TRIPLE="aarch64-apple-darwin" ;;
        "Darwin x86_64") TRIPLE="x86_64-apple-darwin" ;;
        *) echo "未知のプラットフォームです。rustc を入れるか TRIPLE を手動指定してください" >&2; exit 1 ;;
    esac
fi

echo "== PyInstaller ビルド (target: $TRIPLE) =="
cd "$ROOT/backend"
pyinstaller --clean --noconfirm es_sim_server.spec

mkdir -p "$BIN_DIR"
cp dist/es-sim-backend "$BIN_DIR/es-sim-backend-$TRIPLE"
chmod +x "$BIN_DIR/es-sim-backend-$TRIPLE"
echo "== 配置完了: $BIN_DIR/es-sim-backend-$TRIPLE =="
