#!/usr/bin/env bash
# ES-Sim バックエンドの配布ビルド (Linux / macOS の検証用、prompts/44、onedir は prompts/133 P8b)。
#
# PyInstaller の入った Python 環境で実行する (uv なら backend で uv sync --group dist):
#   bash scripts/build_backend.sh
#
# PyInstaller で backend/dist/es-sim-backend/ (onedir) を作り、自己テスト (selftest) を走らせる。
# CuPy と NVRTC の同梱は Windows だけ (es_sim_server.spec)。GPU の確認は使えるときだけ行う。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"

echo "== PyInstaller ビルド (onedir) =="
python -m PyInstaller --clean --noconfirm es_sim_server.spec

echo "== 自己テスト =="
./dist/es-sim-backend/es-sim-backend selftest
echo "== 完了: $ROOT/backend/dist/es-sim-backend =="
