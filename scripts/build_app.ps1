# ES-Sim の配布ビルド一式 (Windows、prompts/133 P8c): バックエンド (onedir と自己テスト) → UI と Tauri → インストーラ。
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_app.ps1 [-RequireGpu] [-SkipBackend]
#
# 前提: backend で uv sync --extra gpu --group dist、ui で npm ci、Rust (stable, MSVC)。
# Tauri は src-tauri\tauri.bundle.json を重ねてビルドし、backend\dist\es-sim-backend\ をリソースの backend\ に同梱する
# (tauri.conf.json だけの tauri dev・cargo はバックエンドのビルドを要らない)。
# 生成物: ui\src-tauri\target\release\bundle\nsis\ES-Sim_<版>_x64-setup.exe と msi\ES-Sim_<版>_x64_en-US.msi。
# 確かめ方: scripts\verify_installer.ps1 (インストール → GPU の計算 → アンインストール)。
param(
    [switch]$RequireGpu,
    [switch]$SkipBackend
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not $SkipBackend) {
    & (Join-Path $PSScriptRoot "build_backend.ps1") -RequireGpu:$RequireGpu
}
$Backend = Join-Path $Root "backend\dist\es-sim-backend\es-sim-backend.exe"
if (-not (Test-Path $Backend)) { throw "バックエンドがありません ($Backend)。scripts\build_backend.ps1 を先に" }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH" }

Write-Host "== Tauri のビルド (バックエンドをリソースとして同梱) =="
Push-Location (Join-Path $Root "ui")
try {
    & npx tauri build --config src-tauri/tauri.bundle.json
    if ($LASTEXITCODE -ne 0) { throw "Tauri のビルドに失敗しました" }
}
finally {
    Pop-Location
}
Get-ChildItem (Join-Path $Root "ui\src-tauri\target\release\bundle") -Recurse -Include *.exe, *.msi |
    ForEach-Object { Write-Host ("{0} ({1:N0} MB)" -f $_.FullName, ($_.Length / 1MB)) }
