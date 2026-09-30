# ES-Sim バックエンドの配布ビルド (Windows 用、prompts/44)。
#
# venv を有効化した PowerShell で実行する:
#   powershell -ExecutionPolicy Bypass -File scripts\build_backend.ps1
#
# PyInstaller で backend\dist\es-sim-backend.exe を生成し、Tauri の externalBin が
# 要求する target triple 付きファイル名 (es-sim-backend-x86_64-pc-windows-msvc.exe)
# で ui\src-tauri\binaries\ へ配置する (P6f で v1 の frontend\ から移した)。
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $Root "ui\src-tauri\binaries"

# ---- target triple の決定 (rustc があればホスト triple、無ければ MSVC 既定) ----
$Triple = "x86_64-pc-windows-msvc"
if (Get-Command rustc -ErrorAction SilentlyContinue) {
    $HostLine = (rustc -vV | Select-String "^host: ").Line
    if ($HostLine) { $Triple = $HostLine -replace "^host: ", "" }
}

Write-Host "== PyInstaller ビルド (target: $Triple) =="
Push-Location (Join-Path $Root "backend")
try {
    pyinstaller --clean --noconfirm es_sim_server.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller のビルドに失敗しました" }

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Copy-Item -Force "dist\es-sim-backend.exe" (Join-Path $BinDir "es-sim-backend-$Triple.exe")
    Write-Host "== 配置完了: $BinDir\es-sim-backend-$Triple.exe =="
}
finally {
    Pop-Location
}
