# ES-Sim バックエンドの配布ビルド (Windows 用、prompts/44、onedir と GPU は prompts/133 P8b)。
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_backend.ps1 [-RequireGpu] [-SkipTest]
#
# 前提: backend の venv に gpu と依存グループ dist (PyInstaller・NVRTC の wheel) が入っていること
#   cd backend; uv sync --extra gpu --group dist      (開発用の dev も残すなら --extra dev も)
#
# PyInstaller で backend\dist\es-sim-backend\ (onedir: es-sim-backend.exe と _internal\) を作り、CUDA の環境変数を
# 消して PATH から CUDA Toolkit を除いた状態 (CUDA Toolkit の無い PC と同じ) で自己テスト (selftest) を走らせる。
# -RequireGpu は GPU を使えないことも失敗にする (GPU のある開発機での確認用。CI は GPU が無いので付けない)。
# 署名 (prompts/133: 当面は署名しない): 環境変数 ES_SIM_SIGN_SCRIPT に「引数のファイル 1 つに署名する」PowerShell の
# スクリプトを指定すると、署名の無い exe・dll・pyd に署名してから自己テストする (Smart App Control の PC で動かすには
# 同梱のもの全部に署名が要る)。失敗したらスクリプトが例外を投げるか、終了コード 0 以外を返すこと。
param(
    [switch]$RequireGpu,
    [switch]$SkipTest
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Python = Join-Path $Backend ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "backend\.venv がありません。backend で uv sync --extra gpu --group dist を実行してください" }

Push-Location $Backend
try {
    & $Python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller がありません。backend で uv sync --extra gpu --group dist を実行してください" }

    Write-Host "== PyInstaller ビルド (onedir) =="
    & $Python -m PyInstaller --clean --noconfirm es_sim_server.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller のビルドに失敗しました" }
    $Exe = Join-Path $Backend "dist\es-sim-backend\es-sim-backend.exe"

    if ($env:ES_SIM_SIGN_SCRIPT) {
        $sign = (Resolve-Path $env:ES_SIM_SIGN_SCRIPT).Path
        $files = @(Get-ChildItem -Recurse -File (Split-Path $Exe) -Include *.exe, *.dll, *.pyd |
                Where-Object { (Get-AuthenticodeSignature $_.FullName).Status -ne "Valid" })
        Write-Host "== 署名 ($($files.Count) ファイル、$sign) =="
        foreach ($f in $files) {
            $global:LASTEXITCODE = 0
            & $sign $f.FullName
            if ($LASTEXITCODE -ne 0) { throw "署名に失敗しました: $($f.FullName)" }
        }
    }

    if (-not $SkipTest) {
        Write-Host "== 自己テスト (CUDA の環境変数なし) =="
        $saved = @{}
        foreach ($name in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(CUDA_PATH|CUDA_HOME)' } | ForEach-Object { $_.Name })) {
            $saved[$name] = [Environment]::GetEnvironmentVariable($name)
            Remove-Item "Env:$name"
        }
        $savedPath = $env:PATH
        $env:PATH = (($env:PATH -split ';') | Where-Object {
                $_ -and -not ((Test-Path (Join-Path $_ 'nvrtc64_*.dll')) -or (Test-Path (Join-Path $_ 'cublas64_*.dll')) -or
                    (Test-Path (Join-Path $_ 'x64\nvrtc64_*.dll')))
            }) -join ';'
        try {
            $testArgs = @("selftest")
            if ($RequireGpu) { $testArgs += "--require-gpu" }
            & $Exe @testArgs
            if ($LASTEXITCODE -ne 0) { throw "自己テストに失敗しました" }
        }
        finally {
            $env:PATH = $savedPath
            foreach ($name in $saved.Keys) { Set-Item "Env:$name" $saved[$name] }
        }
    }
    $size = (Get-ChildItem -Recurse -File (Split-Path $Exe) | Measure-Object -Sum Length).Sum / 1MB
    Write-Host ("== 完了: {0} ({1:N0} MB) ==" -f (Split-Path $Exe), $size)
}
finally {
    Pop-Location
}
