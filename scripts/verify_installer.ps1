# ES-Sim のインストーラの確認 (Windows、prompts/133 P8c)。
#
#   powershell -ExecutionPolicy Bypass -File scripts\verify_installer.ps1 [-Installer <setup.exe>] [-KeepInstalled] [-NoGpu]
#
# 1. 一時フォルダへ黙ってインストール (今のユーザーだけ、ショートカットは作らない: /S /NS /D=)
# 2. 同梱のバックエンドの自己テスト (selftest --require-gpu。-NoGpu なら GPU の確認は飛ばす)
# 3. CUDA の環境変数を消して PATH から CUDA Toolkit を除いた状態 (CUDA Toolkit の無い PC と同じ) でアプリを起動。
#    WebView2 のデータは一時フォルダ (普段のアプリのデータ・自動保存に触れない)、CDP のポートを開ける
# 4. ui/e2e-installed (Playwright を CDP でつなぐ): 同梱のバックエンドにつながり GPU を使えること、サンプルの
#    容量結合プラズマを直交格子 (PIC は GPU だけ) で走らせて結果まで。-NoGpu (GPU の無い PC・CI) は
#    CUDA_VISIBLE_DEVICES=-1 で GPU を隠し、CPU で動くこと・GPU を使えない理由の表示・静電場 (gmsh と FEM) を見る
# 5. アプリを閉じて、バックエンドのプロセスが残らないこと
# 6. 黙ってアンインストール (-KeepInstalled なら残す)
# アプリの設定フォルダ (AppConfig) の backend.log には追記される。-NoGpu でなければ GPU が要る (ドライバ R580 以降)。
param(
    [string]$Installer = "",
    [switch]$KeepInstalled,
    [switch]$NoGpu,
    [int]$CdpPort = 9233
)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not $Installer) {
    $Installer = Get-ChildItem (Join-Path $Root "ui\src-tauri\target\release\bundle\nsis\*-setup.exe") -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1 -ExpandProperty FullName
}
if (-not $Installer -or -not (Test-Path $Installer)) { throw "インストーラが見つかりません (scripts\build_app.ps1 で作る)" }
$Work = Join-Path ([IO.Path]::GetTempPath()) ("es-sim-verify-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
$Dir = Join-Path $Work "app"
$WebView = Join-Path $Work "webview"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

function Step([string]$msg) { Write-Host "== $msg ==" }

function Backend-Processes {
    @(Get-Process -Name "es-sim-backend" -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($Dir, [StringComparison]::OrdinalIgnoreCase) })
}

# ---- CUDA Toolkit の無い PC を模す (このスクリプトのプロセスと、ここから起動するものだけ) ----
foreach ($name in @(Get-ChildItem Env: | Where-Object { $_.Name -match '^(CUDA_PATH|CUDA_HOME)' } | ForEach-Object { $_.Name })) {
    Remove-Item "Env:$name"
}
$env:PATH = (($env:PATH -split ';') | Where-Object {
        $_ -and -not ((Test-Path (Join-Path $_ 'nvrtc64_*.dll')) -or (Test-Path (Join-Path $_ 'cublas64_*.dll')) -or
            (Test-Path (Join-Path $_ 'x64\nvrtc64_*.dll')))
    }) -join ';'

if ($NoGpu) {
    $env:CUDA_VISIBLE_DEVICES = "-1"   # GPU のある PC でも CPU の流れを確かめる (CI はもともと GPU が無い)
    $env:E2E_EXPECT_GPU = "0"
}

$app = $null
$ok = $false
try {
    $size = (Get-Item $Installer).Length / 1MB
    Step ("インストール: {0} ({1:N0} MB) → {2}" -f (Split-Path $Installer -Leaf), $size, $Dir)
    $t0 = Get-Date
    $p = Start-Process -FilePath $Installer -ArgumentList "/S", "/NS", "/D=$Dir" -PassThru -Wait
    if ($p.ExitCode -ne 0) { throw "インストーラが失敗しました (exit $($p.ExitCode))" }
    foreach ($f in @("es-sim.exe", "uninstall.exe", "THIRD_PARTY_NOTICES.txt", "backend\es-sim-backend.exe", "backend\_internal\cuda\bin\x64\nvrtc64_130_0.dll")) {
        if (-not (Test-Path (Join-Path $Dir $f))) { throw "インストール先に $f がありません" }
    }
    $installed = (Get-ChildItem -Recurse -File $Dir | Measure-Object -Sum Length).Sum / 1MB
    Write-Host ("インストール {0:N1} s、{1:N0} MB" -f ((Get-Date) - $t0).TotalSeconds, $installed)

    Step "同梱のバックエンドの自己テスト"
    $testArgs = @("selftest")
    if (-not $NoGpu) { $testArgs += "--require-gpu" }
    & (Join-Path $Dir "backend\es-sim-backend.exe") @testArgs
    if ($LASTEXITCODE -ne 0) { throw "同梱のバックエンドの自己テストに失敗しました" }

    Step "アプリを起動 (CDP $CdpPort、WebView2 のデータ $WebView)"
    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = "--remote-debugging-port=$CdpPort"
    $env:WEBVIEW2_USER_DATA_FOLDER = $WebView
    $t0 = Get-Date
    $app = Start-Process -FilePath (Join-Path $Dir "es-sim.exe") -PassThru
    $cdp = $false
    while (((Get-Date) - $t0).TotalSeconds -lt 60) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$CdpPort/json/version" -TimeoutSec 2 | Out-Null
            $cdp = $true
            break
        }
        catch { Start-Sleep -Milliseconds 300 }
    }
    if (-not $cdp) { throw "WebView2 の CDP ($CdpPort) につながりません" }
    Write-Host ("ウィンドウ (CDP) まで {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds)
    if (-not (Test-Path (Join-Path $WebView "EBWebView"))) {
        throw "WebView2 のデータが一時フォルダに作られていません (普段のデータを使っている恐れがあるので止めます)"
    }
    if ((Backend-Processes).Count -lt 1) { throw "同梱のバックエンドが起動していません (AppConfig の backend.log を確認)" }

    Step ("UI から計算 (Playwright、e2e-installed、{0})" -f $(if ($NoGpu) { "CPU" } else { "GPU" }))
    $env:E2E_CDP_URL = "http://127.0.0.1:$CdpPort"
    $Shot = Join-Path ([IO.Path]::GetTempPath()) "es-sim-verify-last.png"   # 確認用 (一時フォルダの外に残す)
    $env:E2E_SCREENSHOT = $Shot
    Push-Location (Join-Path $Root "ui")
    try {
        & npx playwright test --config playwright.installed.config.ts
        if ($LASTEXITCODE -ne 0) { throw "インストールしたアプリの E2E に失敗しました" }
    }
    finally { Pop-Location }

    Step "アプリを閉じる"
    $app.CloseMainWindow() | Out-Null
    if (-not $app.WaitForExit(30000)) { throw "アプリが閉じません" }
    $t0 = Get-Date
    while ((Backend-Processes).Count -gt 0 -and ((Get-Date) - $t0).TotalSeconds -lt 15) { Start-Sleep -Milliseconds 300 }
    if ((Backend-Processes).Count -gt 0) { throw "アプリを閉じてもバックエンドが残っています" }
    Write-Host "バックエンドも止まった"
    Write-Host "画面: $Shot"
    $ok = $true
}
finally {
    if ($app -and -not $app.HasExited) { Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue }
    foreach ($b in Backend-Processes) { Stop-Process -Id $b.Id -Force -ErrorAction SilentlyContinue }
    if ($KeepInstalled) {
        Write-Host "インストールしたまま: $Dir (アンインストールは $Dir\uninstall.exe)"
    }
    elseif (Test-Path (Join-Path $Dir "uninstall.exe")) {
        Step "アンインストール"
        Start-Process -FilePath (Join-Path $Dir "uninstall.exe") -ArgumentList "/S" -Wait
        $t0 = Get-Date
        while ((Test-Path (Join-Path $Dir "es-sim.exe")) -and ((Get-Date) - $t0).TotalSeconds -lt 120) { Start-Sleep -Milliseconds 500 }
        if (Test-Path (Join-Path $Dir "es-sim.exe")) { Write-Warning "アンインストールが終わりません: $Dir" }
        else {
            Write-Host "アンインストールした"
            Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
        }
    }
}
if ($ok) { Write-Host "== インストーラの確認: すべて通りました ==" }
