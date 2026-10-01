# ES-Sim のインストーラの確認 (Windows、prompts/133 P8c)。
#
#   powershell -ExecutionPolicy Bypass -File scripts\verify_installer.ps1 [-Installer <setup.exe>] [-KeepInstalled] [-NoGpu] [-AllowNoCdp]
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
# -AllowNoCdp: WebView2 の CDP が開かないとき (GitHub のランナーで起きた) は手がかりを出し、UI がバックエンドに
# つながったこと (スキーマとイベントの要求) を backend.log で確かめて続ける (UI の操作は飛ばす)。
param(
    [string]$Installer = "",
    [switch]$KeepInstalled,
    [switch]$NoGpu,
    [switch]$AllowNoCdp,
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

# CDP が開かないときの手がかり: WebView2 のランタイムの版、データの置き場、WebView2 のプロセスの引数
function Show-WebView2Diagnostics {
    $guid = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    foreach ($k in @("HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$guid", "HKCU:\Software\Microsoft\EdgeUpdate\Clients\$guid")) {
        $pv = (Get-ItemProperty $k -ErrorAction SilentlyContinue).pv
        if ($pv) { Write-Host "WebView2 ランタイム: $pv ($k)" }
    }
    Write-Host "WebView2 のデータが一時フォルダにある: $(Test-Path (Join-Path $WebView 'EBWebView'))"
    Get-CimInstance Win32_Process -Filter "Name='msedgewebview2.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "--webview-exe-name=es-sim\.exe" -and $_.CommandLine -notmatch "--type=" } |
        ForEach-Object { Write-Host "msedgewebview2 (pid $($_.ProcessId)): $($_.CommandLine)" }
}

# backend.log (AppConfig) の、このインストール先から起動した後の行
function Backend-Log-Since-Spawn {
    $log = Join-Path $env:APPDATA "com.tk.es-sim\backend.log"
    if (-not (Test-Path $log)) { return @() }
    $lines = @(Get-Content $log -Encoding UTF8)
    $start = -1
    for ($i = $lines.Count - 1; $i -ge 0; $i--) {
        if ($lines[$i] -match ("spawning .*" + [regex]::Escape((Split-Path (Split-Path $Dir) -Leaf)))) { $start = $i; break }
    }
    if ($start -lt 0) { return @() }
    return $lines[$start..($lines.Count - 1)]
}

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
    foreach ($f in @("es-sim.exe", "uninstall.exe", "LICENSE.txt", "THIRD_PARTY_NOTICES.txt", "backend\es-sim-backend.exe", "backend\_internal\cuda\bin\x64\nvrtc64_130_0.dll")) {
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
    while (((Get-Date) - $t0).TotalSeconds -lt 90) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$CdpPort/json/version" -TimeoutSec 2 | Out-Null
            $cdp = $true
            break
        }
        catch { Start-Sleep -Milliseconds 300 }
    }
    if ((Backend-Processes).Count -lt 1) { throw "同梱のバックエンドが起動していません (AppConfig の backend.log を確認)" }
    $Shot = $null
    if ($cdp) {
        Write-Host ("ウィンドウ (CDP) まで {0:N1} s" -f ((Get-Date) - $t0).TotalSeconds)
        if (-not (Test-Path (Join-Path $WebView "EBWebView"))) {
            throw "WebView2 のデータが一時フォルダに作られていません (普段のデータを使っている恐れがあるので止めます)"
        }
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
    }
    else {
        Show-WebView2Diagnostics
        if (-not $AllowNoCdp) { throw "WebView2 の CDP ($CdpPort) につながりません" }
        Write-Warning "WebView2 の CDP ($CdpPort) が開かないので、UI の操作は飛ばして backend.log で接続を確かめます"
        $log = Backend-Log-Since-Spawn
        $schema = @($log | Where-Object { $_ -match "GET /v2/schema HTTP/1.1. 200" }).Count
        $events = @($log | Where-Object { $_ -match "WebSocket /v2/events. \[accepted\]" }).Count
        if ($schema -lt 1 -or $events -lt 1) { throw "UI がバックエンドにつながっていません (backend.log: スキーマ $schema 回、イベント $events 回)" }
        Write-Host "UI がバックエンドにつながった (backend.log: スキーマ $schema 回、イベントの接続 $events 回)"
    }

    Step "アプリを閉じる"
    $app.CloseMainWindow() | Out-Null
    if (-not $app.WaitForExit(30000)) { throw "アプリが閉じません" }
    $t0 = Get-Date
    while ((Backend-Processes).Count -gt 0 -and ((Get-Date) - $t0).TotalSeconds -lt 15) { Start-Sleep -Milliseconds 300 }
    if ((Backend-Processes).Count -gt 0) { throw "アプリを閉じてもバックエンドが残っています" }
    Write-Host "バックエンドも止まった"
    if ($Shot) { Write-Host "画面: $Shot" }
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
