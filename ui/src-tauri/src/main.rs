// Rust シェルは薄く保つ (仕様書 §14)。
// 配布ビルドでは Python バックエンド (PyInstaller 製) をサイドカーとして
// 自動起動し、アプリ終了時に kill する (prompts/44)。
// 開発モード (`tauri dev`) ではサイドカーを起動しない — デバッグは従来通り
// 手動 uvicorn (`uvicorn es_sim.server:app --port 8317`) を使う。
// フロントエンドは UI v2 (`ui/`、prompts/130 P6f)。v1 (`frontend/`) からここへ移した。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::Manager;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// 起動したバックエンドの子プロセスハンドル (終了時の kill 用)
struct BackendChild(Mutex<Option<CommandChild>>);

/// サイドカー診断ログ (AppConfig/backend.log) へ1行追記する。
/// バックエンドが起動しない問題の調査用 — 失敗しても本体動作には影響させない。
fn log_line(path: &Option<PathBuf>, msg: &str) {
    if let Some(p) = path {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(f, "[{ts}] {msg}");
        }
    }
}

/// サイドカーをプロセスツリーごと止める。PyInstaller の onefile は展開役の親と本体の子の 2 つのプロセスで動き、
/// 親だけを kill すると本体 (uvicorn) が残ってポートを塞ぐ (P6f の確認で判明、v1 も同じだった)。
#[cfg(windows)]
fn kill_tree(pid: u32) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .status();
}

#[cfg(not(windows))]
fn kill_tree(_pid: u32) {}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(BackendChild(Mutex::new(None)))
        .setup(|app| {
            // 配布ビルドのみサイドカーを起動する (cfg!(debug_assertions) で分岐)
            if !cfg!(debug_assertions) {
                // GUI (フロントエンド) が AppConfig ディレクトリへ書き込む backend-port.txt を
                // 読み、サイドカーの --port に渡す (prompts/45)。
                // 読み取り専用: ディレクトリ・ファイルが無ければ作成せず既定値 8317 を使う。
                // parse失敗・不存在時も既定値 8317 にフォールバックする。
                let config_dir = app.path().app_config_dir().ok();
                // 診断ログの出力先 (AppConfig/backend.log)。ディレクトリは無ければ作成する
                let log_path = config_dir.as_ref().map(|d| {
                    let _ = std::fs::create_dir_all(d);
                    d.join("backend.log")
                });
                let port = config_dir
                    .as_ref()
                    .and_then(|dir| std::fs::read_to_string(dir.join("backend-port.txt")).ok())
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(8317);
                log_line(&log_path, &format!("spawning es-sim-backend --port {port}"));
                let sidecar = match app.shell().sidecar("es-sim-backend") {
                    Ok(cmd) => cmd.args(["--port", &port.to_string()]),
                    Err(e) => {
                        // 以前は expect でパニックしていたが、それだとアプリごと落ちて
                        // 原因が全く分からない。ログに残して GUI は起動させる。
                        log_line(&log_path, &format!("sidecar command error: {e}"));
                        return Ok(());
                    }
                };
                match sidecar.spawn() {
                    Ok((mut rx, child)) => {
                        log_line(&log_path, &format!("spawned pid={}", child.pid()));
                        *app.state::<BackendChild>().0.lock().unwrap() = Some(child);
                        // 出力イベントを排出しつつ診断ログへ記録する (チャネル詰まり防止も兼ねる)
                        tauri::async_runtime::spawn(async move {
                            while let Some(event) = rx.recv().await {
                                match event {
                                    CommandEvent::Stdout(line) => log_line(
                                        &log_path,
                                        &format!("stdout: {}", String::from_utf8_lossy(&line).trim_end()),
                                    ),
                                    CommandEvent::Stderr(line) => log_line(
                                        &log_path,
                                        &format!("stderr: {}", String::from_utf8_lossy(&line).trim_end()),
                                    ),
                                    CommandEvent::Error(e) => {
                                        log_line(&log_path, &format!("process error: {e}"))
                                    }
                                    CommandEvent::Terminated(t) => log_line(
                                        &log_path,
                                        &format!("terminated: code={:?} signal={:?}", t.code, t.signal),
                                    ),
                                    _ => {}
                                }
                            }
                        });
                    }
                    Err(e) => {
                        log_line(&log_path, &format!("spawn error: {e}"));
                    }
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                // アプリ終了時にバックエンドの子プロセスを確実に殺す
                if let Some(child) = app.state::<BackendChild>().0.lock().unwrap().take() {
                    kill_tree(child.pid());
                    let _ = child.kill();
                }
            }
        });
}
