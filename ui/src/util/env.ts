/** Tauri アプリ内か ("__TAURI_INTERNALS__" は Tauri v2 の WebView に注入されるオブジェクト) */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}
