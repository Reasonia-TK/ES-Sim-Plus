// インストールしたアプリ (Tauri・WebView2) の確認 (配布 P8c、prompts/133)。scripts/verify_installer.ps1 がアプリを
// WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=<port> と一時フォルダの WebView2 のデータで起動し、
// E2E_CDP_URL を渡して走らせる。同梱のバックエンドにつながり、GPU で計算できることを UI から確かめる。

import { chromium, expect, test } from "@playwright/test";
import { expectGpuInAbout, runGpuPic } from "../e2e/gpuFlow";

const CDP_URL = process.env.E2E_CDP_URL ?? "http://127.0.0.1:9233";

test("installed app connects to its bundled backend and runs PIC on the GPU", async () => {
  const browser = await chromium.connectOverCDP(CDP_URL);
  try {
    const page = browser.contexts()[0].pages()[0];
    const t0 = Date.now();
    await expect(page.locator(".status-bar")).toContainText("backend v", { timeout: 120_000 });
    console.log(`backend connected after ${((Date.now() - t0) / 1000).toFixed(1)} s (from CDP attach)`);
    // 一時フォルダの WebView2 のデータなので、前回の未保存の変更 (自動保存の復元) は出ないはず。出たら触らずに止める
    await expect(page.getByRole("alertdialog", { name: "前回の未保存の変更" })).toHaveCount(0);
    await expect(page.locator(".status-badge")).toContainText("(GPU)");
    console.log(await expectGpuInAbout(page));
    await runGpuPic(page);
    if (process.env.E2E_SCREENSHOT) await page.screenshot({ path: process.env.E2E_SCREENSHOT });
    // verify_installer.ps1 がウィンドウを閉じるとき未保存の確認で止まらないよう、新規 (保存しない) にしておく
    await page.getByRole("menuitem", { name: "ファイル", exact: true }).click();
    await page.getByRole("menuitem", { name: /^新規/ }).click();
    await page.getByRole("alertdialog").getByRole("button", { name: "保存しない", exact: true }).click();
    await expect(page).not.toHaveTitle(/^● /);
  } finally {
    await browser.close(); // CDP の接続を切るだけ (アプリは閉じない。閉じるのは verify_installer.ps1)
  }
});
