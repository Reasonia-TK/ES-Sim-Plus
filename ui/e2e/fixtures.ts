// E2E の共通: UI を E2E 用のバックエンドのポートにつなぎ、ファイルの読み書きを <input type=file> とダウンロードに
// させる (File System Access API のダイアログは自動操作できないため)。

import { test as base, expect, type Page } from "@playwright/test";

export const BACKEND_PORT = Number(process.env.E2E_BACKEND_PORT || 8327);

export const test = base.extend({
  page: async ({ page }, use) => {
    await page.addInitScript((port) => {
      localStorage.setItem("es-sim.backendPort", String(port));
      for (const k of ["showOpenFilePicker", "showSaveFilePicker"]) Object.defineProperty(window, k, { value: undefined, configurable: true });
    }, BACKEND_PORT);
    await use(page);
  },
});

export { expect };

/** アプリを開いてバックエンドにつながるまで待つ */
export async function openApp(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.locator(".statusbar, .status-bar").first()).toContainText("backend v", { timeout: 60_000 });
}

/** ツリーの節を選ぶ (表示名が一致する最初のもの) */
export async function selectNode(page: Page, label: string): Promise<void> {
  await page.locator(".tree").getByText(label, { exact: true }).first().click();
}

/** スタディのページで「このスタディを使う」をオンにする */
export async function enableStudy(page: Page): Promise<void> {
  const toggle = page.getByRole("checkbox", { name: "このスタディを使う" });
  if (!(await toggle.isChecked())) await page.getByText("このスタディを使う", { exact: true }).click();
  await expect(toggle).toBeChecked();
}

/** 数値欄に入れて確定する */
export async function setNumber(page: Page, label: string, value: string): Promise<void> {
  const input = page.getByLabel(label, { exact: true }).first();
  await input.fill(value);
  await input.press("Enter");
}

/** メニューの項目を選ぶ (Radix のメニューバー) */
export async function menu(page: Page, top: string, item: string): Promise<void> {
  await page.getByRole("menuitem", { name: top, exact: true }).click();
  // 項目の名前にはショートカットの表記も続くので前方一致
  const re = new RegExp(`^${item.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`);
  await page.getByRole("menuitem", { name: re }).click();
}
