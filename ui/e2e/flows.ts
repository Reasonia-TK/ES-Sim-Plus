// UI から計算を確かめる流れ (配布 P8c・P8d、prompts/133)。開発用の E2E (gpu.spec.ts) と、インストールしたアプリに
// CDP でつなぐ確認 (e2e-installed/、scripts/verify_installer.ps1。GPU の無い PC・CI では CPU の流れ) の両方で使う。

import { expect, type Page } from "@playwright/test";
import { enableStudy, selectNode, setNumber } from "./fixtures";

/** サンプルの容量結合プラズマを直交格子 (v2。PIC は GPU だけで動く) にして短く走らせ、実行のページの実行時間の内訳まで見る */
export async function runGpuPic(page: Page, steps = 300): Promise<void> {
  await page.getByRole("menuitem", { name: "ファイル", exact: true }).click();
  await page.getByRole("menuitem", { name: /^サンプル/ }).click(); // 下位のメニュー (名前に › が続く)
  await page.getByRole("menuitem", { name: "容量結合プラズマ (PIC)", exact: true }).click();
  await expect(page.locator(".tree")).toContainText("PIC-MCC (2D)");
  await selectNode(page, "メッシュ");
  await page.getByLabel("方式", { exact: true }).selectOption({ label: "直交格子 + 埋め込み境界 (v2・GPU)" });
  await selectNode(page, "PIC-MCC (2D)");
  await enableStudy(page);
  await setNumber(page, "ステップ数", String(steps));
  await page.locator(".run-controls").getByRole("button", { name: "実行", exact: true }).click();
  await expect(page.locator(".run-controls")).toContainText("完了", { timeout: 300_000 });
  await page.locator(".tree").getByText(/^PIC #\d+$/).last().click();
  await expect(page.getByRole("heading", { name: "実行時間の内訳" })).toBeVisible();
  await expect(page.getByText("経過時間 (壁時計)")).toBeVisible();
  await expect(page.getByText(`${steps} / ${steps}`, { exact: true })).toBeVisible();
}

/** バージョン情報の計算デバイスを確かめる (GPU なら GPU の名前、CPU なら使えない理由)。閉じて戻り、中身を返す */
export async function expectDeviceInAbout(page: Page, gpu: boolean): Promise<string> {
  await page.getByRole("menuitem", { name: "ヘルプ", exact: true }).click();
  await page.getByRole("menuitem", { name: "バージョン情報", exact: true }).click();
  const dialog = page.getByRole("dialog");
  if (gpu) await expect(dialog).toContainText(/計算デバイス\s*GPU \(/);
  else await expect(dialog).toContainText(/計算デバイス\s*CPU[\s\S]*GPU を使えない理由/);
  const text = (await dialog.innerText()).replace(/\s+/g, " ");
  await dialog.getByRole("button", { name: "閉じる" }).click();
  await expect(dialog).toBeHidden();
  return text;
}

/** 開いている文書 (既定のサンプルは非構造メッシュ) でメッシュを作り静電場を解く (CPU: gmsh と FEM) */
export async function runStaticSolve(page: Page): Promise<void> {
  await selectNode(page, "メッシュ");
  await page.getByRole("button", { name: "メッシュを作成" }).click();
  await expect(page.getByText(/節点 \d+・要素 \d+/).first()).toBeVisible();
  await selectNode(page, "静電場");
  await page.getByRole("button", { name: "計算", exact: true }).click();
  await expect(page.getByText("静電容量")).toBeVisible();
}

/** 未保存の変更があれば新規 (保存しない) にする (ウィンドウを閉じるときの未保存の確認で止まらないように) */
export async function discardChanges(page: Page): Promise<void> {
  if (!(await page.title()).startsWith("● ")) return;
  await page.getByRole("menuitem", { name: "ファイル", exact: true }).click();
  await page.getByRole("menuitem", { name: /^新規/ }).click();
  await page.getByRole("alertdialog").getByRole("button", { name: "保存しない", exact: true }).click();
  await expect(page).not.toHaveTitle(/^● /);
}
