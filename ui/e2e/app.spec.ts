// E2E: シェル・作図と元に戻す・静電場・ジョブと結果・結果付き保存と読み込み・サンプルの保存された結果
// (実物のバックエンドで)。

import { readFileSync } from "node:fs";
import { enableStudy, expect, menu, openApp, selectNode, setNumber, test } from "./fixtures";

test("opens the sample and connects to the backend", async ({ page }) => {
  await openApp(page);
  await expect(page).toHaveTitle(/ES-Sim/);
  await expect(page.locator(".tree")).toContainText("平行平板");
  await expect(page.locator(".status-bar")).toContainText("backend v");
  // 設定を変えると未保存の印が付き、元に戻すと消える
  await selectNode(page, "ドメイン");
  await setNumber(page, "幅", "120");
  await expect(page).toHaveTitle(/^● /);
  await page.keyboard.press("Control+z");
  await expect(page).not.toHaveTitle(/^● /);
});

test("draws a rectangle on the canvas and undoes it", async ({ page }) => {
  await openApp(page);
  await page.getByRole("button", { name: "矩形", exact: true }).click();
  const canvas = page.getByRole("application");
  const box = (await canvas.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.35);
  await page.mouse.click(box.x + box.width * 0.4, box.y + box.height * 0.5);
  await expect(page.locator(".tree")).toContainText("region1");
  await page.keyboard.press("Escape");
  await page.keyboard.press("Control+z");
  await expect(page.locator(".tree")).not.toContainText("region1");
});

test("builds a mesh and solves electrostatics", async ({ page }) => {
  await openApp(page);
  await selectNode(page, "メッシュ");
  await page.getByRole("button", { name: "メッシュを作成" }).click();
  await expect(page.getByText(/節点 \d+・要素 \d+/).first()).toBeVisible();
  await selectNode(page, "静電場");
  await page.getByRole("button", { name: "計算", exact: true }).click();
  await expect(page.getByText("静電容量")).toBeVisible();
  await expect(page.getByText("電極の電荷")).toBeVisible();
  await expect(page.locator(".viewer-title")).toContainText("電位");
});

test("runs PIC 1D as a job and shows its charts and summary", async ({ page }) => {
  await openApp(page);
  await selectNode(page, "PIC-MCC (1D)");
  await enableStudy(page);
  await setNumber(page, "ステップ数", "3000");
  await page.locator(".run-controls").getByRole("button", { name: "実行", exact: true }).click();
  await expect(page.locator(".run-controls")).toContainText("完了", { timeout: 150_000 });
  // 1D の実行はグラフのタブに出る
  await expect(page.getByRole("tab", { name: "グラフ" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByText("時間平均のプロファイル")).toBeVisible();
  await selectNode(page, "PIC 1D #1");
  await expect(page.getByText("数値サマリ")).toBeVisible();
  await expect(page.getByText("経過時間 (壁時計)")).toBeVisible();
});

test("saves with results and opens them back as an imported run", async ({ page }) => {
  await openApp(page);
  await selectNode(page, "PIC-MCC (1D)");
  await enableStudy(page);
  await setNumber(page, "ステップ数", "2000");
  await page.locator(".run-controls").getByRole("button", { name: "実行", exact: true }).click();
  await expect(page.locator(".run-controls")).toContainText("完了", { timeout: 150_000 });

  const download = page.waitForEvent("download");
  await menu(page, "ファイル", "結果付きで保存…");
  const file = await (await download).path();
  const saved = JSON.parse(readFileSync(file, "utf8"));
  expect(saved.results.version).toBe(1);
  expect(saved.results.pic1d.profiles).toBeTruthy();

  const chooser = page.waitForEvent("filechooser");
  await menu(page, "ファイル", "開く…");
  // 実行の設定を変えたので未保存の確認が出る
  await page.getByRole("button", { name: "保存しない" }).click();
  await (await chooser).setFiles(file);
  await expect(page.locator(".tree")).toContainText("PIC 1D (");
  await expect(page.getByText("時間平均のプロファイル")).toBeVisible();
});

test("opens an example together with its saved results", async ({ page }) => {
  await openApp(page);
  await page.getByRole("menuitem", { name: "ファイル", exact: true }).click();
  await page.getByRole("menuitem", { name: /^サンプル/ }).click(); // 下位のメニュー (名前に › が続く)
  await page.getByRole("menuitem", { name: "容量結合プラズマ (PIC)", exact: true }).click();
  // examples/results/ccp_demo.json.gz が「読み込んだ実行」として並ぶ (prompts/135)
  const run = page.locator(".tree").getByText(/^PIC \(容量結合プラズマ \(PIC\)、保存された結果・RF [\d.]+ 周期\)/);
  await expect(run).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".messages-list")).toContainText("保存された計算結果 1 件");
  await run.click();
  await expect(page.getByRole("heading", { name: /^結果 › PIC \(容量結合プラズマ \(PIC\)、保存された結果/ })).toBeVisible();
  await expect(page.getByText("時間平均したステップ数")).toBeVisible();
  await page.getByRole("button", { name: "グラフで見る" }).click();
  await expect(page.getByRole("tab", { name: "グラフ" })).toHaveAttribute("aria-selected", "true");
});
