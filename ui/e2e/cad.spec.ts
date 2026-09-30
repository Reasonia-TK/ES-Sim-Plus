// E2E: CAD v2 (prompts/132)。外周の辺を円弧にし、キャンバスで辺を選んで境界条件を付け、辺を分けて静電場を解く。

import { expect, openApp, selectNode, test } from "./fixtures";

/** ルーラーの帯の幅 [px] (既定の文字の大きさ 11 px、graphics/overlay.ts の rulerSize と同じ) */
const RULER_PX = Math.max(24, 11 * 2.2);

test("bends a domain edge into an arc, picks it on the canvas, splits it and solves", async ({ page }) => {
  await openApp(page);
  await selectNode(page, "ドメイン");
  await page.getByText("頂点と円弧", { exact: true }).click();
  // 上の辺 (頂点 3 → 4) を中心角 60° の円弧に: 弦 100 mm の円弧は 13.4 mm ふくらむ
  const arc = page.getByLabel("円弧 [°] 3", { exact: true });
  await arc.fill("60");
  await arc.press("Enter");
  await expect(page.locator(".tree")).toContainText("100 × 63.3975 mm");

  // 全体表示にしてから円弧の頂上 (外接矩形の上辺の中央) をクリックする
  const canvas = page.getByRole("application");
  await canvas.focus();
  await page.keyboard.press("f");
  const box = (await canvas.boundingBox())!;
  const w = box.width - RULER_PX;
  const h = box.height - RULER_PX;
  const scale = Math.min((0.8 * w) / 0.1, (0.8 * h) / 0.0633975);
  await page.mouse.click(box.x + RULER_PX + w / 2, box.y + RULER_PX + h / 2 - (0.0633975 / 2) * scale + 1);
  await expect(page.locator(".settings-panel .panel-title")).toContainText("境界条件 › 辺 2");
  await expect(page.getByText("円弧、中心角 60°、半径 100 mm")).toBeVisible();

  // Dirichlet (0 V) にしてから中点で分けると、2 本とも同じ条件になる
  await page.getByLabel("種類", { exact: true }).selectOption("dirichlet");
  await page.getByRole("button", { name: "中点で 2 本に分ける" }).click();
  await expect(page.getByText("この境界条件は 2 本の辺で共有しています", { exact: false })).toBeVisible();
  await expect(page.getByText("円弧、中心角 30°、半径 100 mm")).toBeVisible();

  // 静電場: 円弧の弦に分けた辺の電荷も元の辺ごとにまとまる
  await selectNode(page, "静電場");
  await page.getByRole("button", { name: "計算", exact: true }).click();
  await expect(page.getByText("電極の電荷")).toBeVisible();
  for (const label of ["辺 1 (100 V)", "辺 2 (0 V)", "辺 3 (0 V)", "辺 4 (0 V)"]) await expect(page.getByText(label, { exact: true })).toBeVisible();
});
