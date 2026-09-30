// E2E: CAD v2 (prompts/132)。外周の辺を円弧にし、キャンバスで辺を選んで境界条件を付け、辺を分けて静電場を解く (P7a)。
// スケッチの線から囲まれた所で領域を作る・円弧のある折れ線の領域・スケッチの矩形をドメインに・範囲選択と削除・
// キャンバスでドメインの頂点を動かし辺を曲げる (P7b)。

import type { Page } from "@playwright/test";
import { expect, openApp, selectNode, test } from "./fixtures";

/** ルーラーの帯の幅 [px] (既定の文字の大きさ 11 px、graphics/overlay.ts の rulerSize と同じ) */
const RULER_PX = Math.max(24, 11 * 2.2);

/** 全体表示にして、ワールド座標 [mm] → ページの座標 (外接矩形 bounds [mm] に合わせた camera.fitCamera と同じ計算) */
async function fitView(page: Page, bounds = { x0: 0, y0: 0, x1: 100, y1: 50 }): Promise<(x: number, y: number) => [number, number]> {
  const canvas = page.getByRole("application");
  await canvas.focus();
  await page.keyboard.press("f");
  const box = (await canvas.boundingBox())!;
  const w = box.width - RULER_PX;
  const h = box.height - RULER_PX;
  const bw = bounds.x1 - bounds.x0;
  const bh = bounds.y1 - bounds.y0;
  const scale = Math.min((0.8 * w) / bw, (0.8 * h) / bh);
  const cx = (bounds.x0 + bounds.x1) / 2;
  const cy = (bounds.y0 + bounds.y1) / 2;
  return (x, y) => [box.x + RULER_PX + w / 2 + (x - cx) * scale, box.y + RULER_PX + h / 2 - (y - cy) * scale];
}

async function clickAt(page: Page, p: [number, number]): Promise<void> {
  await page.mouse.click(p[0], p[1]);
}

async function drag(page: Page, a: [number, number], b: [number, number], shift = false): Promise<void> {
  if (shift) await page.keyboard.down("Shift");
  await page.mouse.move(a[0], a[1]);
  await page.mouse.down();
  await page.mouse.move((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, { steps: 4 });
  await page.mouse.move(b[0], b[1], { steps: 4 });
  await page.mouse.up();
  if (shift) await page.keyboard.up("Shift");
}

const tool = (page: Page, name: string) => page.getByRole("button", { name, exact: true }).click();

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
  const at = await fitView(page, { x0: 0, y0: 0, x1: 100, y1: 63.3975 });
  await clickAt(page, at(50, 63.3975 - 0.05));
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

test("draws sketch lines and makes a region from the enclosed area", async ({ page }) => {
  await openApp(page);
  const at = await fitView(page);
  await tool(page, "線 (スケッチ)");
  // 続けて 4 本 (最後に最初の点に戻る)
  for (const p of [at(5, 5), at(30, 5), at(30, 20), at(5, 20), at(5, 5)]) await clickAt(page, p);
  await page.keyboard.press("Escape");
  await selectNode(page, "スケッチ");
  await expect(page.getByText("スケッチ 4 個")).toBeVisible();
  // 囲まれた所をクリックすると、その形の領域
  await tool(page, "囲まれた所から領域");
  await clickAt(page, at(15, 12));
  await expect(page.locator(".settings-panel .panel-title")).toContainText("region1");
  await expect(page.locator(".vertex-table tbody tr")).toHaveCount(4);
  await expect(page.getByLabel("x 1", { exact: true })).toHaveValue(/^(5|30)$/);
});

test("draws a region polyline with an arc segment and closes it on the first point", async ({ page }) => {
  await openApp(page);
  const at = await fitView(page);
  await tool(page, "折れ線");
  const canvas = page.getByRole("application");
  await clickAt(page, at(5, 5));
  await clickAt(page, at(30, 5));
  // A で円弧: 通る点 (35, 15) と終点 (30, 25)。弦 20 mm・矢高 5 mm → 半径 12.5 mm、中心角 106.26°
  await canvas.press("a");
  await clickAt(page, at(35, 15));
  await clickAt(page, at(30, 25));
  await canvas.press("l");
  await clickAt(page, at(5, 25));
  await clickAt(page, at(5, 5));
  await expect(page.locator(".settings-panel .panel-title")).toContainText("region1");
  await expect(page.locator(".vertex-table tbody tr")).toHaveCount(4);
  await expect(page.getByLabel("円弧 [°] 2", { exact: true })).toHaveValue(/^106\.26/);
});

test("turns a sketch rectangle into the domain and keeps the side conditions", async ({ page }) => {
  await openApp(page);
  const at = await fitView(page);
  await page.getByRole("radio", { name: "スケッチ" }).click();
  await tool(page, "矩形");
  await clickAt(page, at(0, 0));
  await clickAt(page, at(100, 40));
  await expect(page.locator(".settings-panel .panel-title")).toContainText("スケッチ");
  await page.getByRole("button", { name: "ドメインにする" }).click();
  await expect(page.locator(".tree")).toContainText("100 × 40 mm");
  // 左 0 V・右 100 V は同じ直線の上に残った辺に引き継ぐ
  await selectNode(page, "境界条件");
  await expect(page.locator(".settings-panel")).toContainText("Dirichlet 0 V");
  await expect(page.locator(".settings-panel")).toContainText("Dirichlet 100 V");
});

test("box-selects sketch items and deletes them", async ({ page }) => {
  await openApp(page);
  const at = await fitView(page);
  await tool(page, "線 (スケッチ)");
  for (const p of [at(5, 5), at(20, 10), at(30, 5)]) await clickAt(page, p);
  await page.keyboard.press("Escape");
  await tool(page, "選択");
  // 何も無い所から範囲で囲む
  await drag(page, at(2, 2), at(35, 15));
  await expect(page.locator(".settings-panel")).toContainText("2 個選んでいます");
  await page.getByRole("application").press("Delete");
  await selectNode(page, "スケッチ");
  await expect(page.getByText("スケッチ 0 個")).toBeVisible();
});

test("edits the domain on the canvas: moves a vertex and bends an edge", async ({ page }) => {
  await openApp(page);
  await selectNode(page, "ドメイン");
  const at = await fitView(page);
  // 右上の頂点 (100, 50) を (90, 45) へ
  await drag(page, at(100, 50), at(90, 45));
  await expect(page.getByLabel("x 3", { exact: true })).toHaveValue("90");
  await expect(page.getByLabel("y 3", { exact: true })).toHaveValue("45");
  // 右の辺 (100, 0) → (90, 45) の中点を Shift + ドラッグで外へ曲げる
  await drag(page, at(95, 22.5), at(100, 22.5), true);
  await expect(page.getByLabel("円弧 [°] 2", { exact: true })).not.toHaveValue("0");
  // 境界条件 (右 100 V) はそのまま
  await selectNode(page, "境界条件");
  await expect(page.locator(".settings-panel")).toContainText("Dirichlet 100 V");
});
