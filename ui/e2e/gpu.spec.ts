// E2E: GPU の計算 (直交格子の PIC は GPU だけ)。バックエンドが GPU を使えない環境 (CI) では飛ばす (配布 P8c、prompts/133)。

import { expectGpuInAbout, runGpuPic } from "./gpuFlow";
import { openApp, test } from "./fixtures";

test("runs the cartesian PIC sample on the GPU", async ({ page }) => {
  await openApp(page);
  const badge = (await page.locator(".status-badge").textContent()) ?? "";
  test.skip(!badge.includes("(GPU)"), "バックエンドが GPU を使えない環境");
  await expectGpuInAbout(page);
  await runGpuPic(page);
});
