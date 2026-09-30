// E2E (Playwright): 実物のバックエンド (uvicorn) と UI の開発サーバを、普段の開発 (8317・1421) とぶつからない
// ポートで立て、PC の Chrome で操作する (ブラウザの取得は不要)。CI では Playwright の Chromium を使う
// (E2E_CHANNEL を空にする)。Python は backend/.venv があればそれ、無ければ PATH の python (E2E_PYTHON で指定可)。

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const root = fileURLToPath(new URL("..", import.meta.url));
const venvPython = process.platform === "win32" ? `${root}backend/.venv/Scripts/python.exe` : `${root}backend/.venv/bin/python`;
const python = process.env.E2E_PYTHON || (existsSync(venvPython) ? venvPython : "python");
const backendPort = Number(process.env.E2E_BACKEND_PORT || 8327);
const uiPort = Number(process.env.E2E_UI_PORT || 1431);
const channel = process.env.E2E_CHANNEL ?? (process.env.CI ? "" : "chrome");

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  expect: { timeout: 30_000 },
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://localhost:${uiPort}`,
    channel: channel || undefined,
    viewport: { width: 1600, height: 900 },
    locale: "ja-JP",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `"${python}" -m uvicorn es_sim.server:app --port ${backendPort} --app-dir "${root}backend"`,
      url: `http://127.0.0.1:${backendPort}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
    {
      command: `npx vite --port ${uiPort} --strictPort`,
      url: `http://localhost:${uiPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
