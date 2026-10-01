// インストールしたアプリに CDP でつなぐ確認 (配布 P8c、prompts/133)。開発サーバは立てない
// (scripts/verify_installer.ps1 がアプリを起動し、E2E_CDP_URL を渡す)。

import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "e2e-installed",
  timeout: 300_000,
  expect: { timeout: 60_000 },
  workers: 1,
  retries: 0,
  reporter: "list",
});
