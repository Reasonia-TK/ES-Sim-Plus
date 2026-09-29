/// <reference types="vitest/config" />
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// v1 (frontend/) の 1420 と並べて開発できるよう 1421 を固定で使う (Tauri 切替時は devUrl を合わせる)
const repoRoot = fileURLToPath(new URL("..", import.meta.url));

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1421,
    strictPort: true,
    // リポジトリ直下の examples/*.json をサンプルとして取り込むため
    fs: { allow: [repoRoot] },
  },
  build: { target: "es2022" },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
    setupFiles: ["tests/setup.ts"],
  },
});
