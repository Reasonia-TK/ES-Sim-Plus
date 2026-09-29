// jsdom に無いブラウザ API の代わり (ペインのリサイズ・キャンバスの大きさの監視に使う)
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "../src/i18n";

class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

afterEach(() => {
  cleanup();
  localStorage.clear();
});
