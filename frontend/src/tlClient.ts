// VHF 定在波 (非線形径方向伝送線路モデル) WebSocket クライアント (薄いラッパ)。
// dsmcClient.ts と同じ流儀。ws://127.0.0.1:<port>/ws/tl に接続し、start/stop コマンドを
// 送信する。continue は無い (tl.py の docstring 参照: 毎回フルの定常化をやり直す設計のため、
// dsmcClient/pic1dClient と違って continueRun() は持たない)。
// サーバーからの started/progress/done/error 通知はコールバックへそのまま中継する。
// 接続エラー・切断時は onError / onClose を呼び、パネル側で待機状態に戻せるようにする。

import type {
  TlClientCommand,
  TlDoneMsg,
  TlErrorMsg,
  TlProgressMsg,
  TlServerMessage,
  TlStartedMsg,
  Project,
} from "./types";
import { getPort } from "./backendPort";

// 接続の都度ポート番号を組み立てる (GUIでの変更を即座に反映するため、定数 URL は使わない)
function wsUrl(): string {
  return `ws://127.0.0.1:${getPort()}/ws/tl`;
}

export interface TlClientCallbacks {
  onStarted?: (msg: TlStartedMsg) => void;
  onProgress?: (msg: TlProgressMsg) => void;
  onDone?: (msg: TlDoneMsg) => void;
  onError?: (detail: string) => void;
  // 接続が閉じた (正常終了・異常切断いずれも) ときに呼ばれる
  onClose?: () => void;
}

export class TlClient {
  private ws: WebSocket | null = null;
  private cb: TlClientCallbacks;

  constructor(cb: TlClientCallbacks) {
    this.cb = cb;
  }

  setCallbacks(cb: TlClientCallbacks): void {
    this.cb = cb;
  }

  // WebSocket を新規に張り、接続確立後に onOpenSend で渡されたコマンドを送信する。
  private connect(onOpenSend: (ws: WebSocket) => void): void {
    this.close();
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl());
    } catch (e) {
      this.cb.onError?.(String(e));
      return;
    }
    this.ws = ws;

    ws.onopen = () => onOpenSend(ws);

    ws.onmessage = (ev: MessageEvent<string>) => {
      let msg: TlServerMessage;
      try {
        msg = JSON.parse(ev.data) as TlServerMessage;
      } catch (e) {
        this.cb.onError?.(`受信データの解析に失敗しました: ${String(e)}`);
        return;
      }
      switch (msg.type) {
        case "started":
          this.cb.onStarted?.(msg);
          break;
        case "progress":
          this.cb.onProgress?.(msg);
          break;
        case "done":
          this.cb.onDone?.(msg);
          break;
        case "error":
          this.cb.onError?.((msg as TlErrorMsg).detail);
          break;
      }
    };

    ws.onerror = () => {
      this.cb.onError?.("WebSocket接続エラーが発生しました (backend が起動しているか確認してください)");
    };

    ws.onclose = () => {
      this.ws = null;
      this.cb.onClose?.();
    };
  }

  // 接続して start コマンドを送る (project は tl 設定込みで渡すこと)
  start(project: Project): void {
    this.connect((ws) => {
      const cmd: TlClientCommand = { cmd: "start", project };
      ws.send(JSON.stringify(cmd));
    });
  }

  // stop コマンドを送る (接続していなければ何もしない)
  stop(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const cmd: TlClientCommand = { cmd: "stop" };
      this.ws.send(JSON.stringify(cmd));
    }
  }

  // 接続を明示的に閉じる (コールバックは発火させない)
  close(): void {
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
