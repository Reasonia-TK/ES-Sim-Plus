// DSMC WebSocket クライアント (薄いラッパ)。picClient.ts と同じ流儀。
// ws://127.0.0.1:<port>/ws/dsmc に接続し、start/stop コマンドを送信する。
// サーバーからの started/progress/done/error 通知はコールバックへそのまま中継する。
// 接続エラー・切断時は onError / onClose を呼び、パネル側で待機状態に戻せるようにする。

import type {
  DsmcClientCommand,
  DsmcDoneMsg,
  DsmcErrorMsg,
  DsmcProgressMsg,
  DsmcServerMessage,
  DsmcStartedMsg,
  Project,
} from "./types";
import { getPort } from "./backendPort";

// 接続の都度ポート番号を組み立てる (GUIでの変更を即座に反映するため、定数 URL は使わない)
function wsUrl(): string {
  return `ws://127.0.0.1:${getPort()}/ws/dsmc`;
}

export interface DsmcClientCallbacks {
  onStarted?: (msg: DsmcStartedMsg) => void;
  onProgress?: (msg: DsmcProgressMsg) => void;
  onDone?: (msg: DsmcDoneMsg) => void;
  onError?: (detail: string) => void;
  // 接続が閉じた (正常終了・異常切断いずれも) ときに呼ばれる
  onClose?: () => void;
}

export class DsmcClient {
  private ws: WebSocket | null = null;
  private cb: DsmcClientCallbacks;

  constructor(cb: DsmcClientCallbacks) {
    this.cb = cb;
  }

  // コールバックを差し替える (picClient と同様、将来的な使い回しに備える)
  setCallbacks(cb: DsmcClientCallbacks): void {
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
      let msg: DsmcServerMessage;
      try {
        msg = JSON.parse(ev.data) as DsmcServerMessage;
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
          this.cb.onError?.((msg as DsmcErrorMsg).detail);
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

  // 接続して start コマンドを送る (project は dsmc 設定込みで渡すこと)
  start(project: Project): void {
    this.connect((ws) => {
      const cmd: DsmcClientCommand = { cmd: "start", project };
      ws.send(JSON.stringify(cmd));
    });
  }

  // 保持中のシミュレーション状態から追加実行する continue コマンドを送る
  // (picClient.continueRun と同じ流儀)。直前の start/continue と同じ接続が
  // 開いたままならそれをそのまま使い、閉じていれば新規接続してから送信する
  continueRun(opts: { n_steps: number; avg_steps?: number | null }): void {
    const cmd: DsmcClientCommand = { cmd: "continue", ...opts };
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
      return;
    }
    this.connect((ws) => ws.send(JSON.stringify(cmd)));
  }

  // stop コマンドを送る (接続していなければ何もしない)
  stop(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const cmd: DsmcClientCommand = { cmd: "stop" };
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
