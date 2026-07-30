// パラメータスイープ WebSocket クライアント (薄いラッパ)。dsmcClient.ts/picClient.ts と同じ流儀。
// ws://127.0.0.1:<port>/ws/sweep に接続し、start/stop コマンドを送信する。
// サーバーからの started/progress/case_done/done/error 通知はコールバックへそのまま中継する。
// 接続エラー・切断時は onError / onClose を呼び、パネル側で待機状態に戻せるようにする。

import type {
  SweepCaseDoneMsg,
  SweepClientCommand,
  SweepDoneMsg,
  SweepErrorMsg,
  SweepProgressMsg,
  SweepServerMessage,
  SweepStartedMsg,
  Project,
} from "./types";
import { getPort } from "./backendPort";

// 接続の都度ポート番号を組み立てる (GUIでの変更を即座に反映するため、定数 URL は使わない)
function wsUrl(): string {
  return `ws://127.0.0.1:${getPort()}/ws/sweep`;
}

export interface SweepClientCallbacks {
  onStarted?: (msg: SweepStartedMsg) => void;
  onProgress?: (msg: SweepProgressMsg) => void;
  onCaseDone?: (msg: SweepCaseDoneMsg) => void;
  onDone?: (msg: SweepDoneMsg) => void;
  onError?: (detail: string) => void;
  // 接続が閉じた (正常終了・異常切断いずれも) ときに呼ばれる
  onClose?: () => void;
}

export class SweepClient {
  private ws: WebSocket | null = null;
  private cb: SweepClientCallbacks;

  constructor(cb: SweepClientCallbacks) {
    this.cb = cb;
  }

  setCallbacks(cb: SweepClientCallbacks): void {
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
      let msg: SweepServerMessage;
      try {
        msg = JSON.parse(ev.data) as SweepServerMessage;
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
        case "case_done":
          this.cb.onCaseDone?.(msg);
          break;
        case "done":
          this.cb.onDone?.(msg);
          break;
        case "error":
          this.cb.onError?.((msg as SweepErrorMsg).detail);
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

  // 接続して start コマンドを送る (project は pic/pic1d/particles 等を合成済みのものを渡すこと)。
  // module は自動判定に頼らず、UI (SweepPanel) で確定した対象 ("pic"/"pic1d") を明示送信する
  // (prompts/96)。サーバーは同じ値を started.module としてそのまま返す
  start(project: Project, paramPath: string, values: number[], parallel: number, module: "pic" | "pic1d"): void {
    this.connect((ws) => {
      const cmd: SweepClientCommand = {
        cmd: "start", project, param_path: paramPath, values, parallel, module,
      };
      ws.send(JSON.stringify(cmd));
    });
  }

  // stop コマンドを送る (接続していなければ何もしない)
  stop(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const cmd: SweepClientCommand = { cmd: "stop" };
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
