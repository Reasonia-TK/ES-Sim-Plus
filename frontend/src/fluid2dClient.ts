// 2D/軸対称 プラズマ流体 (ドリフト拡散 + 電子エネルギー) WebSocket クライアント
// (薄いラッパ、prompts/111-113)。ws://127.0.0.1:<port>/ws/fluid2d に接続し、
// start/continue/stop コマンドを送信する。fluid1dClient.ts (/ws/fluid1d) の忠実な複製 —
// バックエンド側の実行状態・ロックが fluid1d/pic とは完全に独立している
// (server.py の専用ロック・直近 sim 保持スロット、prompts/112) ため、別クラスとして持つ。
// continue の extra_steps は毎回明示指定が必要 (Fluid2dPanel が管理する、fluid1d と同じ設計)。

import type {
  Fluid2dClientCommand,
  Fluid2dDoneMsg,
  Fluid2dErrorMsg,
  Fluid2dFrameMsg,
  Fluid2dServerMessage,
  Fluid2dStartedMsg,
  Project,
} from "./types";
import { getPort } from "./backendPort";

// 接続の都度ポート番号を組み立てる (GUIでの変更を即座に反映するため、定数 URL は使わない)
function wsUrl(): string {
  return `ws://127.0.0.1:${getPort()}/ws/fluid2d`;
}

export interface Fluid2dClientCallbacks {
  onStarted?: (msg: Fluid2dStartedMsg) => void;
  onFrame?: (msg: Fluid2dFrameMsg) => void;
  onDone?: (msg: Fluid2dDoneMsg) => void;
  onError?: (detail: string) => void;
  // 接続が閉じた (正常終了・異常切断いずれも) ときに呼ばれる
  onClose?: () => void;
}

export class Fluid2dClient {
  private ws: WebSocket | null = null;
  private cb: Fluid2dClientCallbacks;

  constructor(cb: Fluid2dClientCallbacks) {
    this.cb = cb;
  }

  // コールバックを差し替える (continue で同じ接続・インスタンスを使い回しつつ、
  // 呼び出し側の状態管理を start 時と切り替えたい場合に使う)
  setCallbacks(cb: Fluid2dClientCallbacks): void {
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
      let msg: Fluid2dServerMessage;
      try {
        msg = JSON.parse(ev.data) as Fluid2dServerMessage;
      } catch (e) {
        this.cb.onError?.(`受信データの解析に失敗しました: ${String(e)}`);
        return;
      }
      switch (msg.type) {
        case "started":
          this.cb.onStarted?.(msg);
          break;
        case "frame":
          this.cb.onFrame?.(msg);
          break;
        case "done":
          this.cb.onDone?.(msg);
          break;
        case "error":
          this.cb.onError?.((msg as Fluid2dErrorMsg).detail);
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

  // 接続して start コマンドを送る (project は fluid2d 設定込みで渡すこと)
  start(project: Project): void {
    this.connect((ws) => {
      const cmd: Fluid2dClientCommand = { cmd: "start", project };
      ws.send(JSON.stringify(cmd));
    });
  }

  // 保持中のシミュレーション状態から追加実行する continue コマンドを送る (fluid1dClient と同じ設計)
  continueRun(opts: {
    extra_steps: number;
    frame_every?: number;
    avg_steps?: number | null;
    phase_bins?: number | null;
  }): void {
    const cmd: Fluid2dClientCommand = { cmd: "continue", ...opts };
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(cmd));
      return;
    }
    this.connect((ws) => ws.send(JSON.stringify(cmd)));
  }

  // stop コマンドを送る (接続していなければ何もしない)
  stop(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const cmd: Fluid2dClientCommand = { cmd: "stop" };
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
