// boltzpm による E/N 掃引 → LMEA 係数テーブル生成の WebSocket クライアント (薄いラッパ、prompts/118)。
// tlClient.ts と同じ流儀 (continue は無い。backend/es_sim/boltz.py の1回性の生成という設計に
// 合わせて server.py 側も start/stop のみの最も単純な配線になっている、server.py コメント参照)。
// ws://127.0.0.1:<port>/ws/boltz に接続し、project+module ("fluid1d"/"fluid2d") を送ることで
// backend 側がその設定 (electron_processes/ion_mass_amu/gas_pressure_pa/gas_temperature_k) を
// 取り出して掃引する (server.py _run_boltz_session 参照)。

import type {
  BoltzClientCommand,
  BoltzDoneMsg,
  BoltzErrorMsg,
  BoltzOpts,
  BoltzProgressMsg,
  BoltzServerMessage,
  BoltzStartedMsg,
  Project,
} from "./types";
import { getPort } from "./backendPort";

// 接続の都度ポート番号を組み立てる (GUIでの変更を即座に反映するため、定数 URL は使わない)
function wsUrl(): string {
  return `ws://127.0.0.1:${getPort()}/ws/boltz`;
}

export interface BoltzClientCallbacks {
  onStarted?: (msg: BoltzStartedMsg) => void;
  onProgress?: (msg: BoltzProgressMsg) => void;
  onDone?: (msg: BoltzDoneMsg) => void;
  onError?: (detail: string) => void;
  // 接続が閉じた (正常終了・異常切断いずれも) ときに呼ばれる
  onClose?: () => void;
}

export class BoltzClient {
  private ws: WebSocket | null = null;
  private cb: BoltzClientCallbacks;

  constructor(cb: BoltzClientCallbacks) {
    this.cb = cb;
  }

  setCallbacks(cb: BoltzClientCallbacks): void {
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
      let msg: BoltzServerMessage;
      try {
        msg = JSON.parse(ev.data) as BoltzServerMessage;
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
          this.cb.onError?.((msg as BoltzErrorMsg).detail);
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

  // 接続して start コマンドを送る (project は module 側の fluid1d/fluid2d 設定込みで渡すこと。
  // opts省略時は backend 既定 (DEFAULT_BOLTZ_OPTS) が使われる)
  start(project: Project, module: "fluid1d" | "fluid2d", opts?: BoltzOpts): void {
    this.connect((ws) => {
      const cmd: BoltzClientCommand = { cmd: "start", project, module, opts };
      ws.send(JSON.stringify(cmd));
    });
  }

  // stop コマンドを送る (接続していなければ何もしない)。中断時点までの収束済み点で
  // テーブルが返る (server.py 参照)
  stop(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const cmd: BoltzClientCommand = { cmd: "stop" };
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
