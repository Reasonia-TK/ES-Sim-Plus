// バックエンドとの接続状態。/health を一定間隔で確かめ、つながったらスキーマ (/v2/schema) を取り直す。
// v1 は WebSocket が自動で再接続しなかった。v2 のイベント接続 (P6d) もこの状態を見て張り直す。

import { create } from "zustand";
import { apiGet } from "./api";
import { getPort, initPort, setPort } from "./port";

export const HEALTH_INTERVAL_MS = 5000;

export interface BackendInfo {
  version: string;
  gpu: boolean;
  numba: boolean;
  /** サーバープロセスの識別子 (変われば再起動した。古いバックエンドは返さない) */
  instance?: string;
  v2?: Record<string, unknown>;
}

export interface ProjectSchema {
  version: string;
  project: Record<string, unknown>;
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

interface ConnectionState {
  status: ConnectionStatus;
  port: number;
  info: BackendInfo | null;
  schema: ProjectSchema | null;
  lastError: string | null;
  /** 今すぐ確かめる */
  check: () => Promise<void>;
  changePort: (port: number) => Promise<void>;
}

export function parseHealth(body: unknown): BackendInfo {
  const b = (body ?? {}) as Record<string, unknown>;
  if (b.status !== "ok") throw new Error("/health の応答が不正です");
  return {
    version: String(b.version ?? "?"),
    gpu: Boolean(b.gpu),
    numba: b.numba === undefined ? true : Boolean(b.numba),
    instance: typeof b.instance === "string" ? b.instance : undefined,
    v2: (b.v2 as Record<string, unknown> | undefined) ?? undefined,
  };
}

let inflight: Promise<void> | null = null;

export const useConnection = create<ConnectionState>()((set, get) => ({
  status: "connecting",
  port: getPort(),
  info: null,
  schema: null,
  lastError: null,

  check: () => {
    if (inflight) return inflight;
    inflight = (async () => {
      try {
        const info = parseHealth(await apiGet("/health", { timeoutMs: 3000 }));
        const prev = get().info;
        const wasConnected = get().status === "connected";
        // 版が同じでも再起動 (instance が変わった) ならスキーマが変わっているかもしれない
        const restarted = prev?.version !== info.version || prev?.instance !== info.instance;
        set({ status: "connected", info, lastError: null, port: getPort() });
        if (!wasConnected || restarted || get().schema === null) {
          try {
            set({ schema: await apiGet<ProjectSchema>("/v2/schema", { timeoutMs: 10000 }) });
          } catch (e) {
            // 古いバックエンド (/v2/schema 無し) でも接続自体は続ける
            set({ schema: null, lastError: e instanceof Error ? e.message : String(e) });
          }
        }
      } catch (e) {
        set({ status: "disconnected", lastError: e instanceof Error ? e.message : String(e), port: getPort() });
      } finally {
        inflight = null;
      }
    })();
    return inflight;
  },

  changePort: async (port) => {
    await setPort(port);
    set({ port, status: "connecting", info: null, schema: null });
    await get().check();
  },
}));

let timer: ReturnType<typeof setInterval> | null = null;

/** 起動時に 1 回呼ぶ: ポートを読み、すぐに確かめ、以後は一定間隔で確かめる */
export async function startConnection(): Promise<void> {
  const port = await initPort();
  useConnection.setState({ port });
  await useConnection.getState().check();
  if (timer === null) {
    timer = setInterval(() => void useConnection.getState().check(), HEALTH_INTERVAL_MS);
  }
}

export function stopConnection(): void {
  if (timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}
