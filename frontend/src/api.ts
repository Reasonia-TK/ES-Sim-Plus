import type { DsmcResult, Health, MeshResult, Pic1dSettings, Point, Project, ProfileResult, SolveResult, TraceResult, XsProcess } from "./types";
import { getPort } from "./backendPort";

// リクエストの都度ポート番号を組み立てる (GUIでの変更を即座に反映するため、定数 BASE は使わない)
function base(): string {
  return `http://127.0.0.1:${getPort()}`;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${base()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(`${path} failed: ${JSON.stringify(detail)}`);
  }
  return res.json();
}

export const api = {
  health: (): Promise<Health> =>
    fetch(`${base()}/health`).then((r) => r.json()),
  mesh: (project: Project): Promise<MeshResult> => post("/mesh", project),
  solve: (project: Project): Promise<SolveResult> => post("/solve", project),
  profile: (project: Project, p1: Point, p2: Point, n = 200): Promise<ProfileResult> =>
    post("/profile", { project, p1, p2, n }),
  trace: (project: Project): Promise<TraceResult> => post("/trace", project),
  dsmc: (project: Project): Promise<DsmcResult> => post("/dsmc", project),
  // LXCat形式テキストを断面積プロセス列にパースする (MCC設定のインポート用)
  lxcatParse: (text: string, species: "electron" | "ion"): Promise<{ processes: XsProcess[]; warnings: string[] }> =>
    post("/lxcat/parse", { text, species }),
  // 1D PIC/MCC のベンチマークプリセット一覧 (prompts/91)。project 不要な GET のみの単純なエンドポイント
  pic1dPresets: async (): Promise<
    Record<string, { label: string; description: string; pic1d: Pic1dSettings; note?: string }>
  > => {
    const res = await fetch(`${base()}/pic1d/presets`);
    if (!res.ok) {
      // 旧バックエンド (v0.1.33 未満) には本 endpoint が無く 404 になる。res.ok を見ずに
      // r.json() すると、エラー系 JSON (detail等) を presets として扱ってしまうため注意
      if (res.status === 404) {
        throw new Error("バックエンドが古い可能性があります (上部バーの backend バージョンを確認してください)");
      }
      const detail = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(`/pic1d/presets failed: ${JSON.stringify(detail)}`);
    }
    return res.json();
  },
  // パラメータスイープ (prompts/79): ケース i の結果付きJSONを取得する (loadProject と
  // 同じ形式。project 本体 + results)。未完了/失敗は 404 (呼び出し側で catch すること)
  sweepResult: async (i: number): Promise<unknown> => {
    const res = await fetch(`${base()}/sweep/result/${i}`);
    if (!res.ok) {
      const detail = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(`/sweep/result/${i} failed: ${JSON.stringify(detail)}`);
    }
    return res.json();
  },
};
