// REST の呼び出し (エラーは「{path} に失敗: {detail}」の形の例外)。

import { baseUrl } from "./port";

export class ApiError extends Error {
  constructor(
    readonly path: string,
    readonly status: number,
    readonly detail: string,
  ) {
    super(`${path}: ${detail}`);
  }
}

async function detailOf(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const d = (body as { detail?: unknown }).detail;
    if (typeof d === "string") return d;
    if (d !== undefined) return JSON.stringify(d);
  } catch {
    // JSON でない応答
  }
  return `HTTP ${res.status}`;
}

export async function apiGet<T>(path: string, opts?: { signal?: AbortSignal; timeoutMs?: number }): Promise<T> {
  const ctl = new AbortController();
  const timer = opts?.timeoutMs ? setTimeout(() => ctl.abort(), opts.timeoutMs) : null;
  opts?.signal?.addEventListener("abort", () => ctl.abort());
  try {
    const res = await fetch(baseUrl() + path, { signal: ctl.signal });
    if (!res.ok) throw new ApiError(path, res.status, await detailOf(res));
    return (await res.json()) as T;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** POST して応答をテキストで受ける (DXF の書き出しなど) */
export async function apiPostText(path: string, body: unknown): Promise<string> {
  const res = await fetch(baseUrl() + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new ApiError(path, res.status, await detailOf(res));
  return await res.text();
}

export async function apiPost<T>(path: string, body: unknown, opts?: { signal?: AbortSignal }): Promise<T> {
  const res = await fetch(baseUrl() + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: opts?.signal,
  });
  if (!res.ok) throw new ApiError(path, res.status, await detailOf(res));
  return (await res.json()) as T;
}
