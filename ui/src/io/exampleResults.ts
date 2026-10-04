// サンプルの保存された計算結果 (examples/results/<キー>.json.gz、prompts/135)。中身はビルドの資産になり、サンプルを
// 開いたときだけ読む (サンプルの設定と違い、起動時には読まない)。中身は「結果付きで保存」と同じ結果の束
// {version, meta, fluid2d?, pic?, ...} (results/bundle.ts の importResultsBundle で読み込む)。作り直しは
// backend/scripts/build_sample_results.py。

const urls = import.meta.glob("../../../examples/results/*.json.gz", {
  query: "?url",
  import: "default",
  eager: true,
}) as Record<string, string>;

/** サンプル (キー) の保存された結果の URL。無ければ undefined */
export function exampleResultsUrl(key: string): string | undefined {
  const hit = Object.entries(urls).find(([path]) => path.replace(/^.*\//, "") === `${key}.json.gz`);
  return hit?.[1];
}

/** 結果の束の生成の記録 (build_sample_results.py の meta) */
export interface ExampleResultsMeta {
  sample?: string;
  generated?: string;
  es_sim?: string;
  runs?: Record<string, { engine?: string; steps?: number; periods?: number | null; elapsed_s?: number }>;
}

/** gzip を展開して JSON を読む。配信側が展開済みで渡してきた (Content-Encoding: gzip) ときはそのまま読む */
export async function readGzipJson(bytes: Uint8Array): Promise<unknown> {
  let text: string;
  if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Response(new Uint8Array(bytes)).body!.pipeThrough(new DecompressionStream("gzip"));
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(bytes);
  }
  return JSON.parse(text);
}

/** 保存された結果の束を読む */
export async function loadExampleResults(url: string): Promise<Record<string, unknown>> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await readGzipJson(new Uint8Array(await res.arrayBuffer()));
  if (!data || typeof data !== "object") throw new Error("not a results bundle");
  return data as Record<string, unknown>;
}
