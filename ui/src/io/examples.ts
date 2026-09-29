// リポジトリの examples/*.json をビルド時に取り込む (メニューの「サンプル」)。

const modules = import.meta.glob("../../../examples/*.json", { eager: true, import: "default" }) as Record<
  string,
  unknown
>;

export interface Example {
  /** ファイル名の拡張子なし (翻訳キー examples.<key>) */
  key: string;
  data: unknown;
}

/** 表示順 (無いものは名前順で後ろに) */
const ORDER = ["parallel_plates", "coaxial", "ccp_demo", "egun_rz", "fn_diode"];

export const EXAMPLES: Example[] = Object.entries(modules)
  .map(([path, data]) => ({ key: path.replace(/^.*\//, "").replace(/\.json$/, ""), data }))
  .sort((a, b) => {
    const ia = ORDER.indexOf(a.key);
    const ib = ORDER.indexOf(b.key);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.key.localeCompare(b.key);
  });

export function findExample(key: string): Example | undefined {
  return EXAMPLES.find((e) => e.key === key);
}
