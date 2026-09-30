// 設定のページを切り替えても残しておく一時的な値 (保存しない、元に戻す対象外): 区分の開閉・DSMC の結果の選択・
// 係数の表づくりの条件・プリセットの選択・取り込みの警告など。v1 はページを表示したまま隠していたので残っていた。

import { create } from "zustand";

interface PageState {
  values: Record<string, unknown>;
  setValue: (key: string, value: unknown) => void;
}

export const usePageState = create<PageState>()((set) => ({
  values: {},
  setValue: (key, value) => set((s) => (s.values[key] === value ? {} : { values: { ...s.values, [key]: value } })),
}));

/** ページをまたいで残す値 (未設定なら既定値) */
export function usePageValue<T>(key: string, def: T): [T, (v: T) => void] {
  const v = usePageState((s) => s.values[key]) as T | undefined;
  return [v === undefined ? def : v, (x: T) => usePageState.getState().setValue(key, x)];
}
