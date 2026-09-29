// 翻訳キーを型で検査する (存在しないキーはコンパイルエラー)
import "i18next";
import type { Resources } from "./ja";

declare module "i18next" {
  interface CustomTypeOptions {
    defaultNS: "translation";
    resources: { translation: Resources };
  }
}
