import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { usePrefs } from "../prefs/prefs";
import en from "./en";
import { fieldsEn } from "./fields.en";
import { fieldsJa } from "./fields.ja";
import ja from "./ja";

// 設定項目の表示名 (fields/enums/hints) はキーが動的なので型付きのキーとは別に足す
void i18n.use(initReactI18next).init({
  resources: {
    ja: { translation: { ...ja, ...fieldsJa } },
    en: { translation: { ...en, ...fieldsEn } },
  },
  lng: usePrefs.getState().language,
  fallbackLng: "ja",
  interpolation: { escapeValue: false },
});

// 言語の設定を変えたら切り替える
usePrefs.subscribe((s, prev) => {
  if (s.language !== prev.language) void i18n.changeLanguage(s.language);
});

export default i18n;
export const t = i18n.t.bind(i18n);
