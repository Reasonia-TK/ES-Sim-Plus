// ステータスバー: 直近のメッセージ (無ければ「準備完了」)・未保存の印・長さの単位・バックエンドの状態
// (クリックで接続の設定)。

import { useTranslation } from "react-i18next";
import { useConnection } from "../backend/connection";
import { useIsDirty } from "../model/documentStore";
import { usePrefs } from "../prefs/prefs";
import { lengthUnitLabel } from "../util/format";
import { showPortDialog } from "./dialogs";
import { useMessages } from "./messages";

export function BackendBadge() {
  const { t } = useTranslation();
  const { status, info, port } = useConnection();
  let text: string;
  let tone: string;
  if (status === "connected" && info) {
    text = t("status.backendConnected", { version: info.version, device: info.gpu ? t("status.gpu") : t("status.cpu") });
    tone = "ok";
    if (!info.numba) {
      text += ` · ${t("status.backendNumbaOff")}`;
      tone = "warn";
    }
  } else if (status === "connecting") {
    text = t("status.backendConnecting");
    tone = "muted";
  } else {
    text = t("status.backendDisconnected", { port });
    tone = "error";
  }
  return (
    <button type="button" className={`status-badge badge-${tone}`} onClick={() => void showPortDialog()} title={t("menu.backendPort")}>
      {text}
    </button>
  );
}

export function StatusBar() {
  const { t } = useTranslation();
  const last = useMessages((s) => s.items[s.items.length - 1]);
  const dirty = useIsDirty();
  const unit = usePrefs((s) => s.lengthUnit);
  return (
    <footer className="status-bar">
      <span className={`status-text${last ? ` message-${last.level}` : ""}`}>{last ? last.text : t("app.ready")}</span>
      <span className="spacer" />
      {dirty && <span className="status-item text-warn">● {t("app.unsavedMark")}</span>}
      <span className="status-item">{lengthUnitLabel(unit)}</span>
      <BackendBadge />
    </footer>
  );
}
