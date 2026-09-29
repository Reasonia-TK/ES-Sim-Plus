// 確認・情報のダイアログ (dialogs.ts の要求の列の先頭を表示する)。

import { AlertDialog, Dialog } from "radix-ui";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useConnection } from "../backend/connection";
import { parsePort } from "../backend/port";
import { useDialogs, type DialogRequest } from "./dialogs";
import { errorText, logError } from "./messages";

const UI_VERSION = "0.2.0";

function schemaStats(schema: Record<string, unknown> | undefined): { fields: number; defs: number } {
  const defs = (schema?.$defs ?? {}) as Record<string, { properties?: Record<string, unknown> }>;
  const root = (schema?.properties ?? {}) as Record<string, unknown>;
  let fields = Object.keys(root).length;
  for (const d of Object.values(defs)) fields += Object.keys(d.properties ?? {}).length;
  return { fields, defs: Object.keys(defs).length };
}

function AboutBody() {
  const { t } = useTranslation();
  const { status, info, schema } = useConnection();
  const stats = schemaStats(schema?.project);
  return (
    <div className="kv">
      <span>{t("dialog.uiVersion")}</span>
      <span>{UI_VERSION}</span>
      <span>{t("dialog.backendVersion")}</span>
      <span>{status === "connected" && info ? info.version : t("dialog.notConnected")}</span>
      <span>{t("dialog.device")}</span>
      <span>{info ? (info.gpu ? t("status.gpu") : t("status.cpu")) + (info.numba ? "" : ` · ${t("status.backendNumbaOff")}`) : "-"}</span>
      <span>{t("dialog.schemaFields")}</span>
      <span>{schema ? t("dialog.schemaFieldCount", { n: stats.fields, defs: stats.defs }) : "-"}</span>
    </div>
  );
}

function PortBody({ onDone }: { onDone: () => void }) {
  const { t } = useTranslation();
  const { port, changePort } = useConnection();
  const [text, setText] = useState(String(port));
  const valid = parsePort(text) !== null;
  return (
    <form
      onSubmit={async (e) => {
        e.preventDefault();
        const n = parsePort(text);
        if (n === null) return;
        try {
          await changePort(n);
        } catch (err) {
          logError(t("msg.source.backend"), errorText(err));
        }
        onDone();
      }}
    >
      <div className="field">
        <label className="field-label" htmlFor="port-input">
          {t("dialog.port")}
        </label>
        <div className="field-control">
          <input id="port-input" className="input" value={text} inputMode="numeric" autoFocus onChange={(e) => setText(e.target.value)} />
        </div>
        {!valid && <div className="field-error">{t("dialog.portInvalid")}</div>}
      </div>
      <p className="hint">{t("dialog.portHint")}</p>
      <p className="hint mono">{t("dialog.startCommand", { port: parsePort(text) ?? port })}</p>
      <div className="dialog-buttons">
        <button type="button" className="button" onClick={onDone}>
          {t("dialog.cancel")}
        </button>
        <button type="submit" className="button primary" disabled={!valid}>
          {t("dialog.apply")}
        </button>
      </div>
    </form>
  );
}

function Alert({ req, onClose }: { req: Extract<DialogRequest, { kind: "unsaved" | "confirm" | "recovery" }>; onClose: () => void }) {
  const { t, i18n } = useTranslation();
  // 閉じる経路は 1 つだけ (Radix の Cancel/Action は onOpenChange も呼ぶので使わない: 二重に閉じると次の要求まで消える)
  const done = (fn: () => void) => () => {
    fn();
    onClose();
  };
  const button = (label: string, cls: string, fn: () => void, autoFocus = false) => (
    <button type="button" className={`button ${cls}`} onClick={done(fn)} autoFocus={autoFocus}>
      {label}
    </button>
  );
  let title = "";
  let message = "";
  let buttons: React.ReactNode = null;
  let dismiss: (() => void) | null = null;
  if (req.kind === "unsaved") {
    title = t("dialog.unsavedTitle");
    message = t("dialog.unsavedMessage", { name: req.name });
    dismiss = done(() => req.resolve("cancel"));
    buttons = (
      <>
        {button(t("dialog.cancel"), "", () => req.resolve("cancel"))}
        {button(t("dialog.dontSave"), "danger", () => req.resolve("discard"))}
        {button(t("dialog.save"), "primary", () => req.resolve("save"), true)}
      </>
    );
  } else if (req.kind === "confirm") {
    title = req.title;
    message = req.message;
    dismiss = done(() => req.resolve(false));
    buttons = (
      <>
        {button(t("dialog.cancel"), "", () => req.resolve(false), true)}
        {button(req.okLabel ?? t("dialog.ok"), req.danger ? "danger" : "primary", () => req.resolve(true))}
      </>
    );
  } else {
    // 復元の確認は Esc では閉じない (破棄は明示的に選ばせる)
    title = t("dialog.recoveryTitle");
    const time = new Intl.DateTimeFormat(i18n.language, { dateStyle: "short", timeStyle: "medium" }).format(req.savedAt);
    message = t("dialog.recoveryMessage", { name: req.name, time });
    buttons = (
      <>
        {button(t("dialog.discard"), "danger", () => req.resolve(false))}
        {button(t("dialog.restore"), "primary", () => req.resolve(true), true)}
      </>
    );
  }
  return (
    <AlertDialog.Root open onOpenChange={(open) => !open && dismiss?.()}>
      <AlertDialog.Portal>
        <AlertDialog.Overlay className="dialog-overlay" />
        <AlertDialog.Content className="dialog-content">
          <AlertDialog.Title className="dialog-title">{title}</AlertDialog.Title>
          <AlertDialog.Description className="dialog-message">{message}</AlertDialog.Description>
          <div className="dialog-buttons">{buttons}</div>
        </AlertDialog.Content>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  );
}

export function DialogHost() {
  const { t } = useTranslation();
  const req = useDialogs((s) => s.queue[0]);
  const close = useDialogs((s) => s.close);
  if (!req) return null;
  if (req.kind === "about" || req.kind === "port") {
    const finish = () => {
      req.resolve();
      close();
    };
    return (
      <Dialog.Root open onOpenChange={(open) => !open && finish()}>
        <Dialog.Portal>
          <Dialog.Overlay className="dialog-overlay" />
          <Dialog.Content className="dialog-content">
            <Dialog.Title className="dialog-title">{req.kind === "about" ? t("dialog.aboutTitle") : t("dialog.portTitle")}</Dialog.Title>
            <Dialog.Description className="sr-only">{req.kind === "about" ? t("dialog.aboutTitle") : t("dialog.portTitle")}</Dialog.Description>
            {req.kind === "about" ? (
              <>
                <AboutBody />
                <div className="dialog-buttons">
                  <Dialog.Close className="button primary">{t("dialog.close")}</Dialog.Close>
                </div>
              </>
            ) : (
              <PortBody onDone={finish} />
            )}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    );
  }
  return <Alert req={req} onClose={close} />;
}
