// アプリのシェル: メニューバー / (ツリー | 設定 | グラフィックス) / 下部パネル / ステータスバー。
// ペインはドラッグで大きさを変えられ、配置は localStorage に保存する (表示 > レイアウトを初期化 で戻す)。

import { useEffect } from "react";
import { Group, Panel, Separator, useDefaultLayout } from "react-resizable-panels";
import { useTranslation } from "react-i18next";
import { useConnection } from "../backend/connection";
import { GraphicsPanel } from "../graphics/GraphicsPanel";
import { documentName, useDocument, useIsDirty } from "../model/documentStore";
import { SettingsPanel } from "../pages/SettingsPanel";
import { usePrefs } from "../prefs/prefs";
import { usePlaybackDriver } from "../graphics/PlaybackBar";
import { useForgetRemovedRun, useWatchActiveRun } from "../results/runData";
import { ModelTree } from "../tree/ModelTree";
import { isTauri } from "../util/env";
import { BottomPanel } from "./BottomPanel";
import { DialogHost } from "./DialogHost";
import { LAYOUT_IDS, useLayoutEpoch } from "./layout";
import { MenuBar } from "./MenuBar";
import { logInfo, logWarning } from "./messages";
import { useShortcuts } from "./shortcuts";
import { StatusBar } from "./StatusBar";

function useDocumentChrome(): void {
  const { t } = useTranslation();
  const theme = usePrefs((s) => s.theme);
  const language = usePrefs((s) => s.language);
  const dirty = useIsDirty();
  const name = useDocument((s) => documentName(s, t("app.untitled")));
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => {
    document.documentElement.lang = language;
  }, [language]);
  useEffect(() => {
    const title = `${dirty ? "● " : ""}${name} — ES-Sim`;
    document.title = title;
    if (isTauri()) {
      void import("@tauri-apps/api/window").then(({ getCurrentWindow }) => getCurrentWindow().setTitle(title)).catch(() => {});
    }
  }, [dirty, name]);
}

/** 接続・切断をメッセージに残す */
function useConnectionLog(): void {
  const { t } = useTranslation();
  useEffect(
    () =>
      useConnection.subscribe((s, prev) => {
        if (s.status === prev.status) return;
        const src = t("msg.source.backend");
        if (s.status === "connected" && s.info) {
          logInfo(src, t("msg.connected", { version: s.info.version, device: s.info.gpu ? t("status.gpu") : t("status.cpu") }));
        } else if (s.status === "disconnected" && prev.status === "connected") {
          logWarning(src, t("msg.disconnected"));
        }
      }),
    [t],
  );
}

function Workspace() {
  const main = useDefaultLayout({ id: LAYOUT_IDS[0], storage: localStorage });
  const cols = useDefaultLayout({ id: LAYOUT_IDS[1], storage: localStorage });
  return (
    <Group
      id={LAYOUT_IDS[0]}
      orientation="vertical"
      className="workspace"
      defaultLayout={main.defaultLayout}
      onLayoutChanged={main.onLayoutChanged}
    >
      <Panel id="top" minSize="30">
        <Group id={LAYOUT_IDS[1]} orientation="horizontal" defaultLayout={cols.defaultLayout} onLayoutChanged={cols.onLayoutChanged}>
          <Panel id="tree" defaultSize={260} minSize={160} collapsible collapsedSize={0}>
            <ModelTree />
          </Panel>
          <Separator className="separator" />
          <Panel id="settings" defaultSize={360} minSize={240}>
            <SettingsPanel />
          </Panel>
          <Separator className="separator" />
          <Panel id="graphics" minSize="20">
            <GraphicsPanel />
          </Panel>
        </Group>
      </Panel>
      <Separator className="separator" />
      <Panel id="bottom" defaultSize={170} minSize={60} collapsible collapsedSize={0}>
        <BottomPanel />
      </Panel>
    </Group>
  );
}

export function App() {
  useShortcuts();
  useDocumentChrome();
  useConnectionLog();
  useWatchActiveRun();
  useForgetRemovedRun();
  usePlaybackDriver();
  const epoch = useLayoutEpoch((s) => s.epoch);
  return (
    <div className="app">
      <MenuBar />
      <Workspace key={epoch} />
      <StatusBar />
      <DialogHost />
    </div>
  );
}
