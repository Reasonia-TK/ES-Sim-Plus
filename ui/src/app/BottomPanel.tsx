// 下部パネル: メッセージ (レベルで絞れる)・進捗 (実行中・待ちのジョブ)・ジョブ (全部)。

import { Tabs } from "radix-ui";
import { useEffect, useRef, useState } from "react";
import { create } from "zustand";
import { JobsView, ProgressView } from "../jobs/JobsPanels";
import { useTranslation } from "react-i18next";
import { useMessages, type MessageLevel } from "./messages";

export type BottomTab = "messages" | "progress" | "jobs";

/** 開いているタブ (ステータスバーから「進捗」を開く) */
export const useBottomTab = create<{ tab: BottomTab; setTab: (t: BottomTab) => void }>()((set) => ({ tab: "messages", setTab: (tab) => set({ tab }) }));

function MessagesView() {
  const { t, i18n } = useTranslation();
  const items = useMessages((s) => s.items);
  const clear = useMessages((s) => s.clear);
  const [level, setLevel] = useState<MessageLevel | "all">("all");
  const listRef = useRef<HTMLDivElement>(null);
  const shown = level === "all" ? items : items.filter((m) => m.level === level);
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown.length]);
  const fmt = new Intl.DateTimeFormat(i18n.language, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const levels: (MessageLevel | "all")[] = ["all", "info", "warning", "error"];
  const levelLabel = { all: t("bottom.levelAll"), info: t("bottom.levelInfo"), warning: t("bottom.levelWarning"), error: t("bottom.levelError") };
  return (
    <div className="messages">
      <div className="messages-toolbar">
        {levels.map((l) => (
          <button key={l} type="button" className={`chip${level === l ? " active" : ""}`} onClick={() => setLevel(l)}>
            {levelLabel[l]}
            {l !== "all" && <span className="chip-count">{items.filter((m) => m.level === l).length}</span>}
          </button>
        ))}
        <span className="spacer" />
        <button type="button" className="button small" onClick={clear} disabled={items.length === 0}>
          {t("bottom.clear")}
        </button>
      </div>
      <div ref={listRef} className="messages-list" role="log" aria-live="polite">
        {shown.length === 0 && <div className="muted pad">{t("bottom.empty")}</div>}
        {shown.map((m) => (
          <div key={m.id} className={`message message-${m.level}`}>
            <span className="message-time">{fmt.format(m.time)}</span>
            <span className="message-source">{m.source}</span>
            <span className="message-text">{m.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function BottomPanel() {
  const { t } = useTranslation();
  const tab = useBottomTab((s) => s.tab);
  const setTab = useBottomTab((s) => s.setTab);
  return (
    <Tabs.Root className="bottom-panel" value={tab} onValueChange={(v) => setTab(v as BottomTab)}>
      <Tabs.List className="tabs-list" aria-label={t("bottom.messages")}>
        <Tabs.Trigger className="tabs-trigger" value="messages">
          {t("bottom.messages")}
        </Tabs.Trigger>
        <Tabs.Trigger className="tabs-trigger" value="progress">
          {t("bottom.progress")}
        </Tabs.Trigger>
        <Tabs.Trigger className="tabs-trigger" value="jobs">
          {t("bottom.jobs")}
        </Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content className="tabs-content" value="messages">
        <MessagesView />
      </Tabs.Content>
      <Tabs.Content className="tabs-content" value="progress">
        <ProgressView />
      </Tabs.Content>
      <Tabs.Content className="tabs-content" value="jobs">
        <JobsView />
      </Tabs.Content>
    </Tabs.Root>
  );
}
