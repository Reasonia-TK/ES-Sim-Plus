// グラフィックス欄: 2D ビュー (ジオメトリ・メッシュ・場・作図とラインプロファイル) と結果のグラフのタブ。
// 2D で場を出さない実行 (1D・定在波・係数の表・スイープ) を選んだらグラフのタブに切り替え、そこから 2D の実行に
// 移ったら 2D ビューに戻す (自分でグラフのタブを選んでいたらそのまま)。

import { Tabs } from "radix-ui";
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { ResultsCharts } from "../results/ResultsCharts";
import { useResultsView, type GraphicsTab } from "../results/resultsView";
import { useActiveJob } from "../results/runData";
import { ProfilePanel } from "./ProfilePanel";
import { useActiveScene } from "./useScene";
import { VIEWER_KINDS } from "./runScene";
import { Viewer } from "./Viewer";

/** 2D ビュー: ビューアとラインプロファイルが同じ Scene (静電場か実行の結果) を使う */
function ViewTab() {
  const active = useActiveScene();
  return (
    <>
      <Viewer active={active} />
      <ProfilePanel active={active} />
    </>
  );
}

export function GraphicsPanel() {
  const { t } = useTranslation();
  const tab = useResultsView((s) => s.graphicsTab);
  const setTab = useResultsView((s) => s.setGraphicsTab);
  const job = useActiveJob();
  const id = job?.id;
  const chartsOnly = job ? !VIEWER_KINDS.includes(job.kind) : false;
  const forced = useRef(false);
  useEffect(() => {
    if (!id) return;
    const rv = useResultsView.getState();
    if (chartsOnly) {
      forced.current = rv.graphicsTab !== "charts" || forced.current;
      rv.setGraphicsTab("charts");
    } else if (forced.current) {
      forced.current = false;
      rv.setGraphicsTab("view");
    }
  }, [id, chartsOnly]);
  return (
    <Tabs.Root className="graphics-panel" value={tab} onValueChange={(v) => setTab(v as GraphicsTab)}>
      <Tabs.List className="tabs-list graphics-tabs" aria-label={t("results.graphicsTabs")}>
        <Tabs.Trigger className="tabs-trigger" value="view">
          {t("results.tabView")}
        </Tabs.Trigger>
        <Tabs.Trigger className="tabs-trigger" value="charts">
          {t("results.tabCharts")}
        </Tabs.Trigger>
      </Tabs.List>
      {/* ビューアは隠しても残す (WebGL の資源とカメラを保つ) */}
      <Tabs.Content className="tabs-content graphics-view" value="view" forceMount>
        <ViewTab />
      </Tabs.Content>
      <Tabs.Content className="tabs-content graphics-charts" value="charts">
        <ResultsCharts />
      </Tabs.Content>
    </Tabs.Root>
  );
}
