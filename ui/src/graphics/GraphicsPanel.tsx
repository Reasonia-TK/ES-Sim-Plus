// グラフィックス欄 (タブで複数のビュー。P6a はジオメトリの表示のみ、場・プロット・CAD は P6c 以降)

import { Tabs } from "radix-ui";
import { useTranslation } from "react-i18next";
import { GeometryView } from "./GeometryView";

export function GraphicsPanel() {
  const { t } = useTranslation();
  return (
    <Tabs.Root className="graphics-panel" defaultValue="geometry">
      <Tabs.List className="tabs-list" aria-label={t("graphics.geometry")}>
        <Tabs.Trigger className="tabs-trigger" value="geometry">
          {t("graphics.geometry")}
        </Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content className="tabs-content" value="geometry">
        <GeometryView />
      </Tabs.Content>
    </Tabs.Root>
  );
}
