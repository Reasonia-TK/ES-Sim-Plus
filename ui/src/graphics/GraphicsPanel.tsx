// グラフィックス欄: 2D ビューア (ジオメトリ・メッシュ・場・作図) とその下のラインプロファイル。
// P6e で結果のビュー (1D のグラフなど) をタブで足す。

import { ProfilePanel } from "./ProfilePanel";
import { Viewer } from "./Viewer";

export function GraphicsPanel() {
  return (
    <div className="graphics-panel">
      <Viewer />
      <ProfilePanel />
    </div>
  );
}
