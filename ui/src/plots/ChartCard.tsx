// 結果のグラフの枠: 見出し・右の操作 (切り替え・CSV など)・中身。

import type { ReactNode } from "react";

export function ChartCard({ title, tools, children, note }: { title: ReactNode; tools?: ReactNode; children: ReactNode; note?: ReactNode }) {
  return (
    <section className="chart-card">
      <header className="chart-card-head">
        <h3>{title}</h3>
        <span className="spacer" />
        {tools}
      </header>
      {children}
      {note && <div className="chart-note">{note}</div>}
    </section>
  );
}
