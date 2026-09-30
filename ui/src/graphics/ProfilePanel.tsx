// ラインプロファイル: キャンバスで引いた線の上の V と |E| (200 点、2 軸、v1 と同じ /profile)。求解の結果があれば
// その計算に使った設定で取る (表示している解と同じ)、無ければ今の設定で。CSV は v1 と同じ列 (profile.csv)。
// 実行の結果を出しているときは、表示している場 (量・ライブ/時間平均/位相) をその場で線の上に取る。

import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { errorText, logError } from "../app/messages";
import { lineProfile, type ProfileResult } from "../backend/staticApi";
import { saveTextFile } from "../io/fileAccess";
import { useDocument } from "../model/documentStore";
import { LineChart, type LineSeries } from "../plots/LineChart";
import { usePrefs } from "../prefs/prefs";
import { columns, saveCsv as saveCsvFile } from "../plots/csv";
import { useStatic } from "../results/staticResults";
import { formatNumber, lengthUnitLabel, toDisplayLength } from "../util/format";
import { profileCsv } from "./exporting";
import { sampleLine } from "./scene";
import type { ActiveScene } from "./useScene";
import { useViewer } from "./viewerStore";

const V_COLOR = "#4da3ff";
const E_COLOR = "#ffb84d";

export function ProfilePanel({ active }: { active: ActiveScene }) {
  const { t } = useTranslation();
  const profile = useViewer((s) => s.profile);
  const setProfile = useViewer((s) => s.setProfile);
  const solve = useStatic((s) => s.solve);
  const unit = usePrefs((s) => s.lengthUnit);
  const [data, setData] = useState<ProfileResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 実行の結果 (粒子軌道は静電場の上に描くので除く)
  const runMode = Boolean(active.run && active.run.kind !== "trace");
  const runField = runMode ? active.scene.field : null;
  const sampled = useMemo(() => (profile && runField ? sampleLine(runField, profile[0], profile[1]) : null), [profile, runField]);

  useEffect(() => {
    if (!profile || runMode) {
      setData(null);
      return;
    }
    const project = solve?.project ?? useDocument.getState().project;
    const ctl = new AbortController();
    setBusy(true);
    setError(null);
    lineProfile(project, profile[0], profile[1], 200, ctl.signal)
      .then((r) => setData(r))
      .catch((e) => {
        if (e instanceof DOMException && e.name === "AbortError") return;
        setData(null);
        setError(errorText(e));
      })
      .finally(() => {
        if (!ctl.signal.aborted) setBusy(false);
      });
    return () => ctl.abort();
  }, [profile, solve, runMode]);

  const x = useMemo(() => (runMode ? (sampled?.s ?? []) : (data?.s ?? [])).map((s) => toDisplayLength(s, unit)), [runMode, sampled, data, unit]);
  const runLabel = runField ? `${runField.label} [${runField.unit}]` : "";
  const series = useMemo<LineSeries[]>(
    () =>
      runMode
        ? sampled
          ? [{ label: runLabel, values: sampled.v, color: V_COLOR }]
          : []
        : data
          ? [{ label: "V [V]", values: data.v, color: V_COLOR }, { label: "|E| [V/m]", values: data.e_abs, color: E_COLOR, right: true }]
          : [],
    [runMode, sampled, runLabel, data],
  );

  if (!profile) return null;
  const u = lengthUnitLabel(unit);
  const fmt = (p: [number, number]) => `(${formatNumber(toDisplayLength(p[0], unit))}, ${formatNumber(toDisplayLength(p[1], unit))})`;
  const saveCsv = async () => {
    if (runMode) {
      if (!sampled) return;
      const q = (active.controls.quantity ?? "value").replace(/[^A-Za-z0-9_]+/g, "");
      await saveCsvFile(`profile-${q}.csv`, ["s", q], columns(sampled.s, sampled.v));
      return;
    }
    if (!data) return;
    try {
      await saveTextFile("profile.csv", profileCsv(data), "csv", "CSV");
    } catch (e) {
      logError(t("msg.source.app"), t("viewer.exportFailed", { error: errorText(e) }));
    }
  };
  return (
    <section className="profile-panel" aria-label={t("profile.title")}>
      <header className="profile-header">
        <strong>{t("profile.title")}</strong>
        <span className="muted small">
          {fmt(profile[0])} → {fmt(profile[1])} {u}
        </span>
        {!runMode && busy && <span className="muted small">{t("profile.computing")}</span>}
        <span className="spacer" />
        <button type="button" className="button small" disabled={runMode ? !sampled : !data} onClick={() => void saveCsv()}>
          {t("profile.csv")}
        </button>
        <button type="button" className="button small" onClick={() => setProfile(null)}>
          {t("dialog.close")}
        </button>
      </header>
      {!runMode && error && <p className="hint hint-error">{error}</p>}
      {runMode && !runField && <p className="hint">{t("viewer.noField")}</p>}
      {runMode && sampled && <LineChart x={x} series={series} xLabel={`s [${u}]`} yLabel={runLabel} height={170} ariaLabel={t("profile.title")} />}
      {!runMode && data && (
        <LineChart x={x} series={series} xLabel={`s [${u}]`} yLabel="V [V]" yRightLabel="|E| [V/m]" height={170} ariaLabel={t("profile.title")} />
      )}
    </section>
  );
}
