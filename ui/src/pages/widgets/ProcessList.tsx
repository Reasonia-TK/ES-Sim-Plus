// 衝突断面積のプロセス一覧と LXCat の取り込み (POST /lxcat/parse、取り込むと一覧を置き換える。v1 と同じ)。

import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { apiPost } from "../../backend/api";
import { useConnection } from "../../backend/connection";
import { setValue } from "../../forms/useField";
import { useDocument } from "../../model/documentStore";
import { getIn, type Path } from "../../schema/schema";
import { formatNumber } from "../../util/format";
import { Hint } from "./common";

interface Process {
  kind: string;
  label: string;
  threshold_ev: number;
  energy_ev: number[];
}

export function ProcessList({ path, species, emptyHint }: { path: Path; species: "electron" | "ion"; emptyHint?: string }) {
  const { t } = useTranslation();
  const list = (useDocument((s) => getIn(s.project, path)) as Process[] | undefined) ?? [];
  const connected = useConnection((s) => s.status === "connected");
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const label = species === "electron" ? t("widgets.electronProcesses") : t("widgets.ionProcesses");

  const importFile = async (f: File) => {
    setBusy(true);
    setError(null);
    try {
      const res = await apiPost<{ processes: Process[]; warnings: string[] }>("/lxcat/parse", { text: await f.text(), species });
      setValue(path, res.processes, label);
      setWarnings(res.warnings ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="subsection">
      <div className="subsection-title">
        {label} <span className="muted">({list.length})</span>
      </div>
      <input
        ref={input}
        type="file"
        accept=".txt,text/plain"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) void importFile(f);
        }}
      />
      <div className="button-row tight">
        <button type="button" className="button small" disabled={!connected || busy} onClick={() => input.current?.click()} title={connected ? undefined : t("widgets.needBackend")}>
          {busy ? t("widgets.importing") : t("widgets.lxcatImport")}
        </button>
        {list.length > 0 && (
          <button type="button" className="button small danger" onClick={() => setValue(path, [], label)}>
            {t("widgets.clearList")}
          </button>
        )}
      </div>
      {error && <Hint tone="error">{error}</Hint>}
      {warnings.map((w, i) => (
        <Hint key={i} tone="warn">
          {w}
        </Hint>
      ))}
      {list.length === 0 ? (
        emptyHint && <Hint>{emptyHint}</Hint>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>{t("widgets.processKind")}</th>
              <th>{t("widgets.processLabel")}</th>
              <th>{t("widgets.threshold")}</th>
              <th>{t("widgets.points")}</th>
            </tr>
          </thead>
          <tbody>
            {list.map((p, i) => (
              <tr key={i}>
                <td>{p.kind}</td>
                <td className="ellipsis" title={p.label}>
                  {p.label.length > 28 ? `${p.label.slice(0, 27)}…` : p.label}
                </td>
                <td>{p.threshold_ev > 0 ? `${formatNumber(p.threshold_ev)} eV` : "-"}</td>
                <td>{p.energy_ev.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
