// 電子の係数モデル (Maxwell / Boltzmann) と boltzpm の係数表の情報・鮮度 (流体 1D・2D)。
// 表の生成は boltzpm のジョブ (BoltzGenerate)。

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SchemaField } from "../../forms/SchemaField";
import { useDocument } from "../../model/documentStore";
import { getIn, type Path } from "../../schema/schema";
import { computeProcessesHash, type XsProcessLike } from "../../util/boltzHash";
import { formatNumber } from "../../util/format";
import { BoltzGenerate } from "./BoltzGenerate";
import { Hint } from "./common";

interface BoltzTable {
  en_td: number[];
  mean_energy_ev: number[];
  source_hash: string;
  opts?: Record<string, unknown>;
  warnings?: string[];
}

function range(a: number[]): string {
  return a.length ? `${formatNumber(Math.min(...a))} – ${formatNumber(Math.max(...a))}` : "-";
}

export function BoltzSection({ path }: { path: Path }) {
  const { t } = useTranslation();
  const model = useDocument((s) => getIn(s.project, [...path, "electron_model"])) as string | undefined;
  const table = useDocument((s) => getIn(s.project, [...path, "boltz_table"])) as BoltzTable | null | undefined;
  const processes = (useDocument((s) => getIn(s.project, [...path, "electron_processes"])) as XsProcessLike[] | undefined) ?? [];
  const [hash, setHash] = useState<string | null>(null);
  const key = JSON.stringify(processes);
  useEffect(() => {
    let alive = true;
    if (processes.length === 0) {
      setHash(null);
      return;
    }
    void computeProcessesHash(processes).then((h) => alive && setHash(h));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const stale = table && hash !== null && table.source_hash !== hash;
  const label = t("widgets.boltzTable");
  return (
    <div className="subsection">
      <SchemaField path={[...path, "electron_model"]} />
      {model === "boltzmann" && !table && <Hint tone="warn">{t("widgets.boltzNoTable")}</Hint>}
      {table ? (
        <>
          <div className="kv">
            <span>{t("widgets.boltzPoints")}</span>
            <span>{table.en_td.length}</span>
            <span>ε̄ [eV]</span>
            <span>{range(table.mean_energy_ev)}</span>
            <span>E/N [Td]</span>
            <span>{range(table.en_td)}</span>
            {table.opts?.boltzpm_version !== undefined && (
              <>
                <span>boltzpm</span>
                <span>{String(table.opts.boltzpm_version)}</span>
              </>
            )}
          </div>
          {stale && <Hint tone="error">{t("widgets.boltzStale")}</Hint>}
          {processes.length === 0 && <Hint>{t("widgets.boltzDefaultXs")}</Hint>}
          {(table.warnings ?? []).map((w, i) => (
            <Hint key={i} tone="warn">
              {w}
            </Hint>
          ))}
          {/* 表を消すと Maxwell に戻す (v1 と同じ) */}
          <button
            type="button"
            className="button small danger"
            onClick={() =>
              useDocument.getState().update(label, (d) => {
                const blk = getIn(d, path) as Record<string, unknown>;
                blk.boltz_table = null;
                blk.electron_model = "maxwell";
              })
            }
          >
            {t("widgets.boltzDelete")}
          </button>
        </>
      ) : null}
      {(path[0] === "fluid1d" || path[0] === "fluid2d") && <BoltzGenerate module={path[0]} />}
    </div>
  );
}
