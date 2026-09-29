// boltzpm で係数の表を作る (v1 の BoltzSection の生成と同じ条件。条件は文書に保存しない)。できた表は
// ジョブの完了で流体の設定の boltz_table に入る (jobs/effects.ts)。

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { DEFAULT_BOLTZ_OPTS } from "../../schema/defaults";
import { formatNumber, parseNumber } from "../../util/format";
import { CommitText, Field } from "../inputs";
import { Section } from "../../forms/blocks";
import { RunControls } from "./RunControls";

type Opts = typeof DEFAULT_BOLTZ_OPTS;

const FIELDS: { key: keyof Opts; unit?: string; integer?: boolean; nullable?: boolean; min: number }[] = [
  { key: "en_min_td", unit: "Td", min: 1e-6 },
  { key: "en_max_td", unit: "Td", min: 1e-6 },
  { key: "n_points", integer: true, min: 2 },
  { key: "eps_max_ev", unit: "eV", nullable: true, min: 1e-6 },
  { key: "d_eps_ev", unit: "eV", min: 1e-6 },
  { key: "n_theta", integer: true, min: 2 },
];

export function BoltzGenerate({ module }: { module: "fluid1d" | "fluid2d" }) {
  const { t } = useTranslation();
  const [opts, setOpts] = useState<Opts>({ ...DEFAULT_BOLTZ_OPTS });
  const bad = !(opts.en_max_td >= opts.en_min_td);
  return (
    <Section title={t("jobs.boltzTitle")} defaultOpen={false}>
      {FIELDS.map((f) => (
        <Field key={f.key} label={t(`jobs.boltzOpt.${f.key}`)} unit={f.unit}>
          {(id, onError) => (
            <CommitText
              id={id}
              inputMode="decimal"
              value={opts[f.key] === null ? "" : formatNumber(opts[f.key] as number)}
              placeholder={f.nullable ? t("input.auto") : undefined}
              onError={onError}
              validate={(s) => {
                if (s.trim() === "") return f.nullable ? null : t("input.required");
                const v = parseNumber(s);
                if (v === null) return t("input.notNumber");
                if (f.integer && !Number.isInteger(v)) return t("input.notInteger");
                return v < f.min ? `≥ ${formatNumber(f.min)}` : null;
              }}
              onCommit={(s) => setOpts({ ...opts, [f.key]: s.trim() === "" ? null : parseNumber(s) })}
            />
          )}
        </Field>
      ))}
      <RunControls
        kind="boltz"
        runLabel={t("jobs.boltzGenerate")}
        blocked={bad ? t("jobs.boltzRangeError") : null}
        options={() => ({ module, opts })}
        filter={(j) => j.options.module === module}
        shown={2}
      />
    </Section>
  );
}
