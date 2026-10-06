// 収束の判定の設定 (prompts/137): PIC 2D・PIC 1D・流体 1D・流体 2D の設定の中の convergence。周期平均の φ・n_e・
// 電子とイオンの総数・自己バイアスを比べ、変化と残りの変化が閾値以下の周期が続いたら収束とする (backend の
// convergence.py)。止める設定なら、収束から平均区間を取って早めに終える。

import { useTranslation } from "react-i18next";
import { Section } from "../../forms/blocks";
import { SchemaField } from "../../forms/SchemaField";
import type { Path } from "../../schema/schema";

export function ConvergenceSection({ base, kind }: { base: Path; kind: "fluid" | "pic" }) {
  const { t } = useTranslation();
  const P: Path = [...base, "convergence"];
  return (
    <Section title={t("studyCommon.convergence")}>
      <SchemaField path={[...P, "enabled"]} />
      <SchemaField path={[...P, "tol"]} placeholder={t(kind === "pic" ? "studyCommon.convTolPic" : "studyCommon.convTolFluid")} />
      <SchemaField path={[...P, "rf_periods"]} />
      <SchemaField path={[...P, "steps"]} placeholder={t("studyCommon.convStepsAuto")} />
      <SchemaField path={[...P, "hold"]} />
      <SchemaField path={[...P, "stop"]} />
      <SchemaField path={[...P, "max_window"]} />
    </Section>
  );
}
