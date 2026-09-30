// メニューバー (ファイル・編集・表示・実行・ヘルプ)。ショートカットは useShortcuts が処理し、ここは表示だけ。

import { Menubar } from "radix-ui";
import { useTranslation } from "react-i18next";
import { exportDxfFile, importDxfFile } from "../io/dxf";
import { EXAMPLES } from "../io/examples";
import { newDocument, openDocument, openExample, openRecentFile, saveDocument, saveDocumentAs, saveDocumentWithResults } from "../io/documents";
import { useDocument, useRedoLabel, useUndoLabel } from "../model/documentStore";
import { usePrefs, type Language, type Theme } from "../prefs/prefs";
import type { LengthUnit } from "../util/format";
import { showAbout, showPortDialog } from "./dialogs";
import { resetLayout } from "./layout";
import { MOD } from "./shortcuts";
import { useConnection } from "../backend/connection";
import { stopAllRuns, useJobs } from "../jobs/jobsStore";
import { BUNDLE_KINDS } from "../results/bundle";
import { useStatic } from "../results/staticResults";
import { useBottomTab } from "./BottomPanel";

function Item({ label, shortcut, onSelect, disabled }: { label: string; shortcut?: string; onSelect: () => void; disabled?: boolean }) {
  return (
    <Menubar.Item className="menu-item" onSelect={onSelect} disabled={disabled}>
      {label}
      {shortcut && <span className="menu-shortcut">{shortcut}</span>}
    </Menubar.Item>
  );
}

function Radio<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <Menubar.RadioGroup value={value} onValueChange={(v) => onChange(v as T)}>
      {options.map((o) => (
        <Menubar.RadioItem key={o.value} className="menu-item menu-radio" value={o.value}>
          <Menubar.ItemIndicator className="menu-indicator">●</Menubar.ItemIndicator>
          {o.label}
        </Menubar.RadioItem>
      ))}
    </Menubar.RadioGroup>
  );
}

function Sub({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Menubar.Sub>
      <Menubar.SubTrigger className="menu-item">
        {label}
        <span className="menu-shortcut">›</span>
      </Menubar.SubTrigger>
      <Menubar.Portal>
        <Menubar.SubContent className="menu-content" sideOffset={2} alignOffset={-4}>
          {children}
        </Menubar.SubContent>
      </Menubar.Portal>
    </Menubar.Sub>
  );
}

function Menu({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Menubar.Menu>
      <Menubar.Trigger className="menubar-trigger">{label}</Menubar.Trigger>
      <Menubar.Portal>
        <Menubar.Content className="menu-content" align="start" sideOffset={4}>
          {children}
        </Menubar.Content>
      </Menubar.Portal>
    </Menubar.Menu>
  );
}

const Sep = () => <Menubar.Separator className="menu-separator" />;

export function MenuBar() {
  const { t } = useTranslation();
  const prefs = usePrefs();
  const undoLabel = useUndoLabel();
  const redoLabel = useRedoLabel();
  const { undo, redo } = useDocument.getState();
  const connected = useConnection((s) => s.status === "connected");
  const staticBusy = useStatic((s) => s.busy !== null);
  const anyRunning = useJobs((s) => Object.values(s.jobs).some((j) => (j.state === "running" || j.state === "queued") && !j.imported));
  // 結果付きで保存できるもの (静電場の結果か、保存する種類の実行の結果) があるか
  const staticResults = useStatic((s) => s.solve !== null || s.mesh !== null);
  const runResults = useJobs((s) => Object.values(s.jobs).some((j) => j.has_result && BUNDLE_KINDS.some((b) => b.kind === j.kind)));
  const hasResults = staticResults || runResults;
  return (
    <Menubar.Root className="menubar">
      <span className="app-title">{t("app.name")}</span>
      <Menu label={t("menu.file")}>
        <Item label={t("menu.new")} shortcut={`${MOD}+N`} onSelect={() => void newDocument()} />
        <Item label={t("menu.open")} shortcut={`${MOD}+O`} onSelect={() => void openDocument()} />
        <Sub label={t("menu.openRecent")}>
          {prefs.recent.length === 0 && <Item label={t("menu.noRecent")} onSelect={() => {}} disabled />}
          {prefs.recent.map((r) => (
            <Item key={`${r.path ?? r.handleId}`} label={r.name} onSelect={() => void openRecentFile(r)} />
          ))}
          {prefs.recent.length > 0 && (
            <>
              <Sep />
              <Item label={t("menu.clearRecent")} onSelect={prefs.clearRecent} />
            </>
          )}
        </Sub>
        <Sub label={t("menu.examples")}>
          {EXAMPLES.map((ex) => (
            <Item
              key={ex.key}
              label={t(`examples.${ex.key}` as "examples.parallel_plates", { defaultValue: ex.key })}
              onSelect={() => void openExample(ex.key)}
            />
          ))}
        </Sub>
        <Sep />
        <Item label={t("menu.save")} shortcut={`${MOD}+S`} onSelect={() => void saveDocument()} />
        <Item label={t("menu.saveAs")} shortcut={`${MOD}+Shift+S`} onSelect={() => void saveDocumentAs()} />
        <Item label={t("menu.saveWithResults")} onSelect={() => void saveDocumentWithResults()} disabled={!hasResults} />
        <Sep />
        <Item label={t("menu.importDxf")} onSelect={() => void importDxfFile()} disabled={!connected} />
        <Item label={t("menu.exportDxf")} onSelect={() => void exportDxfFile()} disabled={!connected} />
      </Menu>
      <Menu label={t("menu.edit")}>
        <Item
          label={undoLabel ? t("menu.undoWhat", { label: undoLabel }) : t("menu.undo")}
          shortcut={`${MOD}+Z`}
          onSelect={undo}
          disabled={!undoLabel}
        />
        <Item
          label={redoLabel ? t("menu.redoWhat", { label: redoLabel }) : t("menu.redo")}
          shortcut={`${MOD}+Y`}
          onSelect={redo}
          disabled={!redoLabel}
        />
      </Menu>
      <Menu label={t("menu.view")}>
        <Sub label={t("menu.theme")}>
          <Radio<Theme>
            value={prefs.theme}
            onChange={prefs.setTheme}
            options={[
              { value: "dark", label: t("menu.themeDark") },
              { value: "light", label: t("menu.themeLight") },
            ]}
          />
        </Sub>
        <Sub label={t("menu.language")}>
          <Radio<Language>
            value={prefs.language}
            onChange={prefs.setLanguage}
            options={[
              { value: "ja", label: "日本語" },
              { value: "en", label: "English" },
            ]}
          />
        </Sub>
        <Sub label={t("menu.lengthUnit")}>
          <Radio<LengthUnit>
            value={prefs.lengthUnit}
            onChange={prefs.setLengthUnit}
            options={[
              { value: "mm", label: "mm" },
              { value: "um", label: "µm" },
            ]}
          />
        </Sub>
        <Sep />
        <Item label={t("menu.resetLayout")} onSelect={resetLayout} />
      </Menu>
      <Menu label={t("menu.run")}>
        <Item label={t("menu.runMesh")} onSelect={() => void useStatic.getState().runMesh()} disabled={!connected || staticBusy} />
        <Item label={t("menu.runSolve")} onSelect={() => void useStatic.getState().runSolve()} disabled={!connected || staticBusy} />
        <Sep />
        <Item label={t("menu.stopAll")} onSelect={() => void stopAllRuns()} disabled={!anyRunning} />
        <Sep />
        <Item label={t("menu.showProgress")} onSelect={() => useBottomTab.getState().setTab("progress")} />
        <Item label={t("menu.showJobs")} onSelect={() => useBottomTab.getState().setTab("jobs")} />
      </Menu>
      <Menu label={t("menu.help")}>
        <Item label={t("menu.backendPort")} onSelect={() => void showPortDialog()} />
        <Item label={t("menu.about")} onSelect={() => void showAbout()} />
      </Menu>
    </Menubar.Root>
  );
}
