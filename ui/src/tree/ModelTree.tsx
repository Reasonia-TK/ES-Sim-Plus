// モデルツリー: 検索・折りたたみ・キーボード操作 (↑↓←→ Home End Enter F2 Delete)・右クリックメニュー・
// 領域の名前変更 (その場で編集)。ARIA の tree / treeitem で読み上げにも対応する。

import { ContextMenu } from "radix-ui";
import { useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "react-i18next";
import { askConfirm } from "../app/dialogs";
import { documentName, useDocument } from "../model/documentStore";
import { edgeCount } from "../model/project";
import { addCircleRegion, addRectRegion, deleteRegion, duplicateRegion, renameRegion, validateRegionId } from "../model/regionOps";
import { useSelection, type NodeId } from "../model/selection";
import { CommitText } from "../pages/inputs";
import { usePrefs } from "../prefs/prefs";
import { useJobs } from "../jobs/jobsStore";
import { buildTree, filterTree, visibleNodes, type TreeNode } from "./treeModel";

/** 境界条件の枝を最初から開いておく辺の数の上限 (同軸の例は 64 辺) */
const EXPAND_EDGES_MAX = 8;

function defaultExpanded(edges: number): Set<NodeId> {
  const ids = ["project", "geometry", "regions", "studies", "results"];
  if (edges <= EXPAND_EDGES_MAX) ids.push("boundaries");
  return new Set(ids);
}

function regionIdOf(node: NodeId): string | null {
  const m = /^region:(.+)$/.exec(node);
  return m ? m[1] : null;
}

export function ModelTree() {
  const { t } = useTranslation();
  const project = useDocument((s) => s.project);
  const docName = useDocument((s) => documentName(s, t("app.untitled")));
  const unit = usePrefs((s) => s.lengthUnit);
  const active = useSelection((s) => s.activeNode);
  const select = useSelection((s) => s.select);
  const docSerial = useDocument((s) => s.docSerial);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<Set<NodeId>>(() => defaultExpanded(edgeCount(project)));
  // 別の文書を開いたら開閉を初期化する
  const edges = edgeCount(project);
  const [seenDoc, setSeenDoc] = useState(docSerial);
  if (seenDoc !== docSerial) {
    setSeenDoc(docSerial);
    setExpanded(defaultExpanded(edges));
  }
  const [renaming, setRenaming] = useState<string | null>(null);
  const [renameError, setRenameError] = useState<string | null>(null);
  const treeRef = useRef<HTMLDivElement>(null);

  const jobs = useJobs((s) => s.jobs);
  const root = useMemo(() => buildTree(project, t, unit, docName, Object.values(jobs)), [project, t, unit, docName, jobs]);
  const shown = useMemo(() => filterTree(root, query), [root, query]);
  const filtering = query.trim() !== "";
  const rows = useMemo(() => (shown ? visibleNodes(shown, expanded, filtering) : []), [shown, expanded, filtering]);
  const selectable = rows.filter((r) => !r.node.placeholder);

  const toggle = (id: NodeId, open?: boolean) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (open ?? !next.has(id)) next.add(id);
      else next.delete(id);
      return next;
    });

  const update = useDocument.getState().update;

  const addRegion = (kind: "rect" | "circle") => {
    let id = "";
    update(t("action.addRegion"), (d) => {
      id = kind === "rect" ? addRectRegion(d) : addCircleRegion(d);
    });
    toggle("regions", true);
    select(`region:${id}`);
  };

  const removeRegion = async (id: string) => {
    if (!(await askConfirm(t("tree.deleteRegionTitle"), t("tree.deleteRegionMessage", { id }), { okLabel: t("tree.delete"), danger: true })))
      return;
    update(t("action.deleteRegion"), (d) => deleteRegion(d, id));
    select("regions");
  };

  const copyRegion = (id: string) => {
    let copy: string | null = null;
    update(t("action.duplicateRegion"), (d) => {
      copy = duplicateRegion(d, id);
    });
    if (copy) select(`region:${copy}`);
  };

  const focusRow = (id: NodeId) => {
    select(id);
    requestAnimationFrame(() => treeRef.current?.querySelector<HTMLElement>(`[data-node="${CSS.escape(id)}"]`)?.focus());
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (renaming !== null) return;
    const idx = selectable.findIndex((r) => r.node.id === active);
    const cur = selectable[idx];
    const move = (i: number) => {
      const r = selectable[Math.max(0, Math.min(selectable.length - 1, i))];
      if (r) focusRow(r.node.id);
    };
    switch (e.key) {
      case "ArrowDown":
        move(idx + 1);
        break;
      case "ArrowUp":
        move(idx - 1);
        break;
      case "Home":
        move(0);
        break;
      case "End":
        move(selectable.length - 1);
        break;
      case "ArrowRight":
        if (cur?.node.children?.length) {
          if (!expanded.has(cur.node.id) && !filtering) toggle(cur.node.id, true);
          else move(idx + 1);
        }
        break;
      case "ArrowLeft":
        if (cur?.node.children?.length && expanded.has(cur.node.id) && !filtering) toggle(cur.node.id, false);
        else if (cur?.parent) focusRow(cur.parent);
        break;
      case "F2": {
        const rid = cur ? regionIdOf(cur.node.id) : null;
        if (rid) setRenaming(rid);
        break;
      }
      case "Delete": {
        const rid = cur ? regionIdOf(cur.node.id) : null;
        if (rid) void removeRegion(rid);
        break;
      }
      default:
        return;
    }
    e.preventDefault();
  };

  const renderRow = ({ node, level }: { node: TreeNode; level: number }) => {
    const hasKids = !!node.children?.length;
    const open = filtering || expanded.has(node.id);
    const rid = regionIdOf(node.id);
    const isActive = node.id === active;
    const row = (
      <div
        key={node.id}
        role="treeitem"
        aria-level={level}
        aria-expanded={hasKids ? open : undefined}
        aria-selected={isActive}
        aria-disabled={node.placeholder || undefined}
        data-node={node.id}
        tabIndex={isActive ? 0 : -1}
        className={`tree-row${isActive ? " active" : ""}${node.placeholder ? " placeholder" : ""}`}
        style={{ paddingLeft: 6 + (level - 1) * 14 }}
        onClick={() => !node.placeholder && select(node.id)}
        onDoubleClick={() => (hasKids ? toggle(node.id) : rid && setRenaming(rid))}
      >
        <span
          className={`tree-twisty${hasKids ? "" : " leaf"}`}
          onClick={(e) => {
            e.stopPropagation();
            if (hasKids && !filtering) toggle(node.id);
          }}
        >
          {hasKids ? (open ? "▾" : "▸") : ""}
        </span>
        {rid !== null && renaming === rid ? (
          <span className="tree-rename" onBlur={() => setRenaming(null)} onClick={(e) => e.stopPropagation()}>
            <CommitText
              autoFocus
              value={rid}
              aria-label={t("tree.rename")}
              validate={(v) => {
                const k = validateRegionId(project, rid, v);
                return k ? t(k, { id: v.trim() }) : null;
              }}
              onError={setRenameError}
              onCommit={(v) => {
                const to = v.trim();
                update(t("action.renameRegion"), (d) => renameRegion(d, rid, to));
                select(`region:${to}`);
              }}
              onCancel={() => setRenaming(null)}
            />
          </span>
        ) : (
          <span className="tree-label">{node.label}</span>
        )}
        {node.detail && !(rid !== null && renaming === rid) && <span className="tree-detail">{node.detail}</span>}
        {node.badge && <span className={`badge badge-${node.badge.tone}`}>{node.badge.text}</span>}
      </div>
    );
    if (node.id === "regions") {
      return (
        <ContextMenu.Root key={node.id}>
          <ContextMenu.Trigger asChild>{row}</ContextMenu.Trigger>
          <ContextMenu.Portal>
            <ContextMenu.Content className="menu-content">
              <ContextMenu.Item className="menu-item" onSelect={() => addRegion("rect")}>
                {t("tree.addRectangle")}
              </ContextMenu.Item>
              <ContextMenu.Item className="menu-item" onSelect={() => addRegion("circle")}>
                {t("tree.addCircle")}
              </ContextMenu.Item>
            </ContextMenu.Content>
          </ContextMenu.Portal>
        </ContextMenu.Root>
      );
    }
    if (rid !== null) {
      return (
        <ContextMenu.Root key={node.id}>
          <ContextMenu.Trigger asChild onContextMenu={() => select(node.id)}>
            {row}
          </ContextMenu.Trigger>
          <ContextMenu.Portal>
            <ContextMenu.Content className="menu-content">
              <ContextMenu.Item className="menu-item" onSelect={() => setRenaming(rid)}>
                {t("tree.rename")}
                <span className="menu-shortcut">F2</span>
              </ContextMenu.Item>
              <ContextMenu.Item className="menu-item" onSelect={() => copyRegion(rid)}>
                {t("tree.duplicate")}
              </ContextMenu.Item>
              <ContextMenu.Separator className="menu-separator" />
              <ContextMenu.Item className="menu-item danger" onSelect={() => void removeRegion(rid)}>
                {t("tree.delete")}
                <span className="menu-shortcut">Del</span>
              </ContextMenu.Item>
            </ContextMenu.Content>
          </ContextMenu.Portal>
        </ContextMenu.Root>
      );
    }
    return row;
  };

  return (
    <div className="tree-panel">
      <div className="panel-header">
        <input
          className="input tree-search"
          type="search"
          placeholder={t("tree.search")}
          aria-label={t("tree.search")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <div
        ref={treeRef}
        className="tree"
        role="tree"
        aria-label={t("tree.project")}
        onKeyDown={onKeyDown}
        onBlur={() => renameError && setRenameError(null)}
      >
        {rows.map((r) => renderRow(r))}
        {renameError && <div className="field-error tree-error">{renameError}</div>}
      </div>
    </div>
  );
}
