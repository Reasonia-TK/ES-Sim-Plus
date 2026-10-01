"""配布物に入る第三者のソフトウェアの一覧と使用許諾をまとめる (prompts/133 P8d)。

scripts/build_app.ps1 がバックエンドの venv の Python で実行し、THIRD_PARTY_NOTICES.txt を作る
(tauri.bundle.json がインストール先のルートに同梱する)。対象:

- Python: CPython と、es-sim[gpu] と依存グループ dist の nvidia-cuda-nvrtc の実行時の依存 (再帰)。PyInstaller・
  pytest など作るときだけのものは入れない。gmsh (GPL-2.0 以降) と NVIDIA の NVRTC (NVIDIA の使用許諾) を含む
- UI (npm): ui/package.json の dependencies とその依存 (再帰、node_modules から)
- アプリ (Rust): ui/src-tauri の Windows 向けの通常の依存 (cargo metadata、ビルド・開発用の依存は除く)

使い方: python scripts/collect_licenses.py --out ui/src-tauri/target/licenses
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LICENSE_NAME = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT|UNLICENSE)([-_.].*)?$", re.IGNORECASE)
#: 作るときだけ使う Python のパッケージ (配布物に入らない)
_BUILD_ONLY = {"pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile", "pywin32-ctypes", "setuptools", "pip"}


@dataclass
class Component:
    ecosystem: str
    name: str
    version: str
    license: str
    url: str = ""
    texts: list[tuple[str, str]] = field(default_factory=list)  # (ファイル名, 中身)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ---- Python --------------------------------------------------------------------------------------


def _requirements(dist: md.Distribution, extras: set[str]) -> list[tuple[str, set[str]]]:
    """dist の実行時の依存 (名前と、その依存の extras)。extras は dist 自身の有効な extras。"""
    out = []
    for req in dist.requires or []:
        spec, _, marker = req.partition(";")
        m_name = re.match(r"\s*([A-Za-z0-9._-]+)\s*(?:\[([^\]]*)\])?", spec)
        if not m_name:
            continue
        name = m_name.group(1)
        sub_extras = {e.strip() for e in (m_name.group(2) or "").split(",") if e.strip()}
        marker = marker.strip()
        if marker:
            m = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", marker)
            if m:
                if m.group(1) not in extras:
                    continue
                marker = re.sub(r"(and\s+)?extra\s*==\s*['\"][^'\"]+['\"]", "", marker).strip()
            if marker and not _marker_ok(marker):
                continue
        out.append((name, sub_extras))
    return out


def _marker_ok(marker: str) -> bool:
    try:
        from packaging.markers import Marker

        return Marker(marker).evaluate()
    except Exception:  # noqa: BLE001 - packaging が無い・読めない印は入れておく (多めは害が無い)
        return True


def python_components() -> list[Component]:
    seen: dict[str, md.Distribution] = {}
    todo: list[tuple[str, set[str]]] = [("es-sim", {"gpu"}), ("nvidia-cuda-nvrtc", set())]
    while todo:
        name, extras = todo.pop()
        key = _norm(name)
        if key in seen or key in _BUILD_ONLY:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        seen[key] = dist
        todo += _requirements(dist, extras)
    out = [_cpython()]
    for key, dist in sorted(seen.items()):
        if key == "es-sim":
            continue
        meta = dist.metadata
        lic = meta.get("License-Expression") or ""
        if not lic:
            raw = (meta.get("License") or "").strip()
            lic = raw if raw and "\n" not in raw and len(raw) < 80 else ""
        if not lic:
            lic = "; ".join(c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::"))
        urls = [u.split(",", 1)[-1].strip() for u in meta.get_all("Project-URL") or []]
        url = meta.get("Home-page") or (urls[0] if urls else "")
        comp = Component("Python", meta["Name"], dist.version, lic or "(記載なし)", url)
        for f in dist.files or []:
            if _LICENSE_NAME.match(Path(str(f)).name) and ".dist-info" in str(f) or str(f).startswith("../"):
                p = Path(dist.locate_file(f))
                if p.is_file() and _LICENSE_NAME.match(p.name):
                    comp.texts.append((str(f).replace("\\", "/"), _read(p)))
        out.append(comp)
    return out


def _cpython() -> Component:
    import platform

    comp = Component("Python", "CPython", platform.python_version(), "PSF-2.0", "https://www.python.org/")
    lic = Path(sys.base_prefix) / "LICENSE.txt"
    if lic.is_file():
        comp.texts.append(("LICENSE.txt", _read(lic)))
    return comp


# ---- npm -----------------------------------------------------------------------------------------


def npm_components(ui: Path) -> list[Component]:
    nm = ui / "node_modules"
    roots = json.loads((ui / "package.json").read_text(encoding="utf-8")).get("dependencies", {})
    seen: dict[Path, Component] = {}
    todo: list[tuple[str, Path]] = [(name, ui) for name in roots]
    while todo:
        name, base = todo.pop()
        pkg_dir = None
        d = base
        while True:  # node の解決規則: 近い node_modules から上へ
            cand = d / "node_modules" / name
            if (cand / "package.json").is_file():
                pkg_dir = cand
                break
            if d == ui or d.parent == d:
                break
            d = d.parent
            while d != ui and d.name != "node_modules" and d.parent.name != "node_modules" and not (d / "package.json").is_file():
                d = d.parent
        if pkg_dir is None:
            cand = nm / name
            pkg_dir = cand if (cand / "package.json").is_file() else None
        if pkg_dir is None or pkg_dir.resolve() in seen:
            continue
        pj = json.loads((pkg_dir / "package.json").read_text(encoding="utf-8"))
        lic = pj.get("license") or ""
        if isinstance(lic, dict):
            lic = lic.get("type", "")
        if not lic and isinstance(pj.get("licenses"), list):
            lic = " OR ".join(x.get("type", "") for x in pj["licenses"] if isinstance(x, dict))
        repo = pj.get("repository") or pj.get("homepage") or ""
        if isinstance(repo, dict):
            repo = repo.get("url", "")
        comp = Component("npm", pj.get("name", name), pj.get("version", "?"), lic or "(記載なし)", str(repo))
        for f in sorted(pkg_dir.iterdir()):
            if f.is_file() and _LICENSE_NAME.match(f.name):
                comp.texts.append((f.name, _read(f)))
        seen[pkg_dir.resolve()] = comp
        todo += [(dep, pkg_dir) for dep in (pj.get("dependencies") or {})]
    return sorted(seen.values(), key=lambda c: (c.name.lower(), c.version))


# ---- Rust ----------------------------------------------------------------------------------------


def cargo_components(manifest: Path, target: str = "x86_64-pc-windows-msvc") -> list[Component]:
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--filter-platform", target, "--manifest-path", str(manifest)],
        check=True, capture_output=True, text=True, encoding="utf-8").stdout)
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    shipped: set[str] = set()
    todo = [root]
    while todo:
        pid = todo.pop()
        if pid in shipped:
            continue
        shipped.add(pid)
        for dep in nodes[pid]["deps"]:
            if any(k.get("kind") is None for k in dep.get("dep_kinds", [])):  # 通常の依存だけ (build・dev は除く)
                todo.append(dep["pkg"])
    out = []
    for pid in shipped - {root}:
        p = pkgs[pid]
        comp = Component("Rust", p["name"], p["version"], p.get("license") or "(記載なし)",
                         p.get("repository") or p.get("homepage") or "")
        crate_dir = Path(p["manifest_path"]).parent
        files = sorted(f for f in crate_dir.iterdir() if f.is_file() and _LICENSE_NAME.match(f.name))
        if p.get("license_file"):
            lf = crate_dir / p["license_file"]
            if lf.is_file() and lf not in files:
                files.append(lf)
        comp.texts += [(f.name, _read(f)) for f in files]
        out.append(comp)
    return sorted(out, key=lambda c: (c.name.lower(), c.version))


# ---- 書き出し ------------------------------------------------------------------------------------


def write_notices(comps: list[Component], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "ES-Sim に含まれる第三者のソフトウェア (THIRD-PARTY NOTICES)",
        "=" * 78,
        "",
        "ES-Sim は GNU General Public License version 3 またはそれ以降 (GPL-3.0-or-later) で、NVIDIA CUDA のライブラリ",
        "(NVRTC など) と組み合わせて配ることを許す追加許可 (GPL v3 第 7 条) を付けています (LICENSE.txt)。",
        "ES-Sim の配布物 (インストーラ) は次のソフトウェアを含みます。各ソフトウェアはそれぞれの使用許諾に従います。",
        "gmsh は GNU General Public License (version 2 以降) で、そのソースは https://gmsh.info/ から入手できます。",
        "NVIDIA CUDA の NVRTC (nvrtc64_*.dll・nvrtc-builtins64_*.dll) は NVIDIA の使用許諾 (下記) による再配布です。",
        "",
        "This distribution of ES-Sim includes the following third-party software, each under its own license.",
        "",
        f"{'種類':6s}  {'名前':40s} {'版':16s} 使用許諾",
        "-" * 78,
    ]
    for c in comps:
        lines.append(f"{c.ecosystem:6s}  {c.name:40s} {c.version:16s} {c.license}")
    lines += ["", ""]
    for c in comps:
        lines += ["=" * 78, f"{c.name} {c.version} ({c.ecosystem}) — {c.license}"]
        if c.url:
            lines.append(c.url)
        lines.append("=" * 78)
        if not c.texts:
            lines += ["(使用許諾の文面はパッケージに含まれていない。上の使用許諾の名前を参照)", ""]
        for fname, text in c.texts:
            lines += [f"--- {fname} ---", text, ""]
        lines.append("")
    path = out_dir / "THIRD_PARTY_NOTICES.txt"
    path.write_text("\n".join(lines), encoding="utf-8-sig", newline="\r\n")
    (out_dir / "third_party.json").write_text(json.dumps(
        [{"ecosystem": c.ecosystem, "name": c.name, "version": c.version, "license": c.license, "url": c.url}
         for c in comps], ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    # 日本語の Windows 以外 (GitHub のランナーは cp1252) でも標準出力に日本語を書けるように UTF-8 にする
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "ui" / "src-tauri" / "target" / "licenses"))
    args = ap.parse_args(argv)
    py = python_components()
    js = npm_components(ROOT / "ui")
    rs = cargo_components(ROOT / "ui" / "src-tauri" / "Cargo.toml")
    path = write_notices(py + js + rs, Path(args.out))
    no_text = [f"{c.ecosystem}:{c.name}" for c in py + js + rs if not c.texts]
    print(f"Python {len(py)}・npm {len(js)}・Rust {len(rs)} → {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if no_text:
        print(f"使用許諾の文面が無いもの ({len(no_text)}): {', '.join(no_text[:20])}{' …' if len(no_text) > 20 else ''}")
    gpl = [c.name for c in py + js + rs if re.search(r"\bGPL", c.license) and "LGPL" not in c.license]
    if gpl:
        print(f"GPL のもの: {', '.join(gpl)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
