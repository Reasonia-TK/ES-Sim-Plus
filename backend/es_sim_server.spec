# -*- mode: python ; coding: utf-8 -*-
"""ES-Sim バックエンドの PyInstaller 仕様 (prompts/44)。

単一実行ファイル (onefile) を生成し、Tauri のサイドカー (bundle.externalBin)
として同梱する。

- gmsh: Python ラッパ (gmsh.py) は ctypes で共有ライブラリ
  (libgmsh.so.X.Y / gmsh-X.Y.dll) を「自分と同じディレクトリ等」から探すため、
  検出したライブラリ本体をバンドルのルート (= 展開先 sys._MEIPASS) へ同梱する
- numpy / scipy: 公式 hooks (pyinstaller-hooks-contrib) が収集するが、
  uvicorn の文字列インポート "es_sim.server:app" などは静的解析に
  かからないため hiddenimports で明示する
- numba (prompts/76、粒子カーネルの JIT 化。optional 依存): pip install pyinstaller
  が pyinstaller-hooks-contrib を連れてくるため、同梱の hook-numba.py /
  hook-llvmlite.py により hiddenimports (cloudpickle 等) も llvmlite の共有
  ライブラリ (libllvmlite.so/dll、gmsh と同種の ctypes 動的ロード) も自動収集
  される (このリポジトリの Linux 環境で実際に onedir ビルドして確認済み)。
  "numba" は _numba_kernels.py の try/except 内で import しており、PyInstaller の
  静的解析 (AST 走査、try/except の有無に関わらず import 文自体は検出される) で
  通常は拾われるはずだが、念のため下の hiddenimports にも明示しておく。
  実行時の最終確認は GET /health の "numba" が true を返すことで行う
  (release.yml の smoke テストでアサーション、prompts/76)

ビルド:
    pyinstaller --clean --noconfirm es_sim_server.spec
生成物:
    dist/es-sim-backend(.exe)
"""

import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ---- gmsh 共有ライブラリの検出 (gmsh.py と同じ探索規則の簡略版) ----------------
import gmsh as _gmsh_mod

_moduledir = os.path.dirname(os.path.realpath(_gmsh_mod.__file__))
if sys.platform == "win32":
    _lib_pat = re.compile(r"^gmsh-\d+\.\d+\.dll$")
elif sys.platform == "darwin":
    _lib_pat = re.compile(r"^libgmsh\.\d+\.\d+\.dylib$")
else:
    _lib_pat = re.compile(r"^libgmsh\.so\.\d+\.\d+$")

_gmsh_lib = None
_search_dirs = []
for _base in (_moduledir, os.path.dirname(_moduledir), os.path.dirname(os.path.dirname(_moduledir))):
    for _sub in ("", "lib", "Lib", "bin"):
        _search_dirs.append(os.path.join(_base, _sub) if _sub else _base)
for _d in _search_dirs:
    if not os.path.isdir(_d):
        continue
    for _name in os.listdir(_d):
        if _lib_pat.match(_name):
            _gmsh_lib = os.path.join(_d, _name)
            break
    if _gmsh_lib:
        break
if _gmsh_lib is None:
    raise SystemExit(
        "gmsh の共有ライブラリ (libgmsh.so.* / gmsh-*.dll) が見つかりません。"
        "pip install gmsh で入れた環境でビルドしてください"
    )

# バンドルのルートに置けば gmsh.py の moduledir 探索で見つかる
binaries = [(_gmsh_lib, ".")]

# ---- hidden imports ------------------------------------------------------------
hiddenimports = [
    "es_sim",
    "es_sim.server",   # uvicorn.run("es_sim.server:app") の文字列参照
    "gmsh",
    "numba",           # _numba_kernels.py の try/except 内 import の保険 (prompts/76)
    "boltzpmp",        # boltz.py の try/except 内 import の保険 (numba と同じ理由、prompts/117/119)
    "boltzpmp._core",  # boltzpmp の Rust 拡張 (.pyd)。静的解析で拾われない場合の保険
]
# uvicorn のワーカ/ループ/プロトコル実装は動的インポートされる
hiddenimports += collect_submodules("uvicorn")
# v2 AMR の合成格子ソルバー (es_sim/amr/composite.py) は関数内で pyamg を import し、
# pyamg の C++ 拡張 (amg_core) はサブモジュールから読まれるため明示する (prompts/121)
hiddenimports += collect_submodules("pyamg", filter=lambda name: ".tests" not in name)
# DXF の読み書き (es_sim/dxf.py、prompts/132 P7g): ezdxf の C 拡張 (ezdxf.acc) は try の中で読まれるので明示する
# (無くても純 Python で動くが遅い)。ezdxf の同梱データ (GUI のアイコン・フォント) は読み書きに使わない
hiddenimports += collect_submodules("ezdxf.acc")

# boltzpmp 同梱の LXCat 断面積データ (boltzpmp/data/Ar.txt 等、load_argon() が参照する。
# 本体の LMEA テーブル生成 (xsprocess_to_mixture) 自体はこれらを使わないが、
# boltzpmp パッケージの一部として同梱しておく (prompts/117)
datas = collect_data_files("boltzpmp")
# v2 の CUDA カーネル (.cu) は実行時に NVRTC でコンパイルするためソースを同梱する (prompts/119)
datas += collect_data_files("es_sim", includes=["kernels/*.cu", "kernels/*.cuh"])

a = Analysis(
    ["run_server.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 不要な大物を除外してサイズ削減 (バックエンドは GUI を持たない)
        "tkinter",
        "matplotlib",
        "PIL",
        "IPython",
        "pytest",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="es-sim-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,           # UPX はアンチウイルス誤検知の一因になるため使わない
    console=True,        # サイドカーはウィンドウ非表示で起動される (ログは親が回収)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
