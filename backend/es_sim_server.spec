# -*- mode: python ; coding: utf-8 -*-
"""ES-Sim バックエンドの PyInstaller 仕様 (prompts/44、onedir・GPU は prompts/133 P8b)。

**onedir** (dist/es-sim-backend/ に es-sim-backend.exe と _internal/) を作り、Tauri のリソースとして同梱する。
onefile は起動のたびに全部 (数百 MB) を一時フォルダへ展開するので使わない。

- gmsh: Python ラッパ (gmsh.py) は ctypes で共有ライブラリ (libgmsh.so.X.Y / gmsh-X.Y.dll) を「自分と同じ
  ディレクトリ等」から探すため、検出したライブラリ本体をバンドルのルート (_internal) へ同梱する
- numpy / scipy: 公式 hooks (pyinstaller-hooks-contrib) が収集するが、uvicorn の文字列インポート
  "es_sim.server:app" などは静的解析にかからないため hiddenimports で明示する
- numba (prompts/76): pyinstaller-hooks-contrib の hook-numba.py / hook-llvmlite.py が hiddenimports と llvmlite の
  共有ライブラリを集める。_numba_kernels.py の try/except 内の import の保険として hiddenimports にも明示する。
  es_sim はソース (.py) のまま入れる (numba の cache=True はソースの場所にキャッシュを置くため)
- CuPy (prompts/133): 拡張モジュール (.pyd) の中の import は静的解析に掛からないので、cupy・cupy_backends・cupyx の
  サブモジュールを全部集める (P6f の onefile は cupy_backends.cuda._softlink が漏れて import できなかった)。CuPy の
  ヘッダ (cupy/_core/include、CuPy 自身のカーネルを NVRTC でコンパイルするときに使う) と cupy/.data の設定も入れる
- CUDA: NVRTC (依存グループ dist の nvidia-cuda-nvrtc の wheel、無ければ CUDA Toolkit) だけを cuda/bin/x64 に入れ、
  実行時フック pyi_rth_es_sim.py が CUDA_PATH をそこへ向ける。PyInstaller が CuPy の拡張のリンク先として開発機の
  CUDA Toolkit から拾う DLL (cuBLAS・cuSOLVER・cuSPARSE・cuFFT・cuRAND など、計 1.3 GB) は除く。GPU の計算は
  NVRTC だけで動く (P8a、tests/test_v2_gpu_nvrtc_only.py)。CUDA のランタイムは NVIDIA のドライバに含まれる
- Python は UTF-8 モード (X utf8): 標準出力を UTF-8 にする (Tauri が AppConfig/backend.log に UTF-8 として書く)

ビルド (依存グループ dist が要る: uv sync --extra gpu --group dist):
    uv run --no-sync pyinstaller --clean --noconfirm es_sim_server.spec
生成物:
    dist/es-sim-backend/es-sim-backend(.exe) と dist/es-sim-backend/_internal/
確かめ方 (CUDA の環境変数なしで、scripts/build_backend.ps1 が行う):
    dist/es-sim-backend/es-sim-backend selftest --require-gpu
"""

import glob
import importlib.util
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
# es_sim のモジュールは全部 (関数の中で import するもの、ジョブの実行器など)
hiddenimports += collect_submodules("es_sim")
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

# ---- CuPy と NVRTC (prompts/133 P8b) ---------------------------------------------
#: 同梱しない CUDA の DLL (CuPy の拡張のリンク先として拾われる。GPU の計算は NVRTC だけで動く、P8a)
_CUDA_DLL = re.compile(
    r"^(cublas|cublaslt|cusolver|cusolvermg|cusparse|cufft|cufftw|curand|nvjitlink|cutensor|cutensormg|nvrtc|"
    r"nvrtc-builtins|cudart|nvjpeg|npp|nvblas|nvfatbin)[^\\/]*\.dll$",
    re.IGNORECASE,
)
_CUDA_DEST = os.path.join("cuda", "bin", "x64")


def _py_modules_under(pkg: str) -> list[str]:
    """パッケージの下の .py を全部モジュール名にする (__init__.py の無い名前空間のサブパッケージも。
    collect_submodules は pkgutil に頼るのでそれを見落とす: cuda.pathfinder の _dynamic_libs など)。"""
    spec = importlib.util.find_spec(pkg)
    out = []
    for base in list(spec.submodule_search_locations or []) if spec else []:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            rel = os.path.relpath(root, base)
            prefix = pkg if rel == "." else pkg + "." + rel.replace(os.sep, ".")
            out += [prefix if f == "__init__.py" else f"{prefix}.{f[:-3]}" for f in files if f.endswith(".py")]
    return out


def _not_tests(name: str) -> bool:
    # cupy.testing は残す (cupy/__init__ が LazyLoader で find_spec するので、無いと import cupy が失敗する)
    return ".tests" not in name


def _nvrtc_dlls() -> tuple[list[str], str | None]:
    """同梱する NVRTC の DLL と使用許諾のファイル。pip の wheel (nvidia-cuda-nvrtc) を優先し、無ければ CUDA Toolkit。"""
    spec = importlib.util.find_spec("nvidia")
    for base in list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []:
        dlls = sorted(glob.glob(os.path.join(base, "cu13", "bin", "*", "nvrtc64_*_0.dll")))
        if dlls:
            d = os.path.dirname(dlls[-1])
            lic = sorted(glob.glob(os.path.join(os.path.dirname(base), "nvidia_cuda_nvrtc-*.dist-info", "licenses",
                                                "*.txt")))
            return [dlls[-1]] + sorted(glob.glob(os.path.join(d, "nvrtc-builtins64_*.dll"))), (lic[-1] if lic else None)
    cuda = os.environ.get("CUDA_PATH", "")
    for d in (os.path.join(cuda, "bin", "x64"), os.path.join(cuda, "bin")):
        dlls = sorted(glob.glob(os.path.join(d, "nvrtc64_*_0.dll")))
        if dlls:
            print(f"warning: NVRTC を CUDA Toolkit から同梱します ({d})。uv sync --group dist で wheel を入れてください")
            eula = os.path.join(cuda, "EULA.txt")
            return [dlls[-1]] + sorted(glob.glob(os.path.join(d, "nvrtc-builtins64_*.dll"))), (
                eula if os.path.isfile(eula) else None)
    return [], None


_HAVE_CUPY = importlib.util.find_spec("cupy") is not None
if _HAVE_CUPY and sys.platform == "win32":
    hiddenimports += collect_submodules("cupy", filter=_not_tests)
    hiddenimports += collect_submodules("cupy_backends", filter=_not_tests)
    hiddenimports += collect_submodules("cupyx", filter=_not_tests)
    # CuPy が CUDA のライブラリ・ヘッダを探す。子プロセスで自分を呼ぶ (run_server.py の _run_child_module)
    hiddenimports += _py_modules_under("cuda.pathfinder")
    datas += collect_data_files("cupy", includes=["_core/include/**/*", ".data/*.json"])
    _nvrtc, _license = _nvrtc_dlls()
    if len(_nvrtc) < 2:
        raise SystemExit("NVRTC (nvrtc64_*_0.dll と nvrtc-builtins64_*.dll) が見つかりません。"
                         "uv sync --extra gpu --group dist で nvidia-cuda-nvrtc を入れてください")
    binaries += [(f, _CUDA_DEST) for f in _nvrtc]
    if _license:
        datas.append((_license, "cuda"))  # NVIDIA の使用許諾 (NVRTC は再配布できるライブラリ)

a = Analysis(
    ["run_server.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyi_rth_es_sim.py"],
    excludes=[
        # 不要な大物を除外してサイズ削減 (バックエンドは GUI を持たない)
        "tkinter",
        "matplotlib",
        "PIL",
        "IPython",
        "pytest",
    ],
    noarchive=False,
    # es_sim はソースのまま (numba のキャッシュ、トレースバックの行)
    module_collection_mode={"es_sim": "py"},
)
# CUDA Toolkit から拾われた DLL を除く (cuda/bin/x64 に入れた NVRTC だけを残す)
a.binaries = [
    b for b in a.binaries
    if not (_CUDA_DLL.match(os.path.basename(b[0])) and os.path.normpath(os.path.dirname(b[0])) != _CUDA_DEST)
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [("X utf8", None, "OPTION")],
    exclude_binaries=True,
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
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="es-sim-backend",
)
