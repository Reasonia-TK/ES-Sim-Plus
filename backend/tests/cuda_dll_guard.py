"""pytest プラグイン: セッションの終わりに、プロセスが読み込んだ CUDA の DLL を JSON に書く (Windows 専用)。

test_v2_gpu_nvrtc_only.py が NVRTC だけの環境で子の pytest に ``-p cuda_dll_guard`` で読ませる。書き出し先は
環境変数 ES_SIM_DLL_REPORT (prompts/133 P8a)。
"""

from __future__ import annotations

import ctypes
import json
import os
from ctypes import wintypes

_KEYS = ("cuda", "nvrtc", "cublas", "cusparse", "cusolver", "curand", "cufft", "nvjitlink", "cutensor", "nvcu")


def loaded_modules() -> list[str]:
    """このプロセスに読み込まれているモジュール (DLL・pyd) のパス。"""
    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    proc = kernel32.GetCurrentProcess()
    mods = (wintypes.HMODULE * 4096)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModulesEx(proc, mods, ctypes.sizeof(mods), ctypes.byref(needed), 3):
        return []
    buf = ctypes.create_unicode_buffer(1024)
    out = []
    for i in range(needed.value // ctypes.sizeof(wintypes.HMODULE)):
        if psapi.GetModuleFileNameExW(proc, mods[i], buf, 1024):
            out.append(buf.value)
    return out


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("ES_SIM_DLL_REPORT")
    if not path:
        return
    mods = sorted({m for m in loaded_modules() if any(k in os.path.basename(m).lower() for k in _KEYS)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"modules": mods, "exitstatus": int(exitstatus)}, f, ensure_ascii=False, indent=1)
