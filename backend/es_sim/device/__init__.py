"""v2 計算デバイス層 — CPU (NumPy + Numba) / GPU (CuPy + NVRTC) の切替 (prompts/119)。

AMReX の「同じソースを CPU/GPU で動かす」考え方を Python で実現するための薄い層。

- デバイスの選択は ``get_device(kind)`` の引数、または環境変数 ``ES_SIM_DEVICE``
  (``auto`` | ``cpu`` | ``cuda``、既定 ``auto``) で決める。``auto`` は CuPy が import でき
  CUDA デバイスが見える場合に ``cuda``、それ以外は ``cpu``。
- ソルバーは ``Device.xp`` (numpy か cupy) だけを見て配列演算を書き、ホットループは
  CUDA カーネル (``es_sim/kernels/*.cu`` を NVRTC で実行時コンパイル、``device/cuda.py``)
  と Numba カーネルの二系統を同じ数式で持つ (GPU の無い CI では CPU 経路でテストし、
  GPU 環境では CPU/GPU 一致テストを走らせる)。
- 配列はデバイスメモリに置く (CUDA managed memory は Windows で制約が多いため使わない)。

Windows の Smart App Control 下では公開直後の CuPy バイナリがブロックされることがある
(pyproject.toml の [tool.uv] 参照)。その場合 CuPy の import が失敗し ``auto`` は CPU に
フォールバックする。理由は ``cuda_status()`` で確認できる。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

DeviceKind = Literal["cpu", "cuda"]

_ENV_VAR = "ES_SIM_DEVICE"


@lru_cache(maxsize=1)
def cuda_status() -> tuple[bool, str]:
    """(CUDA が使えるか, 使えない場合の理由 / 使える場合は GPU 名) を返す (結果はキャッシュ)。"""
    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:  # ImportError / DLL ブロック (SAC) など
        return False, f"CuPy を import できません: {type(exc).__name__}: {exc}"
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return False, f"CUDA ランタイムを初期化できません: {type(exc).__name__}: {exc}"
    if n <= 0:
        return False, "CUDA デバイスが見つかりません"
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    except Exception:  # pragma: no cover - 名前取得だけの失敗は致命的でない
        name = "CUDA device 0"
    return True, name


def cuda_available() -> bool:
    return cuda_status()[0]


@dataclass(frozen=True)
class Device:
    """計算デバイス。``xp`` は numpy / cupy モジュール。"""

    kind: DeviceKind
    name: str

    @property
    def is_gpu(self) -> bool:
        return self.kind == "cuda"

    @property
    def xp(self) -> Any:
        if self.kind == "cuda":
            import cupy as cp

            return cp
        return np

    def asarray(self, a: Any, dtype: Any = None) -> Any:
        """ホスト/デバイスどちらの配列も、このデバイス上の配列にする。"""
        if self.kind == "cuda":
            import cupy as cp

            return cp.asarray(a, dtype=dtype)
        if hasattr(a, "get") and not isinstance(a, np.ndarray):  # cupy → numpy
            a = a.get()
        return np.asarray(a, dtype=dtype)

    def to_host(self, a: Any) -> np.ndarray:
        """numpy 配列にして返す (cupy ならデバイス→ホスト転送)。"""
        if hasattr(a, "get") and not isinstance(a, np.ndarray):
            return a.get()
        return np.asarray(a)

    def zeros(self, shape: Any, dtype: Any = np.float64) -> Any:
        return self.xp.zeros(shape, dtype=dtype)

    def empty(self, shape: Any, dtype: Any = np.float64) -> Any:
        return self.xp.empty(shape, dtype=dtype)

    def synchronize(self) -> None:
        if self.kind == "cuda":
            import cupy as cp

            cp.cuda.Device().synchronize()


_CPU = Device(kind="cpu", name="CPU (NumPy/Numba)")


def get_device(kind: str | None = None) -> Device:
    """計算デバイスを返す。

    kind: ``"auto"`` | ``"cpu"`` | ``"cuda"`` | None (None は環境変数 ES_SIM_DEVICE、
    未設定なら ``"auto"``)。``"cuda"`` を明示したのに使えない場合は RuntimeError。
    """
    k = (kind or os.environ.get(_ENV_VAR) or "auto").strip().lower()
    if k == "cpu":
        return _CPU
    if k in ("cuda", "gpu"):
        ok, info = cuda_status()
        if not ok:
            raise RuntimeError(f"CUDA デバイスを使えません ({info})")
        return Device(kind="cuda", name=info)
    if k == "auto":
        ok, info = cuda_status()
        return Device(kind="cuda", name=info) if ok else _CPU
    raise ValueError(f"不明なデバイス指定です: {kind!r} (auto / cpu / cuda)")


def describe() -> dict:
    """/health 等に載せる診断情報。"""
    ok, info = cuda_status()
    out: dict = {"cuda": ok, "cuda_info": info, "default": get_device().kind}
    if ok:
        import cupy as cp

        out["cupy"] = cp.__version__
        out["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
    return out


__all__ = ["Device", "DeviceKind", "cuda_available", "cuda_status", "describe", "get_device"]
