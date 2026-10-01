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

GPU を使う条件 (``cuda_status()`` が順に確かめる、prompts/133): CuPy が import できる、NVIDIA のドライバが
CuPy の CUDA と同じメジャー版 (CUDA 13 = R580 以降) に対応している、Compute Capability 7.5 以降
(CUDA 13 の対象 = Turing 以降)、NVRTC で小さなカーネルをコンパイルして実行できる (配布版は NVRTC を同梱)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

DeviceKind = Literal["cpu", "cuda"]

_ENV_VAR = "ES_SIM_DEVICE"

#: CUDA 13 が対象とする GPU の Compute Capability の下限 (7.5 = Turing。Maxwell・Pascal・Volta は対象外)
MIN_COMPUTE_CAPABILITY = (7, 5)
#: CUDA のメジャー版ごとのドライバの下限 (Windows、案内用)
_MIN_DRIVER = {13: "R580"}


def _cuda_version_text(v: int) -> str:
    return f"{v // 1000}.{(v % 1000) // 10}"


@lru_cache(maxsize=1)
def cuda_status() -> tuple[bool, str]:
    """(CUDA が使えるか, 使えない場合の理由 / 使える場合は GPU 名) を返す (結果はキャッシュ)。"""
    try:
        import cupy as cp  # noqa: F401
    except Exception as exc:  # ImportError / DLL ブロック (SAC) など
        return False, f"CuPy を import できません: {type(exc).__name__}: {exc}"
    try:
        driver = int(cp.cuda.runtime.driverGetVersion())
        runtime = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as exc:
        return False, f"NVIDIA のドライバが見つかりません: {type(exc).__name__}: {exc}"
    if driver // 1000 < runtime // 1000:
        major = runtime // 1000
        need = f"CUDA {major} に対応したドライバ" + (f" ({_MIN_DRIVER[major]} 以降)" if major in _MIN_DRIVER else "")
        if driver <= 0:
            return False, f"NVIDIA のドライバが見つかりません ({need}が必要です)"
        return False, (
            f"NVIDIA のドライバが古いため GPU を使えません (ドライバは CUDA {_cuda_version_text(driver)} まで。"
            f"{need}が必要です)"
        )
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return False, f"CUDA ランタイムを初期化できません: {type(exc).__name__}: {exc}"
    if n <= 0:
        return False, "CUDA デバイスが見つかりません"
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        cc = (int(props["major"]), int(props["minor"]))
    except Exception:  # pragma: no cover - 名前取得だけの失敗は致命的でない
        name, cc = "CUDA device 0", MIN_COMPUTE_CAPABILITY
    if cc < MIN_COMPUTE_CAPABILITY:
        return False, (
            f"{name} (Compute Capability {cc[0]}.{cc[1]}) は CUDA {runtime // 1000} の対象外です "
            f"({MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]} 以降が必要)"
        )
    try:
        # NVRTC・CuPy のヘッダ・カーネルの起動までを確かめる (配布版で同梱が欠けていても CPU で動くように)
        ok = float((cp.arange(4, dtype=cp.float64) * 2.0).sum()) == 12.0
    except Exception as exc:
        return False, f"GPU ({name}) でカーネルを実行できません: {type(exc).__name__}: {exc}"
    if not ok:  # pragma: no cover
        return False, f"GPU ({name}) の計算の確認に失敗しました"
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
        out["cuda_driver"] = int(cp.cuda.runtime.driverGetVersion())
        out["nvrtc"] = ".".join(str(v) for v in cp.cuda.nvrtc.getVersion())
        props = cp.cuda.runtime.getDeviceProperties(0)
        out["compute_capability"] = f"{int(props['major'])}.{int(props['minor'])}"
    return out


__all__ = ["Device", "DeviceKind", "cuda_available", "cuda_status", "describe", "get_device"]
