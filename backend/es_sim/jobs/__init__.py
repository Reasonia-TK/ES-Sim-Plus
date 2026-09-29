"""UI v2 のジョブ (prompts/130 P6d): 同じソルバーの同時実行・実行ごとの結果・停止・続き・イベントの多重化。"""

from .api import make_router
from .manager import JobError, JobManager
from .runners import Hooks, default_runners

__all__ = ["Hooks", "JobError", "JobManager", "default_runners", "make_router"]
