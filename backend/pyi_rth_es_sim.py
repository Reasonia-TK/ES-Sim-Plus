"""PyInstaller の実行時フック (es_sim_server.spec、prompts/133 P8b)。

CuPy が同梱の NVRTC (_internal/cuda/bin/x64) を使うように CUDA_PATH をそこへ向ける。CuPy は import のときに
CUDA_PATH/bin と bin/x64 を DLL の探索先に足し、NVRTC はそこから読まれる。利用者の PC の CUDA Toolkit の
有無・版によらず、テストした NVRTC を使う (このプロセスの環境変数だけを変える)。
"""

import os
import sys

_cuda = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "cuda")
if os.path.isdir(os.path.join(_cuda, "bin", "x64")):
    os.environ["CUDA_PATH"] = _cuda
