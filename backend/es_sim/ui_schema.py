"""UI v2 (ui/) に同梱するプロジェクトの JSON Schema を書き出す (prompts/130、P6b)。

UI は接続中のバックエンドの /v2/schema を使うが、未接続でも設定フォームを出せるよう同じ内容を
ui/src/schema/project.schema.json に同梱する。schema.py を変えたら書き直すこと
(tests/test_v2_ui_api.py が古いままだと失敗する):

    .venv\\Scripts\\python -m es_sim.ui_schema
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import __version__
from .schema import Project

SNAPSHOT = Path(__file__).resolve().parents[2] / "ui" / "src" / "schema" / "project.schema.json"


def project_schema() -> dict:
    return Project.model_json_schema()


def snapshot_text() -> str:
    """同梱用の JSON (版を含めない: 版の更新だけで差分が出ないように)。項目の並びはモデルの宣言順のまま
    (フォームの項目の順になるので並べ替えない。pydantic の出力の順は決まっている)。"""
    return json.dumps(project_schema(), ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0]) if args else SNAPSHOT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(snapshot_text(), encoding="utf-8", newline="\n")
    print(f"wrote {out} (es_sim {__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
