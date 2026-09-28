"""LXCat 形式断面積ファイルの v1 互換パーサー (prompts/19 → prompts/120 で置換)。

`parse_lxcat(text, species)` のシグネチャと戻り値 (v1 の XsProcess 一覧 + 警告) は
prompts/19 のまま、中身は v2 の断面積パッケージ `es_sim.xs` のラッパー:

    parse_lxcat(text, species) = to_v1_processes(parse_lxcat_document(text), species)

- パーサー (`es_sim.xs.lxcat`) は boltzpmp の LXCat パーサーの上位互換で、標準ブロック形式
  (ELASTIC/EFFECTIVE/EXCITATION/IONIZATION/ATTACHMENT/ROTATION、MOMENTUM は EFFECTIVE の
  別名) とタイプ行なし形式 (Phelps イオンデータの SPECIES:/PROCESS: ブロック) の両方を読む。
  PARAM.:・COLUMNS: の単位・3 列目 (運動量移行)・DATABASE ごとのグループ化も解釈する。
- v1 への変換 (`es_sim.xs.convert.to_v1_processes`) は v1 MCC が扱える種別だけを返す:
  species="electron" は elastic/excitation/ionization (ROTATION は excitation として)、
  "ion" は isotropic/backscat。ATTACHMENT や種別フィルタに合わないブロックは警告付きで
  スキップする。表の 3 列目 (運動量移行断面積) の欄は v1 に無いので、非弾性では捨て、
  ELASTIC では等方散乱の v1 MCC に正しい運動量移行断面積の側を elastic の σ にする (警告)。
- **prompts/120 の意図的な挙動変更 (物理の修正)**: 電子の標的ごとに運動量移行断面積を
  1 系統に整理する。ELASTIC と EFFECTIVE の両方があれば EFFECTIVE を除外 (v1 は両方を
  elastic として取り込み運動量移行を二重計上していた)、EFFECTIVE しか無ければ LXCat/BOLSIG+
  の定義どおり同じ標的の非弾性を差し引いて elastic に変換する (v1 は無変換で elastic 扱い)。
  どちらも警告に明記する。
- 構造が壊れている場合 (テーブル区切りなし・列数不一致・エネルギー減少等) やブロックが
  1 つも無い場合は行番号付きの ValueError (server.py が 422 に変換する)。
"""

from __future__ import annotations

from .schema import XsProcess
from .xs.convert import to_v1_processes
from .xs.lxcat import parse_lxcat_document


def parse_lxcat(text: str, species: str) -> tuple[list[XsProcess], list[str]]:
    """LXCat 形式テキストをパースして (processes, warnings) を返す。

    ブロックが 1 つも見つからない場合は ValueError を送出する
    (フィルタで全てスキップされた場合は空リスト + 警告)。
    """
    if species not in ("electron", "ion"):
        raise ValueError(f"species は 'electron' か 'ion' を指定してください: '{species}'")
    return to_v1_processes(parse_lxcat_document(text), species)  # type: ignore[arg-type]
