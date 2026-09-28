"""LXCat 形式断面積ファイルのパーサー (prompts/120)。

boltzpmp 0.5.0 (Rust 製 `lxcat.rs`、`bp.parse_lxcat`) の解釈を上位互換で再現し、拡張する。

## boltzpmp と同じ解釈 (標準ブロック形式)

- キーワード行: 前後空白を除いて完全一致・大文字の ELASTIC/EFFECTIVE/EXCITATION/
  IONIZATION/ATTACHMENT/ROTATION。2 行目は反応式 `A`・`A -> B`・`A <-> B`
  (`<->` は逆過程 = 超弾性の指定)。
- パラメータ行: ELASTIC/EFFECTIVE は 3 行目の先頭が数値なら m/M (省略可、数値でなければ
  行を消費しない)。EXCITATION は「閾値 [統計重み比 g_up/g_low]」、IONIZATION は閾値
  (必須)。ATTACHMENT は無し。ROTATION は 3・4 行目に下準位・上準位の「エネルギー 統計重み」。
  パラメータ行では `!`・`#` で始まるトークン以降をコメントとして無視する。
- 数値の文法は Rust の f64 と同じ (有限値のみ。`1_0` や `1.0D-5` は数値ではない)。
- テーブル: 「`-` のみ 5 文字以上」の行で挟む。空行は読み飛ばす。2 列 (eV, m²) で、3 列目が
  あれば運動量移行断面積 (このとき 2 列目は積分断面積)。3 列以上の行があるかどうかは全行で
  一致が必要。エネルギー非減少 (重複 = 段差)、値は有限・σ ≥ 0。
- テーブル前の `PROCESS:` (最後の値) と `COMMENT:` (改行で連結) を読む (キーは大文字小文字を
  区別しない)。改行は LF/CRLF、先頭の BOM は除去。

## 拡張 (boltzpmp は読まない・読み捨てる部分)

- `MOMENTUM` を EFFECTIVE の別名として受理 (旧 BOLSIG 形式、param["keyword_alias"])。
- ATTACHMENT の 3 行目が数値トークンのみの行なら閾値として受理 (param["threshold_line"])。
- EXCITATION/IONIZATION の閾値行が無くても PARAM.: の `E = x eV` があれば使う (警告)。
  IONIZATION の 3 行目の余分なトークンは param["extra_tokens"] に残す。
- テーブル前の全 "KEY: value" 行 (SPECIES/PARAM./UPDATED/COLUMNS/その他 → meta)、キーを
  持たない自由記述行は comment に追記。SPECIES 行から入射粒子 (projectile) を決める。
- PARAM.: の "key = value [unit]" を辞書化 (エネルギー単位 eV/meV/keV/MeV は eV に換算、
  他の単位は除去、数値でなければ文字列、`=` の無い要素は True)。パラメータ行の m/M・閾値と
  相対 1e-6 を超えて食い違えば警告 (パラメータ行を優先)。m/M 行が無ければ PARAM の m/M を採用。
- `COLUMNS:` の単位を SI (eV, m²) に換算 (meV/keV、cm²・Å²・1e-20 m² 等)。
- 4 列以上の行は警告して 3 列目まで使う。
- タイプ行なし形式 (Phelps イオンデータ): ブロック外の `SPECIES:` 行から始まり、`PROCESS:` の
  末尾語で ISOTROPIC/BACKSCAT を判定。
- `DATABASE:` 行でデータベースを切替え、ブロック外の "KEY: value" はその DB の meta に入れる。
- 旧 Mac の CR のみの改行、UTF-16 (BOM 付き) のファイルも読む。

## boltzpmp より厳しくした点 (壊れた入力を黙って誤読しないため)

- ブロックのテーブル開始区切りより前に次のキーワード行が現れたら ValueError
  (boltzpmp は次のブロックのテーブルを自分のものとして読み、次のブロックを失う)。
- 負のエネルギーは ValueError (boltzpmp は σ の負値のみ拒否)。
- ブロックが 1 つも無ければ ValueError (v1 と同じ。boltzpmp は空リスト)。

構造エラーは行番号 (1 始まり) を先頭に付けた ValueError ("N 行目: ...")、回復可能な問題は
`LxcatDocument.warnings` に行番号付きの日本語で積む。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import CrossSection, LxcatDatabase, LxcatDocument, ParamValue

#: 標準ブロックのキーワード (boltzpmp と同じ 6 種)
KEYWORDS: tuple[str, ...] = (
    "ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION",
)
#: キーワードの別名 (拡張)
KEYWORD_ALIASES: dict[str, str] = {"MOMENTUM": "EFFECTIVE"}

# タイプ行なし形式: PROCESS 行の末尾語 → 種別
_TAIL_KINDS = {"isotropic": "ISOTROPIC", "backscat": "BACKSCAT", "backscattering": "BACKSCAT"}

# Rust の f64::from_str と同じ文法 (inf/nan を除く)。\d は ASCII 数字のみにする
_NUMBER = r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_NUMBER_RE = re.compile(_NUMBER)
# "KEY: value" のキー (英字で始まる短い語句)
_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9 ._/()+\-]{0,39}")
# ブロック外で大文字小文字を問わず meta として受け付ける既知キー
_KNOWN_KEYS = {
    "DATABASE", "PERMLINK", "DESCRIPTION", "CONTACT", "HOW TO REFERENCE", "REFERENCE",
    "SPECIES", "PROCESS", "PARAM", "PARAM.", "COMMENT", "UPDATED", "COLUMNS",
}
# PARAM.: の値 "数値 [単位]"
_PARAM_VALUE_RE = re.compile(r"(" + _NUMBER + r")\s*(.*)", re.S)
_UNIT_RE = re.compile(r"[A-Za-zÅµμ°%][A-Za-z0-9Åµμ°%^/*.\-²³]*")
_PAREN_RE = re.compile(r"\(([^()]*)\)")

_ENERGY_UNITS = {"eV": 1.0, "meV": 1e-3, "keV": 1e3, "MeV": 1e6}
_SIGMA_UNITS = {"m2": 1.0, "cm2": 1e-4, "nm2": 1e-18, "å2": 1e-20, "a2": 1e-20, "angstrom2": 1e-20}


# ---- 字句レベルの補助 ----------------------------------------------------------------


def _split_lines(text: str) -> list[str]:
    """Rust の `str::lines` と同じ行分割 (LF・CRLF、末尾の改行で空行を作らない)。

    改行が CR だけのテキスト (旧 Mac 形式) は CR で分割する (拡張)。
    """
    if text.startswith("﻿"):
        text = text[1:]
    if "\n" not in text and "\r" in text:
        text = text.replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln[:-1] if ln.endswith("\r") else ln for ln in lines]


def _number(tok: str) -> float | None:
    """有限の数値トークンなら float、そうでなければ None (Rust の f64 と同じ文法)。"""
    if _NUMBER_RE.fullmatch(tok) is None:
        return None
    v = float(tok)
    return v if math.isfinite(v) else None


def _is_nonfinite_token(tok: str) -> bool:
    """inf/nan や桁あふれ (1e999) のトークンか。"""
    if tok.lower().lstrip("+-") in ("inf", "infinity", "nan"):
        return True
    return _NUMBER_RE.fullmatch(tok) is not None and not math.isfinite(float(tok))


def _is_dashed(line: str) -> bool:
    """テーブル区切り行 (前後空白を除いて `-` のみ 5 文字以上) か。"""
    s = line.strip()
    return len(s) >= 5 and s.count("-") == len(s)


def _param_tokens(line: str) -> list[str]:
    """パラメータ行のトークン (`!`・`#` で始まるトークン以降はコメントとして捨てる)。"""
    out: list[str] = []
    for tok in line.split():
        if tok.startswith(("!", "#")):
            break
        out.append(tok)
    return out


def _keyword(stripped: str) -> tuple[str | None, str | None]:
    """キーワード行なら (種別, 別名 or None)、そうでなければ (None, None)。"""
    if stripped in KEYWORDS:
        return stripped, None
    if stripped in KEYWORD_ALIASES:
        return KEYWORD_ALIASES[stripped], stripped
    return None, None


def split_reaction(reaction: str) -> tuple[str, str | None, bool]:
    """反応式 (2 行目) を (標的, 生成物 or None, 逆過程か) に分ける (boltzpmp と同じ)。"""
    if "<->" in reaction:
        left, right = reaction.split("<->", 1)
        return left.strip(), (right.strip() or None), True
    if "->" in reaction:
        left, right = reaction.split("->", 1)
        return left.strip(), (right.strip() or None), False
    return reaction.strip(), None, False


def _split_species(value: str) -> tuple[str, str | None]:
    """SPECIES 値 "e / Ar"・"Ar^+ / Ar" を (入射粒子, 標的) に分ける。"/" が無ければ標的 None。"""
    if "/" not in value:
        return value.strip(), None
    sep = " / " if " / " in value else "/"
    left, right = value.split(sep, 1)
    projectile = left.strip()
    if projectile.lower() == "e":
        projectile = "e"
    return projectile, right.strip()


def _split_key_value(stripped: str) -> tuple[str, str, str] | None:
    """"KEY: value" 行なら (元のキー, 正規化キー (大文字), 値)。最初の ":" で分割する。"""
    idx = stripped.find(":")
    if idx <= 0:
        return None
    raw_key = stripped[:idx].strip()
    value = stripped[idx + 1:].strip()
    if _KEY_RE.fullmatch(raw_key) is None or value.startswith("//"):  # "http://..." は除く
        return None
    return raw_key, raw_key.upper(), value


def _base_key(key: str) -> str:
    """"PARAM." → "PARAM" のように末尾の "." を除いたキー。"""
    return key.rstrip(".").strip()


def _energy_factor(unit: str) -> float | None:
    u = unit.strip().replace(" ", "")
    if u in _ENERGY_UNITS:
        return _ENERGY_UNITS[u]
    if u.lower() == "ev":
        return 1.0
    return None


def _sigma_factor(unit: str) -> float | None:
    """断面積の単位 → m² への換算係数 ("1e-20 m2"・"10^-20 m2"・"cm^2"・"Å2" 等)。"""
    s = unit.strip().replace(" ", "").replace("²", "2").replace("**", "^").replace("×", "*")
    s = s.replace("Å", "Å")  # オングストローム記号 → Å
    factor = 1.0
    m = re.match(r"10\^\(?([+-]?[0-9]+)\)?\*?", s)
    if m is not None:
        factor = float(f"1e{m.group(1)}")
        s = s[m.end():]
    else:
        m = re.match(r"(" + _NUMBER + r")\*?(?=[A-Za-zÅ])", s)
        if m is not None:
            factor = float(m.group(1))
            s = s[m.end():]
    base = s.replace("^", "").lower()
    if base in _SIGMA_UNITS and math.isfinite(factor) and factor > 0.0:
        return factor * _SIGMA_UNITS[base]
    return None


def _column_units(value: str) -> list[str | None]:
    """COLUMNS 値から列ごとの単位 (括弧内) を取り出す。"""
    parts = value.split("|")
    if len(parts) >= 2:
        out: list[str | None] = []
        for p in parts:
            found = _PAREN_RE.findall(p)
            out.append(found[-1] if found else None)
        return out
    found = _PAREN_RE.findall(value)
    return list(found) if found else [None]


def _param_value(raw: str) -> ParamValue:
    m = _PARAM_VALUE_RE.fullmatch(raw)
    if m is None:
        return raw
    v = float(m.group(1))
    if not math.isfinite(v):
        return raw
    unit = m.group(2).strip()
    if not unit:
        return v
    f = _energy_factor(unit)
    if f is not None:
        return v * f if f != 1.0 else v
    if _UNIT_RE.fullmatch(unit) is not None:
        return v
    return raw


def parse_param(value: str) -> dict[str, ParamValue]:
    """PARAM.: の値を辞書にする ("m/M = 0.0000136, complete set" → {"m/M": 1.36e-5, "complete set": True})。

    "," 区切りの各要素を `key = value [unit]` として数値化する (エネルギー単位は eV に換算、
    その他の単位は除去、数値化できなければ文字列)。`=` の無い要素は True。
    """
    out: dict[str, ParamValue] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            out[item] = True
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            out[item] = True
            continue
        out[key] = _param_value(raw.strip())
    return out


def _differs(a: float, b: float, rtol: float = 1e-6) -> bool:
    return abs(a - b) > rtol * max(abs(a), abs(b))


def _add_meta(meta: dict[str, str], key: str, value: str) -> None:
    meta[key] = f"{meta[key]}\n{value}" if key in meta else value


# ---- パーサー本体 -------------------------------------------------------------------


@dataclass
class _Header:
    """ブロックのテーブル前の "KEY: value" 行・自由記述行。"""

    species: str = ""
    process: str | None = None
    param: dict[str, ParamValue] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    updated: str | None = None
    columns: str | None = None
    columns_line: int = 0
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class _Table:
    energy: np.ndarray
    sigma: np.ndarray
    sigma_mt: np.ndarray | None
    next_index: int  # 終了区切りの次の行 (0 始まり)


class _Parser:
    def __init__(self, text: str) -> None:
        self.lines = _split_lines(text)
        self.n = len(self.lines)
        self.warnings: list[str] = []
        self.implicit_db = LxcatDatabase(name=None)
        self.named_dbs: list[LxcatDatabase] = []
        self.db = self.implicit_db
        self.n_blocks = 0
        self.last_meta_key: str | None = None

    # ---- ファイル全体 ----

    def run(self) -> LxcatDocument:
        lines = self.lines
        i = 0
        while i < self.n:
            raw = lines[i]
            st = raw.strip()
            kind, alias = _keyword(st)
            if kind is not None:
                self.last_meta_key = None
                i = self._standard_block(i, kind, alias)
                continue
            kv = _split_key_value(st)
            if kv is not None and (kv[1] in _KNOWN_KEYS or kv[0] == kv[0].upper()):
                _, key, value = kv
                base = _base_key(key)
                if base == "SPECIES":
                    self.last_meta_key = None
                    i = self._typeless_block(i, value)
                    continue
                if base == "DATABASE":
                    self._switch_database(value)
                    self.last_meta_key = None
                else:
                    _add_meta(self.db.meta, key, value)
                    self.last_meta_key = key
            elif st and raw[:1].isspace() and self.last_meta_key is not None and not _is_dashed(st):
                # DESCRIPTION: 等の字下げされた継続行
                self.db.meta[self.last_meta_key] += "\n" + st
            else:
                self.last_meta_key = None  # 説明文・xxxx/**** 区切り・見出し行は読み飛ばす
            i += 1

        if self.n_blocks == 0:
            raise ValueError(
                "LXCat の断面積ブロックが見つかりません "
                "(タイプ行 (ELASTIC/EXCITATION 等) または SPECIES:/PROCESS: ブロックが必要です)"
            )
        dbs = list(self.named_dbs)
        if self.implicit_db.cross_sections or not dbs:
            dbs.insert(0, self.implicit_db)
        return LxcatDocument(databases=dbs, warnings=self.warnings)

    def _switch_database(self, name: str) -> None:
        for db in self.named_dbs:
            if db.name == name:
                self.db = db
                return
        self.db = LxcatDatabase(name=name)
        self.named_dbs.append(self.db)

    # ---- ブロック共通 ----

    def _find_table_start(self, start: int, owner: str, owner_line: int) -> int | None:
        """start 以降でテーブル開始区切りの行 index を返す。

        先にキーワード行が現れたら None (呼び出し側で扱いを決める)。ファイル末尾まで
        無ければ ValueError。
        """
        k = start
        while k < self.n:
            if _is_dashed(self.lines[k]):
                return k
            if _keyword(self.lines[k].strip())[0] is not None:
                return None
            k += 1
        raise ValueError(
            f"{owner_line} 行目: {owner} の断面積テーブル (----- で挟まれた表) がありません"
        )

    def _read_header(self, start: int, stop: int) -> _Header:
        h = _Header()
        for k in range(start, stop):
            st = self.lines[k].strip()
            if not st:
                continue
            kv = _split_key_value(st)
            if kv is None:
                h.comments.append(st)  # キーを持たない自由記述行
                continue
            _, key, value = kv
            base = _base_key(key)
            if base == "SPECIES":
                h.species = value
            elif base == "PROCESS":
                h.process = value  # boltzpmp と同じく最後の値
            elif base == "PARAM":
                h.param.update(parse_param(value))
            elif base == "COMMENT":
                h.comments.append(value)
            elif base == "UPDATED":
                h.updated = value
            elif base == "COLUMNS":
                h.columns = value
                h.columns_line = k + 1
            else:
                _add_meta(h.meta, key, value)
        return h

    def _read_table(self, open_index: int, owner: str) -> _Table:
        """open_index (開始区切りの行) からテーブルを読み、行番号付きで検査する。"""
        lines = self.lines
        rows: list[tuple[int, list[float]]] = []
        k = open_index + 1
        while True:
            if k >= self.n:
                raise ValueError(
                    f"{open_index + 1} 行目: {owner} のテーブルの終了区切り (-----) がありません"
                )
            if _is_dashed(lines[k]):
                break
            st = lines[k].strip()
            if st:
                vals: list[float] = []
                for tok in st.split():
                    v = _number(tok)
                    if v is None:
                        if _is_nonfinite_token(tok):
                            raise ValueError(
                                f"{k + 1} 行目: {owner} のテーブルに有限でない値があります: '{tok}'"
                            )
                        raise ValueError(
                            f"{k + 1} 行目: {owner} のテーブル行を数値として解釈できません: '{st}'"
                        )
                    vals.append(v)
                if len(vals) < 2:
                    raise ValueError(
                        f"{k + 1} 行目: {owner} のテーブル行には 2 列以上の数値が必要です: '{st}'"
                    )
                rows.append((k + 1, vals))
            k += 1
        close_index = k
        if not rows:
            raise ValueError(f"{open_index + 1} 行目: {owner} のテーブルが空です")

        # 列数: 2 列か 3 列 (運動量移行) かは全行で一致 (boltzpmp と同じ)。4 列以上は 3 列目まで
        first_line, first_vals = rows[0]
        ncol = min(len(first_vals), 3)
        wide = False
        for line_no, vals in rows:
            if min(len(vals), 3) != ncol:
                raise ValueError(
                    f"{line_no} 行目: {owner} のテーブルの列数が一致しません "
                    f"({first_line} 行目は {len(first_vals)} 列、この行は {len(vals)} 列。"
                    "2 列 (エネルギー, 断面積) か 3 列 (+運動量移行断面積) に揃えてください)"
                )
            wide = wide or len(vals) > 3
        if wide:
            self.warnings.append(
                f"{open_index + 1} 行目: {owner} のテーブルに 4 列以上の行があります。"
                "3 列目 (運動量移行断面積) までを使い、残りは無視します"
            )

        prev = -math.inf
        for line_no, vals in rows:
            e = vals[0]
            if e < 0.0:
                raise ValueError(f"{line_no} 行目: {owner} のエネルギーが負です: {e!r}")
            if e < prev:
                raise ValueError(
                    f"{line_no} 行目: {owner} のエネルギーが減少しています ({prev!r} の後に {e!r})"
                )
            for v in vals[1:ncol]:
                if v < 0.0:
                    raise ValueError(f"{line_no} 行目: {owner} の断面積が負です: {v!r}")
            prev = e

        arr = np.array([vals[:ncol] for _, vals in rows], dtype=np.float64)
        return _Table(
            energy=np.ascontiguousarray(arr[:, 0]),
            sigma=np.ascontiguousarray(arr[:, 1]),
            sigma_mt=np.ascontiguousarray(arr[:, 2]) if ncol == 3 else None,
            next_index=close_index + 1,
        )

    def _apply_columns(self, table: _Table, h: _Header, owner: str) -> None:
        """COLUMNS: の単位で表を SI (eV, m²) に換算する (無ければ eV・m² とみなす)。"""
        if h.columns is None:
            return
        units = _column_units(h.columns)
        e_unit = units[0]
        s_unit = units[1] if len(units) > 1 else None
        mt_unit = units[2] if len(units) > 2 else None

        def factor(unit: str | None, conv, what: str, assumed: str) -> float:
            f = conv(unit) if unit is not None else None
            if f is None:
                self.warnings.append(
                    f"{h.columns_line} 行目: {owner} の COLUMNS の{what}の単位 "
                    f"'{unit if unit is not None else h.columns}' を解釈できません。{assumed} とみなします"
                )
                return 1.0
            return f

        fe = factor(e_unit, _energy_factor, "エネルギー", "eV")
        fs = factor(s_unit, _sigma_factor, "断面積", "m²")
        if table.sigma_mt is not None and mt_unit is not None:
            fmt = factor(mt_unit, _sigma_factor, "運動量移行断面積", "m²")
        else:
            fmt = fs
        if fe != 1.0:
            table.energy = table.energy * fe
        if fs != 1.0:
            table.sigma = table.sigma * fs
        if table.sigma_mt is not None and fmt != 1.0:
            table.sigma_mt = table.sigma_mt * fmt

    def _skip_table(self, open_index: int, owner: str) -> int:
        """テーブルを検査せず読み飛ばし、終了区切りの次の行 index を返す。"""
        k = open_index + 1
        while k < self.n:
            if _is_dashed(self.lines[k]):
                return k + 1
            k += 1
        raise ValueError(
            f"{open_index + 1} 行目: {owner} のテーブルの終了区切り (-----) がありません"
        )

    # ---- 標準ブロック (タイプ行あり) ----

    def _standard_block(self, i: int, kind: str, alias: str | None) -> int:
        lines, n = self.lines, self.n
        kw_line = i + 1
        kw_word = lines[i].strip()
        j = i + 1
        reaction = lines[j].strip() if j < n else ""
        if not reaction or _is_dashed(reaction):
            raise ValueError(f"{kw_line} 行目: {kw_word} ブロックに反応式の行 (2 行目) がありません")
        target, product, reversible = split_reaction(reaction)
        owner = f"{kw_word} ブロック ({reaction})"
        j += 1

        flags: dict[str, ParamValue] = {}
        if alias is not None:
            flags["keyword_alias"] = alias
        threshold = 0.0
        mass_ratio: float | None = None
        weight_ratio: float | None = None
        lower: tuple[float, float] | None = None
        upper: tuple[float, float] | None = None
        threshold_from_line = False     # 閾値をパラメータ行から得たか
        missing_threshold_line = 0      # 閾値行が無かったときの 3 行目の行番号 (1 始まり)

        if kind in ("ELASTIC", "EFFECTIVE"):
            toks = _param_tokens(lines[j]) if j < n else []
            v = _number(toks[0]) if toks else None
            if v is not None:  # 質量比の行は省略可 (数値でなければ行を消費しない)
                if not 0.0 < v < 1.0:
                    raise ValueError(
                        f"{j + 1} 行目: {owner} の質量比 m/M = {toks[0]} は 0 < m/M < 1 の範囲外です"
                    )
                mass_ratio = v
                j += 1
        elif kind in ("EXCITATION", "IONIZATION"):
            line = lines[j] if j < n else ""
            toks = _param_tokens(line)
            v = _number(toks[0]) if toks else None
            if v is None:
                missing_threshold_line = j + 1  # PARAM.: の E で補えるか後で判定 (行は消費しない)
            else:
                if v < 0.0:
                    raise ValueError(f"{j + 1} 行目: {owner} の閾値が負です: {toks[0]}")
                threshold = v
                threshold_from_line = True
                rest = toks[1:]
                if kind == "EXCITATION" and rest:
                    w = _number(rest[0])
                    if w is None:
                        raise ValueError(
                            f"{j + 1} 行目: {owner} の統計重み比を解釈できません: '{rest[0]}'"
                        )
                    if not w > 0.0:
                        raise ValueError(
                            f"{j + 1} 行目: {owner} の統計重み比は正の値が必要です: {rest[0]}"
                        )
                    weight_ratio = w
                    rest = rest[1:]
                if rest:
                    flags["extra_tokens"] = " ".join(rest)
                j += 1
        elif kind == "ATTACHMENT":
            # LXCat 仕様ではパラメータ行なし。拡張: コロンを含まず数値トークンのみの行は閾値
            if j < n and ":" not in lines[j]:
                toks = _param_tokens(lines[j])
                vals = [_number(t) for t in toks]
                if toks and all(v is not None for v in vals):
                    if vals[0] < 0.0:  # type: ignore[operator]
                        raise ValueError(f"{j + 1} 行目: {owner} の閾値が負です: {toks[0]}")
                    threshold = float(vals[0])  # type: ignore[arg-type]
                    threshold_from_line = True
                    flags["threshold_line"] = True
                    if len(toks) > 1:
                        flags["extra_tokens"] = " ".join(toks[1:])
                    j += 1
        elif kind == "ROTATION":
            states: list[tuple[float, float]] = []
            for which in ("下準位", "上準位"):
                line = lines[j] if j < n else ""
                toks = _param_tokens(line)
                ev = _number(toks[0]) if len(toks) >= 1 else None
                gv = _number(toks[1]) if len(toks) >= 2 else None
                if ev is None or gv is None:
                    raise ValueError(
                        f"{j + 1} 行目: {owner} の{which}を 'エネルギー[eV] 統計重み' の 2 数で"
                        f"与えてください: '{line.strip()}'"
                    )
                if not gv > 0.0:
                    raise ValueError(
                        f"{j + 1} 行目: {owner} の{which}の統計重みは正の値が必要です: {toks[1]}"
                    )
                states.append((ev, gv))
                j += 1
            lower, upper = states
            gap = upper[0] - lower[0]
            if not gap > 0.0:
                raise ValueError(
                    f"{j} 行目: {owner} の上準位 ({upper[0]!r} eV) が下準位 "
                    f"({lower[0]!r} eV) より高くありません"
                )
            threshold = gap
            weight_ratio = upper[1] / lower[1]

        # テーブル前のヘッダ行
        table_open = self._find_table_start(j, owner, kw_line)
        if table_open is None:
            k = j
            while _keyword(lines[k].strip())[0] is None:
                k += 1
            raise ValueError(
                f"{kw_line} 行目: {owner} の断面積テーブルが無いまま {k + 1} 行目で次のブロック "
                f"({lines[k].strip()}) が始まりました"
            )
        h = self._read_header(j, table_open)

        # PARAM.: との突き合わせ・補完
        p_e = h.param.get("E")
        p_mm = h.param.get("m/M")
        if missing_threshold_line:
            if isinstance(p_e, float) and p_e < 0.0:
                raise ValueError(
                    f"{missing_threshold_line} 行目: {owner} の 3 行目に閾値が無く、PARAM.: の "
                    f"E = {p_e:g} eV は負のため閾値に使えません"
                )
            if isinstance(p_e, float):
                threshold = p_e
                flags["threshold_from_param"] = True
                self.warnings.append(
                    f"{missing_threshold_line} 行目: {owner} の 3 行目に閾値がないため、"
                    f"PARAM.: の E = {p_e:g} eV を閾値として使いました"
                )
            else:
                found = lines[missing_threshold_line - 1].strip()  # ヘッダ内の行なので必ず存在する
                raise ValueError(
                    f"{missing_threshold_line} 行目: {owner} の 3 行目に閾値 [eV] がありません "
                    f"(PARAM.: の E = ... もありません): '{found}'"
                )
        elif threshold_from_line and isinstance(p_e, float) and _differs(threshold, p_e):
            self.warnings.append(
                f"{kw_line} 行目: {owner} のパラメータ行の閾値 {threshold:g} eV と PARAM.: の "
                f"E = {p_e:g} eV が一致しません (パラメータ行を優先)"
            )
        if kind in ("ELASTIC", "EFFECTIVE") and isinstance(p_mm, float):
            if mass_ratio is not None:
                if _differs(mass_ratio, p_mm):
                    self.warnings.append(
                        f"{kw_line} 行目: {owner} のパラメータ行の m/M = {mass_ratio:g} と PARAM.: の "
                        f"m/M = {p_mm:g} が一致しません (パラメータ行を優先)"
                    )
            elif 0.0 < p_mm < 1.0:
                mass_ratio = p_mm
                flags["mass_ratio_from_param"] = True
            else:
                self.warnings.append(
                    f"{kw_line} 行目: {owner} の PARAM.: の m/M = {p_mm:g} は 0 < m/M < 1 の"
                    "範囲外のため使いません"
                )

        projectile = "e"
        if h.species:
            sp_projectile, sp_target = _split_species(h.species)
            if sp_target is None:
                sp_target = sp_projectile  # "SPECIES: Ar" (入射粒子の記載なし) は電子とみなす
            else:
                projectile = sp_projectile or "e"
            if sp_target != target:
                self.warnings.append(
                    f"{kw_line} 行目: {owner} の SPECIES の標的 '{sp_target}' と反応式の標的 "
                    f"'{target}' が異なります (反応式を優先)"
                )

        table = self._read_table(table_open, owner)
        self._apply_columns(table, h, owner)
        param = dict(h.param)
        param.update(flags)
        cs = CrossSection(
            kind=kind,  # type: ignore[arg-type]
            projectile=projectile,
            target=target,
            product=product,
            reversible=reversible,
            threshold_ev=threshold,
            mass_ratio=mass_ratio,
            weight_ratio=weight_ratio,
            lower_state=lower,
            upper_state=upper,
            energy_ev=table.energy,
            sigma_m2=table.sigma,
            sigma_mt_m2=table.sigma_mt,
            process=h.process or "",
            species=h.species,
            param=param,
            comment="\n".join(h.comments),
            updated=h.updated,
            columns=h.columns,
            database=self.db.name,
            line=kw_line,
            meta=h.meta,
        )
        self.db.cross_sections.append(cs)
        self.n_blocks += 1
        return table.next_index

    # ---- タイプ行なしブロック (Phelps イオンデータ) ----

    def _typeless_block(self, i: int, species_value: str) -> int:
        sp_line = i + 1
        owner = f"SPECIES ブロック ({species_value})"
        table_open = self._find_table_start(i + 1, owner, sp_line)
        if table_open is None:
            # テーブルより先に標準ブロックが始まる: boltzpmp と同じくこの SPECIES 行は無視する
            self.warnings.append(
                f"{sp_line} 行目: SPECIES 行 ({species_value}) の後にテーブルが無いまま"
                "標準ブロックが始まったため、この SPECIES 行は無視しました"
            )
            return i + 1
        h = self._read_header(i, table_open)
        self.n_blocks += 1
        if not h.process:  # 値が空の PROCESS: も無いのと同じ (v1 と同じ)
            raise ValueError(f"{sp_line} 行目: {owner} に PROCESS: 行がありません")
        words = h.process.replace(",", " ").split()
        tail = words[-1].strip(".").lower() if words else ""
        kind = _TAIL_KINDS.get(tail)
        if kind is None:
            self.warnings.append(
                f"{sp_line} 行目: 未知のプロセス種 '{tail}' のためスキップしました: {h.process}"
            )
            return self._skip_table(table_open, owner)

        projectile, target = _split_species(h.species)
        if target is None:
            self.warnings.append(
                f"{sp_line} 行目: SPECIES '{h.species}' に '/' が無いため入射粒子が不明です "
                "(標的名として扱います)"
            )
            projectile, target = "", h.species.strip()
        table = self._read_table(table_open, owner)
        self._apply_columns(table, h, owner)
        cs = CrossSection(
            kind=kind,  # type: ignore[arg-type]
            projectile=projectile,
            target=target,
            energy_ev=table.energy,
            sigma_m2=table.sigma,
            sigma_mt_m2=table.sigma_mt,
            process=h.process,
            species=h.species,
            param=dict(h.param),
            comment="\n".join(h.comments),
            updated=h.updated,
            columns=h.columns,
            database=self.db.name,
            line=sp_line,
            meta=h.meta,
        )
        self.db.cross_sections.append(cs)
        return table.next_index


# ---- 公開 API -----------------------------------------------------------------------


def parse_lxcat_document(text: str) -> LxcatDocument:
    """LXCat 形式テキストを解釈して LxcatDocument を返す。

    構造エラーは行番号付きの ValueError。ブロックが 1 つも無い場合も ValueError。
    """
    return _Parser(text).run()


def decode_lxcat_bytes(data: bytes) -> str:
    """ファイルのバイト列をテキストにする: UTF-16 (BOM 付き) → UTF-8 (BOM 可) → Latin-1。"""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def load_lxcat(path: str | Path) -> LxcatDocument:
    """LXCat 形式ファイルを読む (UTF-8 (BOM 除去) で読めなければ Latin-1、CRLF 可)。"""
    return parse_lxcat_document(decode_lxcat_bytes(Path(path).read_bytes()))
