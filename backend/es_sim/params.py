"""パラメータと式の束縛 (CAD v2 P7f、prompts/132)。

文書の ``params`` は名前付きの式 (``vars``) と、設定の数値の欄に付けた式 (``bindings``) を持つ。文書の各欄には
計算した数値も入っているので、ソルバーはそのまま動く (ここは使わない)。パラメータをスイープするときや、式から
数値を書き直すときに ``apply_params`` で計算して欄に書く。UI (ui/src/model/expr.ts) と同じ文法で、
共通の試験データ (tests/data/param_expr_cases.json) で両方を確かめる。

式の文法 (値は SI):

- 数と単位: ``2 mm``・``13.56MHz``・``10 mTorr``・``1e10 cm^-3`` (単位は数かかっこの直後だけ。単位の指数も書ける)
- 四則演算 ``+ - * /``、べき ``^`` (右結合、単項のマイナスより強い: ``-2^2 = -4``)、かっこ
- 関数 ``sqrt exp ln log log10 sin cos tan abs``、定数 ``pi``
- パラメータの名前 (英字か _ で始まり英数字と _。関数・pi・単位の記号は名前にできない)

束縛の場所 (``path``) は文書の中の道筋で、各段は辞書のキー (文字列)・配列の番号 (整数)・配列の中の要素の選び方
(``{"id": "anode"}`` = id が anode の要素、``{"edge": "e3"}`` = ドメインの辺 e3 を含む境界条件) のどれか。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any

# ---- 単位 (記号 → SI への倍率) ----

_PREFIX = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "c": 1e-2, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
# SI 接頭辞を付けられる記号 (g は kg の g)
_PREFIXABLE = {"m": 1.0, "s": 1.0, "Hz": 1.0, "V": 1.0, "A": 1.0, "C": 1.0, "K": 1.0, "T": 1.0, "Pa": 1.0, "eV": 1.0, "J": 1.0, "W": 1.0, "N": 1.0, "F": 1.0, "g": 1e-3}
# 接頭辞を付けない記号
_PLAIN = {"Torr": 133.322368, "torr": 133.322368, "mTorr": 0.133322368, "mtorr": 0.133322368, "bar": 1e5, "mbar": 100.0, "sccm": 1.0}


def unit_factor(sym: str) -> float | None:
    """単位の記号 (指数なし) の SI への倍率。単位でなければ None。c は m にだけ付く (cm)。"""
    if sym in _PLAIN:
        return _PLAIN[sym]
    if sym in _PREFIXABLE:
        return _PREFIXABLE[sym]
    if len(sym) >= 2 and sym[0] in _PREFIX:
        base = sym[1:]
        if base in _PREFIXABLE and (sym[0] != "c" or base == "m"):
            return _PREFIX[sym[0]] * _PREFIXABLE[base]
    return None


FUNCS: dict[str, Callable[[float], float]] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log10,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
}
CONSTS = {"pi": math.pi}
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ExprError(ValueError):
    """式を計算できない。kind: "syntax" (読めない)・"unknown" (知らない名前)・"value" (値が有限でない)・"cycle" (循環)"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def name_problem(name: str) -> str | None:
    """パラメータ名に使えない理由 (使えれば None)。"""
    if not NAME_RE.match(name):
        return "英字か _ で始まり、英数字と _ だけの名前にしてください"
    if name in FUNCS or name in CONSTS:
        return f"{name} は関数・定数の名前なので使えません"
    if unit_factor(name) is not None:
        return f"{name} は単位の記号なので使えません"
    return None


# ---- 字句と構文 ----

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_µμ][A-Za-z0-9_]*)|(?P<op>[-+*/^()]))"
)


def _tokens(src: str) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    pos = 0
    s = src.replace("×", "*").replace("÷", "/").replace("π", "pi")
    while pos < len(s):
        if s[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(s, pos)
        if not m or m.end() == pos:
            raise ExprError("syntax", f"式を読めません ({s[pos:].strip()[:10]!r} の所)")
        kind = m.lastgroup or ""
        out.append((kind, m.group(kind), m.start(kind)))
        pos = m.end()
    return out


# 構文木: ("num", 値) / ("name", 名前) / ("neg", a) / ("bin", 演算子, a, b) / ("pow", a, b) / ("call", 関数, a)
Node = tuple


class _Parser:
    def __init__(self, src: str):
        self.toks = _tokens(src)
        self.i = 0

    def peek(self, k: int = 0) -> tuple[str, str, int] | None:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def take(self) -> tuple[str, str, int]:
        t = self.toks[self.i]
        self.i += 1
        return t

    def is_op(self, ch: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t[0] == "op" and t[1] == ch

    def parse(self) -> Node:
        if not self.toks:
            raise ExprError("syntax", "式が空です")
        v = self.expr()
        if self.i != len(self.toks):
            raise ExprError("syntax", f"式を読めません ({self.toks[self.i][1]!r} の所)")
        return v

    def expr(self) -> Node:
        v = self.term()
        while self.is_op("+") or self.is_op("-"):
            op = self.take()[1]
            v = ("bin", op, v, self.term())
        return v

    def term(self) -> Node:
        v = self.unary()
        while self.is_op("*") or self.is_op("/"):
            op = self.take()[1]
            v = ("bin", op, v, self.unary())
        return v

    def unary(self) -> Node:
        if self.is_op("-") or self.is_op("+"):
            neg = self.take()[1] == "-"
            v = self.unary()
            return ("neg", v) if neg else v
        return self.power()

    def power(self) -> Node:
        b = self.postfix()
        if self.is_op("^"):
            self.take()
            return ("pow", b, self.unary())
        return b

    def unit(self) -> float | None:
        """数・かっこの直後の単位 (と指数 ^n / ^-n)。単位でなければ None (読み進めない)。"""
        t = self.peek()
        if t is None or t[0] != "name":
            return None
        f = unit_factor(t[1])
        if f is None:
            return None
        self.take()
        if self.is_op("^"):
            # 単位の指数: ^整数 か ^-整数 (^( は式のべきなので単位の指数ではない)
            neg = self.is_op("-", 1)
            nt = self.peek(2 if neg else 1)
            if nt is not None and nt[0] == "num" and re.fullmatch(r"\d+", nt[1]):
                self.i += 3 if neg else 2
                n = int(nt[1])
                return f ** (-n if neg else n)
        return f

    def postfix(self) -> Node:
        t = self.peek()
        if t is None:
            raise ExprError("syntax", "式が途中で終わっています")
        if t[0] == "num":
            self.take()
            u = self.unit()
            return ("num", float(t[1]) * (u if u is not None else 1.0))
        if self.is_op("("):
            self.take()
            v = self.expr()
            if not self.is_op(")"):
                raise ExprError("syntax", "かっこが閉じていません")
            self.take()
            u = self.unit()
            return ("bin", "*", v, ("num", u)) if u is not None else v
        if t[0] == "name":
            self.take()
            name = t[1]
            if name in FUNCS:
                if not self.is_op("("):
                    raise ExprError("syntax", f"関数 {name} の後に ( が要ります")
                self.take()
                a = self.expr()
                if not self.is_op(")"):
                    raise ExprError("syntax", "かっこが閉じていません")
                self.take()
                return ("call", name, a)
            if name in CONSTS:
                return ("num", CONSTS[name])
            return ("name", name)
        raise ExprError("syntax", f"式を読めません ({t[1]!r} の所)")


def parse(src: str) -> Node:
    """式 → 構文木 (読めなければ ExprError("syntax"))。"""
    return _Parser(src).parse()


def _names(node: Node, out: set[str]) -> None:
    kind = node[0]
    if kind == "name":
        out.add(node[1])
    elif kind == "neg":
        _names(node[1], out)
    elif kind in ("bin",):
        _names(node[2], out)
        _names(node[3], out)
    elif kind == "pow":
        _names(node[1], out)
        _names(node[2], out)
    elif kind == "call":
        _names(node[2], out)


def _eval(node: Node, values: Mapping[str, float]) -> float:
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "name":
        if node[1] not in values:
            if unit_factor(node[1]) is not None:
                raise ExprError("unknown", f"{node[1]} は単位の記号です (単位は数かかっこの直後にだけ書けます)")
            raise ExprError("unknown", f"{node[1]} というパラメータはありません")
        return float(values[node[1]])
    if kind == "neg":
        return -_eval(node[1], values)
    if kind == "bin":
        a = _eval(node[2], values)
        b = _eval(node[3], values)
        if node[1] == "+":
            return a + b
        if node[1] == "-":
            return a - b
        if node[1] == "*":
            return a * b
        if b == 0.0:
            raise ExprError("value", "0 で割っています")
        return a / b
    if kind == "pow":
        a = _eval(node[1], values)
        b = _eval(node[2], values)
        try:
            r = a**b
        except (OverflowError, ZeroDivisionError) as exc:
            raise ExprError("value", "べき乗を計算できません") from exc
        if isinstance(r, complex):
            raise ExprError("value", "べき乗を計算できません")
        return float(r)
    if kind == "call":
        a = _eval(node[2], values)
        try:
            return float(FUNCS[node[1]](a))
        except (ValueError, OverflowError) as exc:
            raise ExprError("value", f"{node[1]}({a:g}) を計算できません") from exc
    raise AssertionError(kind)


def evaluate(src: str, values: Mapping[str, float] | None = None) -> float:
    """式の値 (SI)。values はパラメータの値。計算できなければ ExprError。"""
    v = _eval(parse(src), values or {})
    if not math.isfinite(v):
        raise ExprError("value", "値が有限ではありません")
    return v


def names_in(src: str) -> set[str]:
    """式に出てくるパラメータの名前 (関数・定数・単位は除く)。読めなければ ExprError。"""
    out: set[str] = set()
    _names(parse(src), out)
    return out


def evaluate_params(vars_: list[Mapping[str, Any]], overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """パラメータを依存の順に計算した値 (名前 → SI)。overrides の名前は式の代わりにその値。

    名前の重複・使えない名前・知らない名前・循環・計算できない式は ExprError (メッセージにパラメータ名)。
    """
    over = dict(overrides or {})
    exprs: dict[str, str] = {}
    for v in vars_:
        name = str(v.get("name", ""))
        prob = name_problem(name)
        if prob:
            raise ExprError("syntax", f"パラメータ名 {name!r}: {prob}")
        if name in exprs:
            raise ExprError("syntax", f"パラメータ名 {name} が重複しています")
        exprs[name] = str(v.get("expr", ""))
    for name in over:
        if name not in exprs:
            raise ExprError("unknown", f"{name} というパラメータはありません")
    deps: dict[str, set[str]] = {}
    for name, src in exprs.items():
        if name in over:
            deps[name] = set()
            continue
        try:
            deps[name] = names_in(src)
        except ExprError as exc:
            raise ExprError(exc.kind, f"パラメータ {name}: {exc}") from exc
        for d in deps[name]:
            if d not in exprs:
                raise ExprError("unknown", f"パラメータ {name}: {d} というパラメータはありません")
    values: dict[str, float] = {}
    state: dict[str, int] = {}  # 1: 計算中、2: 済み

    def visit(name: str, chain: list[str]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            cyc = chain[chain.index(name):] + [name]
            raise ExprError("cycle", "パラメータが循環しています (" + " → ".join(cyc) + ")")
        state[name] = 1
        for d in sorted(deps[name]):
            visit(d, [*chain, name])
        if name in over:
            values[name] = float(over[name])
        else:
            try:
                values[name] = evaluate(exprs[name], values)
            except ExprError as exc:
                raise ExprError(exc.kind, f"パラメータ {name}: {exc}") from exc
        state[name] = 2

    for name in exprs:
        visit(name, [])
    return values


# ---- 束縛の場所 ----


def _select(project: Mapping[str, Any], seq: list[Any], sel: Mapping[str, Any], where: str) -> int:
    if "id" in sel:
        for k, item in enumerate(seq):
            if isinstance(item, Mapping) and item.get("id") == sel["id"]:
                return k
        raise ValueError(f"{where}: id が {sel['id']!r} の要素がありません")
    if "edge" in sel:
        ids = list(((project.get("geometry") or {}).get("domain") or {}).get("edge_ids") or [])
        if sel["edge"] not in ids:
            raise ValueError(f"{where}: ドメインの辺 {sel['edge']!r} がありません")
        e = ids.index(sel["edge"])
        for k, item in enumerate(seq):
            if isinstance(item, Mapping) and e in (item.get("edges") or []):
                return k
        raise ValueError(f"{where}: 辺 {sel['edge']!r} の境界条件がありません")
    raise ValueError(f"{where}: 要素の選び方 {dict(sel)!r} を読めません")


def resolve(project: Mapping[str, Any], path: list[Any]) -> tuple[Any, Any]:
    """束縛の場所 → (入れ物, キーか番号)。無ければ ValueError。"""
    if not path:
        raise ValueError("束縛の場所が空です")
    where = "束縛 " + ".".join(str(p) for p in path)
    cur: Any = project
    for k, tok in enumerate(path):
        if isinstance(tok, Mapping):
            if not isinstance(cur, list):
                raise ValueError(f"{where}: 配列ではない所で要素を選んでいます")
            key: Any = _select(project, cur, tok, where)
        elif isinstance(tok, bool):
            raise ValueError(f"{where}: 読めない段 {tok!r}")
        elif isinstance(tok, int):
            if not isinstance(cur, list) or not (0 <= tok < len(cur)):
                raise ValueError(f"{where}: 番号 {tok} の要素がありません")
            key = tok
        elif isinstance(tok, str):
            if not isinstance(cur, Mapping) or tok not in cur:
                raise ValueError(f"{where}: {tok!r} がありません")
            key = tok
        else:
            raise ValueError(f"{where}: 読めない段 {tok!r}")
        if k == len(path) - 1:
            return cur, key
        cur = cur[key]
    raise AssertionError("unreachable")


def apply_params(project: dict[str, Any], overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """パラメータを計算して各パラメータの value と、束縛した欄の数値を書き直す (その場で)。値 (名前 → SI) を返す。

    overrides の名前は式の代わりにその値にする (スイープ。式もその数値に書き換える)。
    """
    params = project.get("params")
    if not isinstance(params, dict):
        if overrides:
            raise ExprError("unknown", "文書にパラメータがありません")
        return {}
    vars_ = list(params.get("vars") or [])
    values = evaluate_params(vars_, overrides)
    for v in vars_:
        name = v.get("name")
        if overrides and name in overrides:
            v["expr"] = repr(float(overrides[name]))
        v["value"] = values[name]
    for b in params.get("bindings") or []:
        path = list(b.get("path") or [])
        try:
            value = evaluate(str(b.get("expr", "")), values)
        except ExprError as exc:
            raise ExprError(exc.kind, "束縛 " + ".".join(str(p) for p in path) + f": {exc}") from exc
        container, key = resolve(project, path)
        old = container[key]
        if isinstance(old, bool) or not isinstance(old, (int, float)):
            raise ValueError("束縛 " + ".".join(str(p) for p in path) + f": 数値の欄ではありません ({old!r})")
        container[key] = value
    return values
