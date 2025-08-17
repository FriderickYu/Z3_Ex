# -*- coding: utf-8 -*-
"""
tautology_check.py
兼容版“恒真式（tautology）”检测流水线
"""

from __future__ import annotations
from typing import Dict, Tuple, Union, Iterable, Optional
import re

from z3 import (
    BoolRef, BoolVal, And, Or, Not, Implies, Solver, simplify, is_true, is_false,
    is_and, is_or, is_quantifier, unsat, sat, Goal, Tactic, Then,
    parse_smt2_string, Bool, Z3Exception
)

TExpr = Union[BoolRef, str]
CheckResult = Tuple[bool, str]   # (is_tautology, detail)

__all__ = [
    "is_tautology_pipeline",
    "is_implication_tautology_pipeline",
    "parse_formula",
    "quick_implication_win",
    "quick_obvious_redundancy",
]

# ===================== 解析 =====================

def parse_formula(
    formula: TExpr,
    symbols: Optional[Dict[str, BoolRef]] = None,
    prefer: str = "auto"
) -> BoolRef:
    """
    将多形态输入解析为 z3 BoolRef。
    prefer: "auto" | "smt2" | "z3str"（大小写不敏感）。
    对于未知取值，会走安全回退：先试 smt2，失败再用 z3 风格解析。
    """
    # 已是 z3 AST
    if isinstance(formula, BoolRef):
        return formula

    s = (formula or "").strip()
    if not s:
        raise ValueError("Empty formula string.")

    # 归一化 prefer，预先定义 prefer_mode 防止作用域告警
    prefer_norm = (prefer or "auto").strip().lower()
    prefer_mode = None  # type: Optional[str]

    if prefer_norm == "auto":
        smt2_keys = ("(assert", "=>", "(and", "(or", "(not", "(exists", "(forall)")
        if s.startswith("(") and any(k in s for k in smt2_keys):
            prefer_mode = "smt2"
        else:
            prefer_mode = "z3str"
    elif prefer_norm == "smt2":
        prefer_mode = "smt2"
    elif prefer_norm in ("z3str", "z3", "z3py"):
        prefer_mode = "z3str"
    else:
        # 未知取值：安全回退（优先尝试 SMT-LIB2）
        try:
            return _parse_smt2(s, symbols)
        except Exception:
            return _parse_z3str_like(s, symbols)

    # 根据模式调用具体解析器
    if prefer_mode == "smt2":
        return _parse_smt2(s, symbols)
    # 默认走 z3 风格
    return _parse_z3str_like(s, symbols)


def _parse_smt2(smt2_text: str, symbols: Optional[Dict[str, BoolRef]]) -> BoolRef:
    text = smt2_text.strip()
    if not text.startswith("(assert"):
        text = f"(assert {text})"
    decls = dict(symbols or {})
    if not decls:
        KW = {"assert", "and", "or", "not", "=>", "true", "false", "exists", "forall", "ite", "xor"}
        for name in set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)):
            if name in KW:
                continue
            decls[name] = Bool(name)
    try:
        fvec = parse_smt2_string(text, decls=decls)
        if len(fvec) == 0:
            raise ValueError("SMT2 parsed no assertions.")
        if len(fvec) == 1:
            return fvec[0]
        return And(list(fvec))
    except Z3Exception as e:
        raise ValueError(f"SMT2 parse error: {e}")

_TOK_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|[,()])")

def _tokenize_z3str(s: str):
    i = 0
    while i < len(s):
        m = _TOK_RE.match(s, i)
        if not m:
            if s[i].isspace():
                i += 1
                continue
            raise ValueError(f"Unexpected char at {i}: {s[i]!r}")
        tok = m.group(1)
        yield tok
        i = m.end()

def _parse_z3str_like(s: str, symbols: Optional[Dict[str, BoolRef]]) -> BoolRef:
    tokens = list(_tokenize_z3str(s))
    pos = 0
    symtab: Dict[str, BoolRef] = dict(symbols or {})

    def get(name: str) -> BoolRef:
        if name.lower() == "true":
            return BoolVal(True)
        if name.lower() == "false":
            return BoolVal(False)
        if name not in symtab:
            symtab[name] = Bool(name)
        return symtab[name]

    def expect(t: str):
        nonlocal pos
        if pos >= len(tokens) or tokens[pos] != t:
            got = tokens[pos] if pos < len(tokens) else "<eof>"
            raise ValueError(f"Expect {t}, got {got}")
        pos += 1

    def parse_expr() -> BoolRef:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected <eof>")
        tok = tokens[pos]
        if re.match(r"[A-Za-z_]", tok):
            pos += 1
            if pos < len(tokens) and tokens[pos] == "(":
                func = tok
                pos += 1
                args = []
                if pos < len(tokens) and tokens[pos] != ")":
                    args.append(parse_expr())
                    while pos < len(tokens) and tokens[pos] == ",":
                        pos += 1
                        args.append(parse_expr())
                expect(")")
                return _build(func, args)
            else:
                return get(tok)
        elif tok == "(":
            pos += 1
            e = parse_expr()
            expect(")")
            return e
        else:
            raise ValueError(f"Unexpected token: {tok}")

    def _build(fn: str, args):
        fn_low = fn.lower()
        if fn_low == "and":
            if len(args) < 2:
                raise ValueError("And needs >=2 args")
            return And(*args)
        if fn_low == "or":
            if len(args) < 2:
                raise ValueError("Or needs >=2 args")
            return Or(*args)
        if fn_low == "not":
            if len(args) != 1:
                raise ValueError("Not needs 1 arg")
            return Not(args[0])
        if fn_low == "implies":
            if len(args) != 2:
                raise ValueError("Implies needs 2 args")
            return Implies(args[0], args[1])
        return get(fn)

    expr = parse_expr()
    if pos != len(tokens):
        raise ValueError(f"Trailing tokens: {' '.join(tokens[pos:])}")
    return expr

# ===================== 快速启发式 =====================

def _eq(a: BoolRef, b: BoolRef) -> bool:
    return is_true(simplify(a == b))

def _flatten_and(expr: BoolRef):
    out, stack = [], [expr]
    while stack:
        e = stack.pop()
        if is_and(e):
            stack.extend(e.children())
        else:
            out.append(e)
    return tuple(sorted(out, key=str))

def _flatten_or(expr: BoolRef):
    out, stack = [], [expr]
    while stack:
        e = stack.pop()
        if is_or(e):
            stack.extend(e.children())
        else:
            out.append(e)
    return tuple(sorted(out, key=str))

def quick_implication_win(prem: BoolRef, concl: BoolRef) -> bool:
    prem_s, concl_s = simplify(prem), simplify(concl)
    if _eq(prem_s, concl_s):
        return True
    if is_false(prem_s) or is_true(concl_s):
        return True
    if is_and(prem_s) and any(_eq(c, concl_s) for c in prem_s.children()):
        return True
    if is_or(concl_s):
        pc = set(_flatten_and(prem_s)) if is_and(prem_s) else {prem_s}
        for disj in _flatten_or(concl_s):
            if disj in pc:
                return True
    return False

def quick_obvious_redundancy(prem: BoolRef, concl: BoolRef) -> bool:
    """
    极简冗余：只判断最显然的情况，避免误杀正常推理。
    - p == q
    - False -> α
    - α -> True
    """
    prem_s, concl_s = simplify(prem), simplify(concl)
    if is_true(concl_s) or is_false(prem_s):
        return True
    return _eq(prem_s, concl_s)

# ===================== 策略/求解 =====================

def simplify_pipeline(b: BoolRef) -> BoolRef:
    b1 = simplify(b, elim_and=True, pull_cheap_ite=True, som=True, ctx_simplify=True)
    if is_true(b1) or is_false(b1):
        return b1
    g = Goal()
    g.add(b1)
    t = Then(
        Tactic('simplify'),
        Tactic('propagate-values'),
        Tactic('solve-eqs'),
        Tactic('ctx-simplify'),
    )
    r = t(g)
    sub = []
    for subg in r:
        sub.extend(list(subg))
    return And(sub) if sub else b1

def try_qe(b: BoolRef) -> BoolRef:
    try:
        g = Goal(); g.add(b)
        r = Tactic('qe')(g)
        conj = []
        for subg in r:
            conj.extend(list(subg))
        return And(conj) if conj else b
    except Exception:
        return b

def _solver_check(exprs: Iterable[BoolRef], timeout_ms: int) -> str:
    s = Solver()
    if timeout_ms is not None:
        s.set(timeout=timeout_ms)
    for e in exprs:
        s.add(e)
    r = s.check()
    if r == unsat: return "unsat"
    if r == sat:   return "sat"
    return "unknown"

def check_tautology(formula: BoolRef, timeout_ms: int = 2000) -> CheckResult:
    r = _solver_check([Not(formula)], timeout_ms)
    if r == "unsat": return True, "proved_unsat_of_negation"
    if r == "sat":   return False, "counterexample_exists"
    return False, "unknown_or_timeout"

def check_implication_tautology(prem: BoolRef, concl: BoolRef, timeout_ms: int = 2000) -> CheckResult:
    r = _solver_check([And(prem, Not(concl))], timeout_ms)
    if r == "unsat": return True, "proved_unsat_of_prem_and_not_concl"
    if r == "sat":   return False, "counterexample_exists"
    return False, "unknown_or_timeout"

# ===================== 管线接口 =====================

def is_tautology_pipeline(
    formula: TExpr,
    symbols: Optional[Dict[str, BoolRef]] = None,
    prefer: str = "auto",
    timeout_ms: int = 2000,
    use_qe: bool = True
) -> CheckResult:
    f = parse_formula(formula, symbols=symbols, prefer=prefer)
    f1 = simplify_pipeline(f)
    if is_true(f1):  return True, "simplify_to_true"
    if is_false(f1): return False, "simplify_to_false"
    f2 = try_qe(f1) if (use_qe and is_quantifier(f1)) else f1
    f2 = simplify_pipeline(f2)
    return check_tautology(f2, timeout_ms=timeout_ms)

def is_implication_tautology_pipeline(
    premise: TExpr,
    conclusion: TExpr,
    symbols: Optional[Dict[str, BoolRef]] = None,
    prefer: str = "auto",
    timeout_ms: int = 2000,
    use_qe: bool = True
) -> CheckResult:
    P = parse_formula(premise, symbols=symbols, prefer=prefer)
    Q = parse_formula(conclusion, symbols=symbols, prefer=prefer)
    if quick_implication_win(P, Q):
        return True, "quick_implication_win"
    P1, Q1 = simplify_pipeline(P), simplify_pipeline(Q)
    if quick_implication_win(P1, Q1):
        return True, "quick_implication_win_after_simplify"
    if use_qe and (is_quantifier(P1) or is_quantifier(Q1)):
        P1, Q1 = try_qe(P1), try_qe(Q1)
        P1, Q1 = simplify_pipeline(P1), simplify_pipeline(Q1)
        if quick_implication_win(P1, Q1):
            return True, "quick_implication_win_after_qe"
    return check_implication_tautology(P1, Q1, timeout_ms=timeout_ms)
