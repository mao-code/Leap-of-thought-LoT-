# answer_normaliser.py
# --------------------
"""
Utility for canonicalising short maths answers coming from either:
  • the MATH-500 gold set (usually TeX) or
  • an LLM (often plain text or lightly-formatted TeX).

Example
-------
>>> from answer_normaliser import normalise
>>> normalise(r"\left( 3, \frac{\pi}{2} \right)")
'(3, pi/2)'
>>> normalise("( 3 , π/2 )")
'(3, pi/2)'
"""

from __future__ import annotations
import re
from typing import Iterable

from sympy.parsing.latex import parse_latex          # needs SymPy ≥1.12
from sympy import srepr, sympify, Tuple

# --------------------------------------------------------------------------- #
#  1.  Cheap text-level cleanup
# --------------------------------------------------------------------------- #
_DOLLAR_RE = re.compile(r"^\$([^$]+)\$$")
_LEFT_RIGHT_RE = re.compile(r"\\left\s*|\s*\\right")
_BOX_RE = re.compile(r"\\boxed\s*{([^}]*)}")
_INLINE_WS_RE = re.compile(r"\s+")

def _preclean(txt: str) -> str:
    """Strip $…$, \left · \right, \boxed{…} and collapse whitespace."""
    txt = txt.strip()

    # $…$
    m = _DOLLAR_RE.match(txt)
    if m:
        txt = m.group(1).strip()

    # \boxed{…}
    m = _BOX_RE.fullmatch(txt)
    if m:
        txt = m.group(1).strip()

    # \left … \right
    txt = _LEFT_RIGHT_RE.sub("", txt)

    # compress internal whitespace
    txt = _INLINE_WS_RE.sub(" ", txt)

    return txt


# --------------------------------------------------------------------------- #
#  2.  SymPy canonicalisation helpers
# --------------------------------------------------------------------------- #
def _expr_to_canonical(expr) -> str:
    """
    Turn a SymPy Expr (or tuple / set thereof) into a deterministic string.

    We use `srepr()` because it is both canonical and hashable.
    """
    if isinstance(expr, Iterable) and not hasattr(expr, "free_symbols"):
        # make sure tuples, lists, sets are immutable & ordered for comparison
        expr = Tuple(*expr)
    return srepr(expr.simplify())


def _try_sympy_parse(txt: str) -> str | None:
    """Return canonical s-repr string or None if parsing failed."""
    try:
        # Primary: TeX → SymPy
        expr = parse_latex(txt)
        return _expr_to_canonical(expr)
    except Exception:  # noqa: BLE001
        pass

    try:
        # Fallback: plain-text maths → SymPy
        expr = sympify(txt, convert_xor=True, rational=True)
        return _expr_to_canonical(expr)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
#  3.  Public API
# --------------------------------------------------------------------------- #
def normalise(raw: str) -> str:
    """
    Canonicalise *raw* answer string.

    Returns
    -------
    str
        A string that is identical **iff** two answers are mathematically
        equivalent (best-effort).  Guaranteed non-empty.
    """
    txt = _preclean(raw)

    sym = _try_sympy_parse(txt)
    if sym is not None:
        return sym                    # high-confidence canonical form

    # final fallback – never compare whitespace / case
    return re.sub(r"\s+", "", txt).lower()
