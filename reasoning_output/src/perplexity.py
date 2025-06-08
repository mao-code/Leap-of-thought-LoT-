"""Reasoningswise perplexity (RWP) utilities."""
from __future__ import annotations

import math
import re
from typing import List

__all__ = [
    "split_sentences",
    "sentence_pp",
    "rwp_from_pp",
    "sliding_rwp",
]

# ── Sentence segmentation (one sentence == one reasoning step) ───────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    """Very naïve rule-based segmentor.  Feel free to plug your own."""
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


# ── PP_k  &  RWP computations ───────────────────────────────────────────────
def sentence_pp(log_probs: List[float]) -> float:
    """PP_k  (equation 1) computed from token log-probabilities."""
    if not log_probs:
        return float("inf")
    avg_neg_logp = -sum(log_probs) / len(log_probs)
    return math.exp(avg_neg_logp)


def rwp_from_pp(pps: List[float]) -> float:
    """RWP  (equation 2) given the list of PP_k."""
    if not pps:
        return float("inf")
    product = math.prod(pps)
    return product ** (1 / len(pps))


def sliding_rwp(pps: List[float], window: int) -> List[float]:
    """Compute RWP^{(w)} for a sliding window of PP_k values (eq. 3)."""
    rwps = []
    for t in range(len(pps)):
        start = max(0, t - window + 1)
        sub = pps[start : t + 1]
        rwps.append(rwp_from_pp(sub))
    return rwps