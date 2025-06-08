"""Reasoningswise perplexity (RWP) utilities."""
from __future__ import annotations

import math
import re
from typing import List

__all__ = [
    "split_sentences",
    "sentence_perplexity",
    "reasoning_wise_perplexity",
    "sliding_rwp",
]

# ── Sentence segmentation (one sentence == one reasoning step) ───────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    """Very naïve rule-based segmentor.  Feel free to plug your own."""
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


# ── PP_k  &  RWP computations ───────────────────────────────────────────────
def sentence_perplexity(sentence: str, context: str, model: BaseModel) -> float:
    """PP_k  (equation 1).  Uses *log* e and def‑n provided by user."""
    log_probs = model.sentence_logprobs(sentence, context=context)
    if not log_probs:
        return float("inf")
    avg_neg_logp = -sum(log_probs) / len(log_probs)
    return math.exp(avg_neg_logp)


def reasoning_wise_perplexity(sentences: List[str], model: BaseModel) -> float:
    """RWP  (equation 2)."""
    pps = []
    ctx = ""
    for sent in sentences:
        pp = sentence_perplexity(sent, ctx, model)
        pps.append(pp)
        ctx += sent + " "  # accumulate context for causal models
    # Geometric mean
    product = math.prod(pps)
    return product ** (1 / len(pps)) if pps else float("inf")


def sliding_rwp(sentences: List[str], window: int, model: BaseModel) -> List[float]:
    """Compute RWP^{(w)} for each window ending at t (eq. 3)."""
    rwps = []
    for t in range(len(sentences)):
        start = max(0, t - window + 1)
        sub = sentences[start : t + 1]
        rwps.append(reasoning_wise_perplexity(sub, model))
    return rwps