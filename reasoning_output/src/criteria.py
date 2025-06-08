"""Criterion deciding whether to inject <leap>…</leap>."""
from __future__ import annotations

from config import settings
from reasoning_output.perplexity import split_sentences, reasoning_wise_perplexity
from model_provider import BaseModel


def should_trigger_leap(rationale: str, model: BaseModel) -> bool:
    """Return **True** if RWP exceeds the configured threshold."""
    sentences = split_sentences(rationale)
    rwp = reasoning_wise_perplexity(sentences, model)
    return rwp >= settings.rwp_threshold