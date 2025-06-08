"""Criterion deciding whether to inject <leap>…</leap>."""
from __future__ import annotations

from config import settings
from reasoning_output.src.perplexity import rwp_from_pp, sliding_rwp

def should_trigger_leap(pps: list[float], use_window: bool=False) -> bool:
    """Return **True** if the sliding-window RWP exceeds the threshold."""
    if use_window:
        window = settings.window
        rwp = sliding_rwp(pps, window)
    else:
        rwp = rwp_from_pp(pps)
        
    return (rwp >= settings.rwp_threshold, rwp)