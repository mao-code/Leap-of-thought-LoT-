from __future__ import annotations

import random
from typing import Iterable, List, Tuple

from .lateral_thinking_loader import LateralThinkingLoader
from .brainteaser_loader import BrainTeaserLoader
from .logic701_loader import LOGIC701Loader
from .gsm8k_loader import GSM8KLoader
from .math500_loader import MATH500Loader


class TrainingDataset:
    """Mix multiple datasets according to fixed proportions."""

    def __init__(self, total_samples: int = 1500):
        if not (1000 <= total_samples <= 2000):
            raise ValueError("total_samples must be between 1000 and 2000")

        # -- category level counts -------------------------------------------------
        n_riddle = round(total_samples * 0.3)
        n_logic = round(total_samples * 0.4)
        n_math = total_samples - n_riddle - n_logic

        # -- dataset level counts (even split inside category) --------------------
        n_lt = n_riddle // 2
        n_bt = n_riddle - n_lt

        n_gsm8k = n_math // 2
        n_math500 = n_math - n_gsm8k

        n_logic701 = n_logic

        # instantiate loaders with sample limits and metadata
        self._loaders: List[Tuple[Iterable[dict], str, str]] = [
            (LateralThinkingLoader(num_samples=n_lt), "riddle", "lateral_thinking"),
            (BrainTeaserLoader(num_samples=n_bt), "riddle", "brainteaser"),
            (LOGIC701Loader(split="train", num_samples=n_logic701), "logic", "logic701"),
            (GSM8KLoader(split="train", num_samples=n_gsm8k), "math", "gsm8k"),
            (MATH500Loader(num_samples=n_math500), "math", "math500"),
        ]

        self._data: List[dict] | None = None

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        if self._data is not None:
            return
        data: List[dict] = []

        def _clean_and_tag(rec: dict, ds_type: str, ds_name: str) -> dict | None:
            q = str(rec.get("question", "")).strip()
            answers = [str(a).strip() for a in rec.get("answers", []) if a is not None]

            # filter N/A or empty
            if not q or q.lower() in {"n/a", "null", "none"}:
                return None
            answers = [a for a in answers if a and a.lower() not in {"n/a", "null", "none"}]
            if not answers:
                return None

            return {
                "question": q,
                "answers": answers,
                "type": ds_type,
                "dataset": ds_name,
            }

        for loader, ds_type, ds_name in self._loaders:
            for rec in loader:
                tagged = _clean_and_tag(rec, ds_type, ds_name)
                if tagged is not None:
                    data.append(tagged)

        random.shuffle(data)
        self._data = data

    # ------------------------------------------------------------------
    def __iter__(self):
        self._load_all()
        return iter(self._data)

    def __len__(self) -> int:  # pragma: no cover - simple container
        self._load_all()
        return len(self._data)

    def get_examples(self, n: int) -> List[dict]:
        """Return first *n* examples after mixing."""
        self._load_all()
        return self._data[:n]
