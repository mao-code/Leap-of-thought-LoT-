from __future__ import annotations

import random
from typing import Iterable, List

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

        # instantiate loaders with sample limits
        self._loaders: List[Iterable[dict]] = [
            LateralThinkingLoader(num_samples=n_lt),
            BrainTeaserLoader(num_samples=n_bt),
            LOGIC701Loader(split="train", num_samples=n_logic701),
            GSM8KLoader(split="train", num_samples=n_gsm8k),
            MATH500Loader(num_samples=n_math500),
        ]

        self._data: List[dict] | None = None

    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        if self._data is not None:
            return
        data = []
        for loader in self._loaders:
            data.extend(list(loader))
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
