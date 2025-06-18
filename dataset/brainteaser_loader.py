from __future__ import annotations

import itertools
from typing import Iterator, List, Optional

import numpy as np


class BrainTeaserLoader:
    """Loader for the BrainTeaser dataset stored in ``dataset/data/brain_teaser``."""

    _FILE_MAP = {
        ("sentence", "train"): "dataset/data/brain_teaser/SP_train.npy",
        ("sentence", "test"): "dataset/data/brain_teaser/SP_test.npy",
        # ("word", "train"): "dataset/data/brain_teaser/WP_train.npy",
        # ("word", "test"): "dataset/data/brain_teaser/WP_test.npy",
    }

    def __init__(self, puzzle_type: str = "sentence", split: str = "train", num_samples: Optional[int] = None):
        key = (puzzle_type.lower(), split.lower())
        if key not in self._FILE_MAP:
            raise ValueError(f"Unknown puzzle_type/split combo: {puzzle_type} {split}")
        path = self._FILE_MAP[key]
        self._records = list(np.load(path, allow_pickle=True))
        self._iter: Iterator = iter(self._records)
        self._num_samples = num_samples
        self._yielded = 0

    def _extract(self, rec) -> tuple[str, str]:
        if isinstance(rec, dict):
            q = str(rec.get("question", "")).strip()
            a = str(rec.get("answer", "")).strip()
        elif isinstance(rec, (list, tuple)) and len(rec) >= 2:
            q, a = str(rec[0]).strip(), str(rec[1]).strip()
        else:
            q = str(rec).strip()
            a = ""
        return q, a

    def __iter__(self):
        for rec in self._iter:
            if self._num_samples is not None and self._yielded >= self._num_samples:
                break
            q, a = self._extract(rec)
            yield {"question": q, "answers": [a] if a else []}
            self._yielded += 1

    def get_examples(self, n: int) -> List[dict]:
        if self._num_samples is not None:
            n = min(n, max(0, self._num_samples - self._yielded))
        batch = []
        for rec in itertools.islice(self._iter, n):
            q, a = self._extract(rec)
            batch.append({"question": q, "answers": [a] if a else []})
            self._yielded += 1
        return batch
