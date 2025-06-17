from __future__ import annotations

import itertools
from typing import Iterator, List, Optional

from datasets import load_dataset


class LOGIC701Loader:
    """Stream records from the LOGIC-701 benchmark."""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None):
        self._iter: Iterator[dict] = iter(
            load_dataset("hivaze/LOGIC-701", split=split, streaming=True)
        )
        self._num_samples = num_samples
        self._yielded = 0

    def _extract(self, rec) -> tuple[str, str]:
        q = rec.get("question", rec.get("input", "")).strip()
        a = rec.get("answer", rec.get("output", "")).strip()
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
