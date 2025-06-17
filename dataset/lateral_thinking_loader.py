from __future__ import annotations

import itertools
from typing import Iterator, List, Optional

import pandas as pd


class LateralThinkingLoader:
    """Simple loader for the Lateral Thinking riddles spreadsheet."""

    def __init__(self, num_samples: Optional[int] = None):
        df = pd.read_excel("dataset/data/lateral_thinking.xlsx")
        self._records = df.to_dict(orient="records")
        self._iter: Iterator[dict] = iter(self._records)
        self._num_samples = num_samples
        self._yielded = 0

    def __iter__(self):
        for rec in self._iter:
            if self._num_samples is not None and self._yielded >= self._num_samples:
                break
            question = str(rec.get("story", "")).strip()
            answer = str(rec.get("answer", "")).strip()
            yield {
                "question": question,
                "answers": [answer] if answer else []
            }
            self._yielded += 1

    def get_examples(self, n: int) -> List[dict]:
        if self._num_samples is not None:
            n = min(n, max(0, self._num_samples - self._yielded))
        batch = []
        for rec in itertools.islice(self._iter, n):
            question = str(rec.get("story", "")).strip()
            answer = str(rec.get("answer", "")).strip()
            batch.append({
                "question": question,
                "answers": [answer] if answer else []
            })
            self._yielded += 1
        return batch
