"""
math500_loader.py
~~~~~~~~~~~~~~~~~
Streaming DataLoader for the HuggingFaceH4/MATH-500 benchmark.

Example
-------
from math500_loader import MATH500Loader

dl = MATH500Loader(num_samples=10)   # stream first 10 items
for ex in dl:
    print(ex["question"])
    print(ex["answers"])   # list[str] – usually length 1
"""

from __future__ import annotations
import itertools
import re
from typing import Iterator, List, Optional

from datasets import load_dataset


_BOX_RE = re.compile(r"\\boxed\s*{([^}]*)}")   # matches \boxed{...}

class MATH500Loader:
    """
    Lazily streams MATH-500 examples and converts each record to:
        { "question": <str>, "answers": [<str>] }

    Parameters
    ----------
    num_samples : Optional[int]
        Maximum number of samples to yield.  None = all 500 rows.
    """
    def __init__(self, num_samples: Optional[int] = None):
        # 1. MATH-500 only has the `test` split
        self._dataset_iter = iter(
            load_dataset("HuggingFaceH4/MATH-500",
                         split="test",
                         streaming=True)
        )

        self._num_samples = num_samples
        self._samples_yielded = 0

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _clean_answer(self, raw: str) -> str:
        """
        Strip whitespace, surrounding $...$, and \\boxed{…} wrappers.

        Returns
        -------
        str
            Normalised short answer string.
        """
        raw = raw.strip()

        # Remove leading/trailing inline-math dollars
        if raw.startswith("$") and raw.endswith("$"):
            raw = raw[1:-1].strip()

        # Unwrap \boxed{...} if present
        m = _BOX_RE.fullmatch(raw)
        if m:
            raw = m.group(1).strip()

        return raw

    # --------------------------------------------------------------------- #
    #  Python iterator protocol
    # --------------------------------------------------------------------- #
    def __iter__(self):
        for rec in self._dataset_iter:
            if (self._num_samples is not None
                    and self._samples_yielded >= self._num_samples):
                break

            question = rec.get("problem", "").strip()
            short_ans = self._clean_answer(rec.get("answer", ""))

            yield {
                "question": question,
                "answers": [short_ans] if short_ans else []
            }
            self._samples_yielded += 1

    # --------------------------------------------------------------------- #
    #  Convenience utility
    # --------------------------------------------------------------------- #
    def get_examples(self, n: int) -> List[dict]:
        """
        Return the next *n* streamed examples (≤ remaining budget).
        """
        if self._num_samples is not None:
            n = min(n, max(0, self._num_samples - self._samples_yielded))

        sliced = itertools.islice(self._dataset_iter, n)
        batch = []

        for rec in sliced:
            question = rec.get("problem", "").strip()
            short_ans = self._clean_answer(rec.get("answer", ""))

            batch.append({
                "question": question,
                "answers": [short_ans] if short_ans else []
            })
            self._samples_yielded += 1

        return batch
