from __future__ import annotations

import itertools
from typing import Iterator, List, Optional, Dict

from datasets import load_dataset


class LOGIC701Loader:
    """
    Stream records from the LOGIC-701 benchmark (English split) and convert each
    row into a dictionary of the form

        {
            "question": <problem_statement>,
            "choices":  [<opt1>, <opt2>, <opt3>, <opt4>, <opt5>],
            "answers":  [<correct_answer_text>],
            "correct_option_number": <int>,          # 1-indexed
            "topic":    <topic>,                     # optional extra
            "solution": <solution_text>              # optional extra
        }

    Blank or malformed rows are skipped.
    """

    def __init__(
        self,
        split: str = "train",
        lang: str = "en",
        num_samples: Optional[int] = None,
    ):
        self._iter: Iterator[Dict] = iter(
            load_dataset(
                "hivaze/LOGIC-701",
                lang,
                split=split,
                streaming=True,   # keeps memory/disk use tiny
            )
        )
        self._num_samples = num_samples
        self._yielded = 0

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract(rec) -> Optional[Dict]:
        """Return a cleaned sample or None if mandatory parts are missing."""
        q = str(rec.get("problem_statement", "")).strip()
        # Gather the 5 candidate answers in order
        choices = [
            str(rec.get(f"answer_option_{i}", "")).strip() for i in range(1, 6)
        ]
        # Identify the correct answer text
        idx = rec.get("correct_option_number")
        if (
            not q
            or not isinstance(idx, int)
            or idx < 1
            or idx > len(choices)
            or not choices[idx - 1]
        ):
            return None  # skip bad row

        return {
            "question": q,
            "choices": choices,
            "answers": [choices[idx - 1]],
            "correct_option_number": idx,
            # Optional extras you might use for analysis later
            "topic": rec.get("topic", "").strip(),
            "solution": str(rec.get("solution", "")).strip(),
        }

    # ------------------------------------------------------------------ #
    #  Python iterator protocol
    # ------------------------------------------------------------------ #
    def __iter__(self):
        for rec in self._iter:
            if self._num_samples is not None and self._yielded >= self._num_samples:
                break

            sample = self._extract(rec)
            if sample is None or sample["answers"][0] == "Another answer":
                continue  # skip rows whose gold answer is opt5 / "Another answer"

            yield sample
            self._yielded += 1

    # ------------------------------------------------------------------ #
    #  Convenience helper — grab *n* ready-to-use examples
    # ------------------------------------------------------------------ #
    def get_examples(self, n: int) -> List[Dict]:
        """Return the next *n* cleaned examples (≤ remaining budget)."""
        if self._num_samples is not None:
            n = min(n, max(0, self._num_samples - self._yielded))

        batch = []
        for rec in itertools.islice(self._iter, n):
            sample = self._extract(rec)
            if sample is None or sample["answers"][0] == "Another answer":
                continue
            batch.append(sample)
            self._yielded += 1

        return batch
