import itertools
from typing import Optional, List, Iterator

from datasets import load_dataset  # Hugging Face datasets library

class GSM8KLoader:
    """
    Lazily streams GSM8K examples and transforms each record into:
      {"question": <str>, "answers": [<str>]}.

    Args:
        split (str): Which split to stream ("train" or "test"). 
        num_samples (Optional[int]): If set, limits the total number of examples; otherwise streams entire split.
    """
    def __init__(self, split: str = "train", num_samples: Optional[int] = None):
        # 1. Load the split in streaming mode (IterableDataset) :contentReference[oaicite:19]{index=19}
        self._dataset_iterable = load_dataset(
            "DaertML/gsm8k-jsonl",
            split=split,
            streaming=True
        )

        # 2. Wrap into an iterator so we can call next() repeatedly
        self._buffered_iterator: Iterator[dict] = iter(self._dataset_iterable)

        # 3. Store how many samples max to yield; None = no limit (stop at dataset end)
        self._num_samples = num_samples
        self._samples_yielded = 0

    def _extract_final_answer(self, raw_ans: str) -> str:
        """
        Given a raw answer string (which may contain multiple lines and
        "####" markers), return just the text after the last "####", stripped.
        If "####" is not present, return raw_ans.strip().
        """
        if "#### " in raw_ans:
            # Split on "####" and take the last segment
            parts = raw_ans.split("#### ")
            answer_part = parts[-1].strip()
        else:
            answer_part = raw_ans.strip()
        return answer_part


    def __iter__(self):
        """
        Yields one transformed example at a time:
        {"question": <str>, "answers": [<str>]}.
        Honors num_samples limit if provided.
        """
        for raw in self._buffered_iterator:
            if self._num_samples is not None and self._samples_yielded >= self._num_samples:
                break

            q_text = raw.get("question", "").strip()  # Get question
            raw_answer_str = raw.get("answer", "").strip()  # Get answer as string
            final_answer = self._extract_final_answer(raw_answer_str)

            # 3) Wrap into a single-element list (or empty list if nothing)
            if final_answer != "":
                answers_list: List[str] = [final_answer]
            else:
                answers_list = []

            transformed = {
                "question": q_text,
                "answers": answers_list
            }

            yield transformed
            self._samples_yielded += 1

    def get_examples(self, n: int) -> List[dict]:
        """
        Returns the next n transformed examples as a list of dicts.
        Uses itertools.islice on the internal iterator, so you only
        pull exactly n records from the stream. :contentReference[oaicite:25]{index=25}
        """
        if self._num_samples is not None:
            # Ensure we don't exceed total limit
            remaining = max(0, self._num_samples - self._samples_yielded)
            n = min(n, remaining)

        # Slice exactly n items from the iterator
        slice_iter = itertools.islice(self._buffered_iterator, n)
        examples = []
        for raw in slice_iter:
            q_text = raw.get("question", "").strip()
            raw_ans = raw.get("answer", "").strip()
            final_ans = self._extract_final_answer(raw_ans)
            answers_list = [final_ans] if final_ans != "" else []

            examples.append({
                "question": q_text,
                "answers": answers_list
            })
            self._samples_yielded += 1

        return examples
