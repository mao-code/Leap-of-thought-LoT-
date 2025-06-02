"""Light‑weight evaluator mirroring the original version but with leap logic."""
from __future__ import annotations

import json
from pathlib import Path
from tqdm import tqdm

from reasoning_output.generator import LeapGenerator, dump_record
import re

ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*(.+)", re.I)

def _extract_answer(text: str):
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None

def evaluate(dataset_path: str, *, fewshot: bool = False):
    gen = LeapGenerator()
    records = []
    with open(dataset_path) as f:
        data = [json.loads(l) for l in f]

    for sample in tqdm(data, desc="Evaluating-with-RWP"):
        rec = gen.generate(sample["question"], fewshot=fewshot)
        pred = _extract_answer(rec.get("leap_answer") or rec["first_answer"])
        correct = pred in sample["answers"]
        rec["pred"] = pred
        rec["gold"] = sample["answers"]
        rec["correct"] = correct
        records.append(rec)
        dump_record(rec)

    accuracy = sum(r["correct"] for r in records) / len(records)
    summary = {"accuracy": accuracy, "n": len(records)}
    Path("logs/reasoning_output_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, records