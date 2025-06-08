"""Light-weight evaluator for the LoT-2 pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from tqdm import tqdm

from reasoning_output.src.generator import LeapGenerator, dump_record
from reasoning_output.src.utils import extract_answer, normalize_answer


def evaluate(dataset_path: str, *, fewshot: bool = False):
    gen = LeapGenerator()
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]

    for sample in tqdm(data, desc="Evaluating-with-RWP"):
        rec = gen.generate(sample["question"], fewshot=fewshot)
        pred_raw = extract_answer(rec.get("leap_answer") or rec["first_answer"])
        pred = normalize_answer(pred_raw)
        gold_norm = [normalize_answer(a) for a in sample["answers"]]
        correct = any(pred.lower() == g.lower() for g in gold_norm)
        rec.update({"pred": pred_raw, "gold": sample["answers"], "correct": correct})
        records.append(rec)
        dump_record(rec)

    accuracy = sum(r["correct"] for r in records) / len(records)
    summary = {"accuracy": accuracy, "n": len(records)}
    Path("logs/reasoning_output_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, records
