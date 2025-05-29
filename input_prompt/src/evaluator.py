import json, re
from pathlib import Path
from tqdm import tqdm
from input_prompt.src.api_client import chat_completion
from input_prompt.src.prompt_engine import build_prompt

ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*(.+)", re.I)

def _extract_answer(text: str):
    """Pull the answer token after '**Answer:**' (DeepSeek standard)."""
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else None

def evaluate(dataset_path: str, fewshot: bool = False):
    records = []
    with open(dataset_path) as f:
        data = [json.loads(l) for l in f]

    for sample in tqdm(data, desc="Evaluating"):
        messages = build_prompt(sample["question"], fewshot=fewshot)
        output, usage = chat_completion(messages)

        pred = _extract_answer(output)
        correct = pred in sample["answers"]
        reasoning_tokens = usage.get("completion_tokens", 0)

        records.append(
            {
                "question": sample["question"],
                "gold": sample["answers"],
                "pred": pred,
                "correct": correct,
                "reasoning_tokens": reasoning_tokens,
                "usage": usage,
            }
        )

    # Aggregate
    accuracy = sum(r["correct"] for r in records) / len(records)
    avg_tokens = sum(r["reasoning_tokens"] for r in records) / len(records)

    summary = {"accuracy": accuracy, "avg_reasoning_tokens": avg_tokens}
    Path("logs/summary.json").write_text(json.dumps(summary, indent=2))
    return summary, records
