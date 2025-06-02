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

        # 1) non-leap reasoning
        non_leap_messages = build_prompt(sample["question"], fewshot=fewshot, leap=False)
        non_leap_output, non_leap_usage = chat_completion(non_leap_messages)

        non_leap_pred = _extract_answer(non_leap_output)
        non_leap_correct = non_leap_pred in sample["answers"]
        non_leap_reasoning_tokens = non_leap_usage.get("completion_tokens", 0)

        # 2) leap reasoning
        leap_messages = build_prompt(sample["question"], fewshot=fewshot, leap=True)
        leap_output, leap_usage = chat_completion(leap_messages)

        leap_pred = _extract_answer(leap_output)
        leap_correct = leap_pred in sample["answers"]
        leap_reasoning_tokens = leap_usage.get("completion_tokens", 0)

        records.append(
            {
                "gold": sample["answers"],

                "non_leap_messages": non_leap_messages,
                "non_leap_pred": non_leap_pred,
                "non_leap_correct": non_leap_correct,
                "non_leap_reasoning_tokens": non_leap_reasoning_tokens,
                "non_leap_usage": non_leap_usage,
                
                "leap_messages": leap_messages,
                "leap_pred": leap_pred,
                "leap_correct": leap_correct,
                "leap_reasoning_tokens": leap_reasoning_tokens,
                "leap_usage": leap_usage,
            }
        )

        # print(records[-1])  # Print the last record for debugging

    # Aggregate
    non_leap_accuracy = sum(r["non_leap_correct"] for r in records) / len(records)
    non_leap_avg_tokens = sum(r["non_leap_reasoning_tokens"] for r in records) / len(records)

    leap_accuracy = sum(r["leap_correct"] for r in records) / len(records)
    leap_avg_tokens = sum(r["leap_reasoning_tokens"] for r in records) / len(records)

    summary = {
        "non_leap_accuracy": non_leap_accuracy, 
        "non_leap_avg_reasoning_tokens": non_leap_avg_tokens,

        "leap_accuracy": leap_accuracy,
        "leap_avg_reasoning_tokens": leap_avg_tokens
    }
    Path("logs/input_prompts_summary.json").write_text(json.dumps(summary, indent=2))
    Path("logs/input_prompts_records.json").write_text(json.dumps(records, indent=2))
    return summary, records
