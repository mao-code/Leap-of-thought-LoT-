from __future__ import annotations

import argparse
import json
import time
import os
from typing import Iterable

from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError
from tqdm import tqdm
import requests

from dataset import TrainingDataset
import logging

from dotenv import load_dotenv
load_dotenv()  

MAX_RETRIES_PER_RECORD = 5         # hard stop to avoid infinite loops
BASE_BACKOFF_SECONDS   = 1.5       # 1.5, 3, 6, 12, … seconds

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

# Focus on waht is ALoT and when to use it
SYSTEM_PROMPT = """
You are an expert problem-solver.  Follow **all** tag rules exactly.

### Glossary
* Vertical Thinking (VT): orderly, rule-based logic steps.
* Chain-of-Thought (CoT): written VT; must be inside <think> ... </think>.
* Lateral Thinking (LT): A manner of solving problems using an indirect, creative or cross-domain approach via reasoning that is not immediately obvious. Synonymous to thinking outside the box, it involves ideas that may not be obtainable using only traditional step-by-step logic; each jump goes in <leap> ... </leap>.
* Adaptive Leap-of-Thought (ALoT): decide when VT alone suffices and when to insert LT jumps.

### When to open a <leap>
1. VT would take > 5 steps **and** an indirect, creative or cross-domain insight can shorten it.
2. You need clever external knowledge not found in the prompt.
3. The problem hints at a trick or hidden pattern.
4. You can solve the problem in short steps with a clever and creative insight.
5. Only use LT leap when it is needed.

### Resoning Format rules
1. Produce exactly **one** <think> ... </think> block.
2. Place every <leap> ... </leap> **inside** that single <think>.
3. Emit any number of <leap> blocks (including zero).
4. After the reasoning block, output on a new line:  **Answer:** {answer}
5. Any text after that line is forbidden.

### Output Format
Return a single JSON object with the following fields:
```json
{
  "reasoning": "<think> ... <leap> ... </leap> ... </think>",
  "answer": "the final answer as plain text"
}
```

### Examples
Question: 1 + 1 = ?
{"reasoning":"<think>Simply add the numbers.</think>\n**Answer:** 2","answer":"2"}

Question: A farmer has pigs and chickens totalling 22 heads and 68 legs. How many pigs?
{"reasoning":"<think>Let x=pigs, y=chickens. Heads: x+y=22. Legs: 4x+2y=68. <leap>Wait, I think it can be solved in a more efficient way! Each pig adds 2 extra legs over a chicken; so (68/2)-22=12.</leap></think>\n**Answer:** 12","answer":"12"}
"""


def build_messages(question: str, answer: str) -> list[dict]:
    prompt = (
        f"Question: {question}\n"
        f"You must arrive at the answer '{answer}'."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def generate_reasoning(client, question: str, answer: str, model_name: str) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=build_messages(question, answer),
        response_format={"type":"json_object"}
    )

    
    message = completion.choices[0].message
    response_text = message.content.strip()

    print("="*10, f"Message: {message}")  # Debugging line
    print("="*10, f"Response: {response_text}")  # Debugging line

    return response_text

def distill_dataset(
    dataset: Iterable[dict],
    model_name: str,
    output_path: str,
    delay: float = 0.0,
    target_samples: int | None = None,    # optional explicit quota
) -> None:
    # If TrainingDataset is finite we know its length; otherwise rely on target_samples
    total_needed = target_samples or (len(dataset) if hasattr(dataset, "__len__") else None)
    if total_needed is None:
        raise ValueError("Must supply target_samples when dataset is infinite / stream.")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    successes = 0
    it = iter(dataset)

    with open(output_path, "w", encoding="utf-8") as fout, \
         tqdm(total=total_needed, desc="Distilling", unit="sample") as pbar:

        while successes < total_needed:
            try:
                record = next(it)
            except StopIteration:
                # Ran out of records; start a fresh iterator to keep sampling
                it = iter(dataset)
                record = next(it)

            retries = 0
            while retries <= MAX_RETRIES_PER_RECORD:
                try:
                    output_raw = generate_reasoning(
                        client, record["question"], record["answers"][0], model_name
                    )

                    output = json.loads(output_raw)
                    reasoning = output.get("reasoning", "").strip()

                    out_rec = {
                        "question": record["question"],
                        "solution": reasoning,
                        "uses_leap": "<leap>" in reasoning,
                        "answer": record["answers"][0],
                        "dataset": record.get("dataset"),
                        "type":    record.get("type"),
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    fout.flush()

                    successes += 1
                    pbar.update(1)
                    if delay:
                        time.sleep(delay)
                    break

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logging.warning(f"Bad JSON for q='{record['question'][:50]}...': {e}")
                except (RateLimitError, APIConnectionError, OpenAIError) as e:
                    logging.warning(f"OpenRouter error: {e}")

                retries += 1
                if retries > MAX_RETRIES_PER_RECORD:
                    logging.error("Max retries hit - skipping this record.")
                    break
                backoff = BASE_BACKOFF_SECONDS * (2 ** (retries - 1))
                time.sleep(backoff)

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw data into ALoT training format using OpenRouter API")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-r1-distill-qwen-32b",
        help="OpenRouter model to use for reasoning",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/distilled_data/alot_dataset_r1distilled_qwen_32B.jsonl",
        help="Where to write the distilled dataset",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between API calls")
    args = parser.parse_args()

    
    dataset = TrainingDataset(total_samples=args.samples)
    distill_dataset(dataset, args.model, args.output, delay=args.delay)


if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m dataset.openrouter_alot_distiller \
        --samples 3 \
        --model deepseek/deepseek-r1-distill-qwen-32b \
        --output dataset/distilled_data/alot_dataset_r1distilled_qwen_32B.jsonl \
        --delay 0.1
    """
