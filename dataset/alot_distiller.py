from __future__ import annotations

import argparse
import json
import time
import os
from typing import Iterable

from openai import OpenAI
from tqdm import tqdm

from dataset import TrainingDataset

from dotenv import load_dotenv
load_dotenv()  

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

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
5. Use at least one LT leap if you can.

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
{"reasoning":"<think>Simply add the numbers.</think> **Answer:** 2","answer":"2"}

Question: A farmer has pigs and chickens totalling 22 heads and 68 legs. How many pigs?
{"reasoning":"<think>Let x=pigs, y=chickens. Heads: x+y=22. Legs: 4x+2y=68. <leap>Each pig adds 2 extra legs over a chicken; so (68/2)-22=12.</leap></think> **Answer:** 12","answer":"12"}
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


def generate_reasoning(client: OpenAI, question: str, answer: str, model_name: str) -> str:
    messages = build_messages(question, answer)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def distill_dataset(dataset: Iterable[dict], model_name: str, output_path: str, delay: float = 0.0) -> None:
    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(output_path, "w", encoding="utf-8") as fout:
        for record in tqdm(dataset, desc="Distilling"):
            question = record["question"]
            answer = record["answers"][0]
            output = generate_reasoning(client, question, answer, model_name)
            output = json.loads(output)  # Parse the JSON response
            out_rec = {
                "question": question,
                "solution": output["reasoning"],
                "uses_leap": "<leap>" in output["reasoning"],
                "answer": answer,
                "dataset": record.get("dataset"),
                "type": record.get("type"),
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            if delay > 0:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw data into ALoT training format using OpenAI API")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--model", type=str, default="o3", help="OpenAI model to use for reasoning")
    parser.add_argument("--output", type=str, default="alot_dataset.jsonl", help="Where to write the distilled dataset")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between API calls")
    args = parser.parse_args()

    
    dataset = TrainingDataset(total_samples=args.samples)
    distill_dataset(dataset, args.model, args.output, delay=args.delay)


if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m dataset.alot_distiller \
        --samples 5 \
        --model o3 \
        --output alot_dataset.jsonl \
        --delay 0.1
    """
