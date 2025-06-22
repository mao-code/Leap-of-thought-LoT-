from __future__ import annotations

import argparse
import json
import time
import os
from typing import Iterable

from openai import OpenAI
from tqdm import tqdm

from dataset import TrainingDataset
from config import settings

from dotenv import load_dotenv
load_dotenv()  

OPEANAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPEANAI_API_KEY:
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
5. Otherwise, skip LT.

### Format rules
1. Produce exactly **one** <think> ... </think> block.
2. Place every <leap> ... </leap> **inside** that single <think>.
3. Emit any number of <leap> blocks (including zero).
4. After the reasoning block, output on a new line:  
   **Answer:** {answer}
5. Any text after that line is forbidden.

### Examples
Question: 1 + 1 = ?
<think>
Simply add the numbers.
</think>
**Answer:** 2

Question: A farmer has pigs and chickens totalling 22 heads and 68 legs. How many pigs?
<think>
Let x = pigs, y = chickens.  
Heads: x + y = 22.  
Legs: 4x + 2y = 68.  
<leap>Notice each pig adds 2 extra legs over a chicken, so subtract heads equation from half the legs: (68 / 2) - 22 = 12 pigs.</leap>
</think>
**Answer:** 12
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


def generate_reasoning(client: OpenAI, question: str, answer: str, model_name: str, temperature: float) -> str:
    messages = build_messages(question, answer)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def distill_dataset(dataset: Iterable[dict], model_name: str, temperature: float, output_path: str, delay: float = 0.0) -> None:
    client = OpenAI(api_key=OPEANAI_API_KEY)

    with open(output_path, "w", encoding="utf-8") as fout:
        for record in tqdm(dataset, desc="Distilling"):
            question = record["question"]
            answer = record["answers"][0]
            reasoning = generate_reasoning(client, question, answer, model_name, temperature)
            out_rec = {
                "question": question,
                "solution": reasoning,
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
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use for reasoning")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for OpenAI API")
    parser.add_argument("--output", type=str, default="alot_dataset.jsonl", help="Where to write the distilled dataset")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between API calls")
    args = parser.parse_args()

    
    dataset = TrainingDataset(total_samples=args.samples)
    distill_dataset(dataset, args.model_name, args.temperature, args.output, delay=args.delay)


if __name__ == "__main__":
    main()
