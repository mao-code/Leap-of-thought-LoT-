"""CLI entry‑point.

Examples
========
# Local model (make sure you have VRAM!)
$ LOT_PROVIDER=local python -m reasoning_output.main --fewshot

# Remote Groq endpoint
$ LOT_PROVIDER=api GROQ_API_KEY=sk‑... python -m reasoning_output.main --fewshot
"""
import argparse
import pprint

from reasoning_output.evaluator import evaluate


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="input_prompt/data/gsm8k_demo.jsonl")
    ap.add_argument("--fewshot", action="store_true")
    args = ap.parse_args()

    summary, _ = evaluate(args.data, fewshot=args.fewshot)
    pprint.pp(summary)


if __name__ == "__main__":
    cli()