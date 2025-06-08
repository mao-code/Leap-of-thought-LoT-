from __future__ import annotations

"""Demo script exercising the RWP-based leap insertion pipeline (LoT-2)."""

import argparse
from tqdm import tqdm

from dataset.gsm8k_loader import GSM8KLoader
from reasoning_output.src.generator import LeapGenerator
from reasoning_output.src.utils import extract_answer, normalize_answer
from utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a few examples with LoT-2")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="How many GSM8K samples to run")
    parser.add_argument("--fewshot", action="store_true",
                        help="Use few-shot prompting")
    args = parser.parse_args()

    set_seed()
    loader = GSM8KLoader(split="train", num_samples=args.num_samples)
    gen = LeapGenerator()

    correct_first = 0
    correct_final = 0

    for sample in tqdm(loader, total=args.num_samples, desc="Evaluating"):
        q = sample["question"]
        gold = sample["answers"]
        rec = gen.generate(q, fewshot=args.fewshot)

        ans_first = extract_answer(rec["first_answer"])
        ans_final = extract_answer(rec.get("leap_answer") or rec["first_answer"])

        norm_first = normalize_answer(ans_first)
        norm_final = normalize_answer(ans_final)
        norm_gold = [normalize_answer(a) for a in gold]

        if any(norm_first.lower() == g.lower() for g in norm_gold):
            correct_first += 1
        if any(norm_final.lower() == g.lower() for g in norm_gold):
            correct_final += 1

        print("\n" + "=" * 80)
        print("Question:", q)
        print("\nFirst pass:\n", rec["first_answer"])
        if rec["trigger_leap"]:
            print("\nLeap pass:\n", rec.get("leap_answer", ""))
        else:
            print("\nNo leap triggered.")
        print("Predicted answer:", ans_final)

    total = args.num_samples
    print("\nAccuracy without leap:", correct_first / total if total else 0)
    print("Accuracy with leap:", correct_final / total if total else 0)


if __name__ == "__main__":
    main()
