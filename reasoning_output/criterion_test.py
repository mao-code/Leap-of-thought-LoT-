from __future__ import annotations

"""Demo script exercising the RWP-based leap insertion pipeline (LoT-2)."""

import argparse
from tqdm import tqdm

from dataset.gsm8k_loader import GSM8KLoader
from dataset.math500_loader import MATH500Loader
from reasoning_output.src.generator import LeapGenerator
from reasoning_output.src.utils import extract_answer     
from dataset.answer_normaliser import normalise as normalize_answer   
from utils import set_seed
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a few examples with LoT-2")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="How many GSM8K samples to run")
    parser.add_argument("--fewshot", action="store_true",
                        help="Use few-shot prompting")
    args = parser.parse_args()

    set_seed()

    if args.dataset == "gsm8k":
        loader = GSM8KLoader(split="train", num_samples=args.num_samples)
    elif args.dataset == "math500":
        loader = MATH500Loader(num_samples=args.num_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    gen = LeapGenerator(args.model)
    tok = gen.tok

    correct_normal = 0
    correct_leap = 0
    sum_tok_normal = 0
    sum_tok_leap = 0
    records = []

    for sample in tqdm(loader, total=args.num_samples, desc="Evaluating"):
        q = sample["question"]
        gold = sample["answers"]

        rec = gen.generate(q, fewshot=args.fewshot)

        ans_normal = extract_answer(rec["normal_reasoning_text"])
        ans_leap = extract_answer(rec.get("leap_reasoning_text") or rec["normal_reasoning_text"])

        norm_ans_normal = normalize_answer(ans_normal)
        norm_ans_leap = normalize_answer(ans_leap)
        norm_ans_gold = [normalize_answer(a) for a in gold]

        is_correct_normal = any(norm_ans_normal.lower() == g.lower() for g in norm_ans_gold)
        is_correct_leap = any(norm_ans_leap.lower() == g.lower() for g in norm_ans_gold)

        tok_normal = len(tok(rec["normal_reasoning_text"]).input_ids)
        tok_leap = len(tok(rec.get("leap_reasoning_text") or rec["normal_reasoning_text"]).input_ids)

        if is_correct_normal:
            correct_normal += 1
        if is_correct_leap:
            correct_leap += 1
        sum_tok_normal += tok_normal
        sum_tok_leap += tok_leap

        records.append({
            "question": q,
            "norm_ans_gold": norm_ans_gold,
            # "ans_normal": ans_normal,
            # "ans_leap": ans_leap,
            "norm_ans_normal": norm_ans_normal,
            "norm_ans_leap": norm_ans_leap,
            "is_correct_normal": is_correct_normal,
            "is_correct_leap": is_correct_leap,
            "tok_normal": tok_normal,
            "tok_leap": tok_leap,
            "normal_reasoning_text": rec["normal_reasoning_text"],
            "leap_reasoning_text": rec.get("leap_reasoning_text", ""),
            "pps_trajectory": rec.get("pps_trajectory", []),
            "rwp_trajectory": rec.get("rwp_trajectory", []),
        })

        print("\n" + "=" * 80)
        print("Question:", q)
        print("\nFirst pass:\n", rec["normal_reasoning_text"])
        if rec["trigger_leap"]:
            print("\nLeap pass:\n", rec.get("normal_reasoning_cut_text", ""))
        else:
            print("\nNo leap triggered.")
        print("Predicted answer:", norm_ans_leap)

        print("PPs trajectory:", rec.get("pps_trajectory", []))
        print("RWP trajectory:", rec.get("rwp_trajectory", []))

    n_examples = args.num_samples
    acc_first = correct_normal / n_examples if n_examples > 0 else 0.0
    acc_final = correct_leap / n_examples if n_examples > 0 else 0.0
    avg_tok_first = sum_tok_normal / n_examples if n_examples > 0 else 0.0
    avg_tok_final = sum_tok_leap / n_examples if n_examples > 0 else 0.0

    print("\n──────────────────────────────────────────────────────")
    print(f"Total examples       : {n_examples}")
    print(f"First pass reasoning : Accuracy = {acc_first:.2%},  Avg Tokens = {avg_tok_first:.1f}")
    print(f"Final reasoning      : Accuracy = {acc_final:.2%},  Avg Tokens = {avg_tok_final:.1f}")
    print("──────────────────────────────────────────────────────")

    with open("criterion_eval.jsonl", "w", encoding="utf-8") as outf:
        for rec in records:
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

    """
    Example usage:
    
    dataset: 
    - gsm8k
    - math500

    python -m reasoning_output.criterion_test \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --dataset math500 \
        --num_samples 3
    """
