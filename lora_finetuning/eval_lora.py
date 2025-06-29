import argparse
import json
import random
import re
from pathlib import Path
from typing import List

from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*([A-D])")


def load_adapter(base_model: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    special_tokens = ["<leap>", "</leap>"]
    if any(t not in tokenizer.get_vocab() for t in special_tokens):
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_path)
    if len(special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


def build_choices(correct: str, pool: List[str]) -> tuple[List[str], str]:
    pool = [a for a in pool if a != correct]
    distractors = random.sample(pool, k=3)
    options = distractors + [correct]
    random.shuffle(options)
    letter = "ABCD"[options.index(correct)]
    return options, letter


def evaluate(dataset_path: str, base_model: str, adapter: str, log_file: str):
    tokenizer, model = load_adapter(base_model, adapter)
    data = [json.loads(l) for l in Path(dataset_path).read_text().splitlines()]
    all_answers = [ex["answer"] for ex in data]

    records = []
    correct = 0
    total_tokens = 0

    for ex in tqdm(data, desc="Evaluating"):
        options, gold_letter = build_choices(ex["answer"], all_answers)
        prompt = (
            f"Problem: {ex['question']}\n"
            "Options:\n" +
            "\n".join(f"{l}. {o}" for l, o in zip("ABCD", options)) +
            "\n\nReason step by step and finish with '**Answer:** <letter>'."
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=256)
        text = tokenizer.decode(out[0], skip_special_tokens=False)
        m = ANSWER_RE.search(text)
        pred_letter = m.group(1) if m else None
        is_correct = pred_letter == gold_letter
        correct += int(is_correct)
        tokens_generated = out.shape[-1] - inputs.input_ids.shape[-1]
        total_tokens += tokens_generated
        records.append({
            "question": ex["question"],
            "options": dict(zip("ABCD", options)),
            "gold_letter": gold_letter,
            "pred_letter": pred_letter,
            "correct": is_correct,
            "reasoning": text,
            "uses_leap": "<leap>" in text,
            "tokens": tokens_generated,
        })

    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    summary = {"accuracy": accuracy, "avg_tokens": avg_tokens}
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model with MC questions")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Base model name")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--log", type=str, default="lora_eval_log.json")
    args = parser.parse_args()

    evaluate(args.dataset, args.model, args.adapter, args.log)


if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m lora_finetuning.eval_lora \
        --dataset dataset/distilled_data/alot_dataset_o4_mini.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --adapter lora_finetuning/lora_adapter/1_5B_o3/ \
        --log lora_eval_log.json
    """

