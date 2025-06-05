from __future__ import annotations

"""
Quick sanity-check for the "reasoning-output" phase using a **local
Hugging Face chat model**.

Example:
    python -m reasoning_output.quick_test \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --num_samples 1 \
        --max-new 1024
"""

import argparse
import re
import sys
import torch
from typing import List

import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,            # HF's dynamic cache implementation
)
from reasoning_output.perplexity import split_sentences
from input_prompt.src.prompt_engine import build_plain_prompt
import json
from tqdm import tqdm
from utils import set_seed
from dataset.gsm8k_loader import GSM8KLoader

#───────────────────────────────────────────────────────────────────────────────
#  Helpers
#───────────────────────────────────────────────────────────────────────────────

DEVICE = None  # will be set once the model is loaded
DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*(.+)", re.I)
set_seed()

def load_model(repo_or_path: str):
    """
    Load tokenizer and model from Hugging Face, add <leap> tokens.
    """
    global DEVICE

    # Load tokenizer, add special <leap> tokens if not present
    tok = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True)
    SPECIAL_TOKENS = ["<leap>", "</leap>"]
    added = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Load model with device_map="auto" for multi-GPU/CPU support
    mdl = AutoModelForCausalLM.from_pretrained(
        repo_or_path,
        torch_dtype=DTYPE,
        device_map="auto",
        use_cache=True,
    )

    # Resize embeddings if new tokens were added
    if added > 0:
        mdl.resize_token_embeddings(len(tok))

    # Set main device based on model parameters
    DEVICE = next(mdl.parameters()).device

    return tok, mdl

def top_p_sample(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9) -> int:
    """
    Sample a single token ID from logits using nucleus (top-p) sampling.
    """
    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask   = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()  # Renormalize

    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_idx[next_token].item()

#───────────────────────────────────────────────────────────────────────────────
#  Token-stream generation with LoT injection using Hugging Face Cache
#───────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def stream_with_leap(
    question: str,
    tok,
    mdl,
    max_new: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9
) -> tuple[str, str]:
    """
    Generate a normal reasoning chain, prune the cache at a pivot, inject a <leap>
    snippet, and continue with Leap-of-Thought (LoT) generation.
    """

    # Build and encode the prompt
    prompt = build_plain_prompt(question=question, leap=True, fewshot=False)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(DEVICE)

    # Generate normal reasoning with cache
    generated_ids: List[int] = []

    out = mdl(input_ids=prompt_ids, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    for _ in range(max_new):
        nxt = top_p_sample(last_logits[0], temperature, top_p)
        if nxt == tok.eos_token_id:
            break
        generated_ids.append(nxt)
        new_id_tensor = torch.tensor([[nxt]], device=DEVICE)
        out = mdl(input_ids=new_id_tensor, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

    normal_text = tok.decode(generated_ids, skip_special_tokens=False)

    # Split text and determine pivot for pruning
    pre_answer_text = ANSWER_RE.split(normal_text)[0]
    sents = split_sentences(pre_answer_text)
    if len(sents) == 0:
        pivot_char_idx = len(pre_answer_text)
    else:
        half = max(len(sents) // 2, 0)
        pivot_char_idx = pre_answer_text.find(sents[half])
        if pivot_char_idx < 0:
            pivot_char_idx = len(pre_answer_text)

    # Calculate token length to keep
    prefix_text = pre_answer_text[:pivot_char_idx]
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
    keep_token_len = prompt_ids.shape[-1] + len(prefix_ids)

    # Prune cache to prefix length
    cache.crop(keep_token_len)

    # Inject <leap> snippet
    leap_text = "<leap>I have a new idea to make it quick and clever. "
    leap_ids = tok(leap_text, add_special_tokens=False).input_ids
    leap_tensor = torch.tensor([leap_ids], device=DEVICE)

    out = mdl(input_ids=leap_tensor, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    # Generate LoT reasoning
    lot_gen_ids: List[int] = []
    for _ in range(max_new - keep_token_len):
        nxt = top_p_sample(last_logits[0], temperature, top_p)
        if nxt == tok.eos_token_id:
            break
        lot_gen_ids.append(nxt)
        new_id_tensor = torch.tensor([[nxt]], device=DEVICE)
        out = mdl(input_ids=new_id_tensor, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

    # Construct final LoT text
    lot_text = prefix_text + leap_text + tok.decode(lot_gen_ids, skip_special_tokens=False)

    return prompt, normal_text, lot_text

def extract_answer(text: str) -> str:
    """Return the raw answer text following the ``**Answer:**`` marker."""
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def normalize_answer(ans: str) -> str:
    """Normalize predicted or gold answers for robust comparison."""
    ans = ans.strip()
    # Remove surrounding punctuation and common currency symbols
    ans = ans.strip(" .,!?")
    ans = ans.lstrip("$€£¥")
    # Remove thousands separators
    ans = ans.replace(",", "")

    # If the remaining text is purely numeric, standardize its format
    num_str = ans
    if re.fullmatch(r"-?\d+(?:\.\d+)?", num_str):
        try:
            num = float(num_str)
        except ValueError:
            pass
        else:
            if num.is_integer():
                ans = str(int(num))
            else:
                ans = ("%f" % num).rstrip("0").rstrip(".")
    return ans

#───────────────────────────────────────────────────────────────────────────────
#  CLI
#───────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    type=str, required=True)
    # ap.add_argument("--question", type=str, default="If Alice has 3 times as many apples as Bob and together they have 16, how many apples does Alice have?")
    # ap.add_argument("--data",     type=str,   default=None)
    ap.add_argument("--num_samples",  type=int, default=10)
    ap.add_argument("--max-new",  type=int,   default=1024)
    ap.add_argument("--temp",     type=float, default=0.6)
    ap.add_argument("--top-p",    type=float, default=0.9)
    args = ap.parse_args()

    tok, mdl = load_model(args.model)

    # prompt, normal_text, lot_text = stream_with_leap(
    #     args.question,
    #     tok,
    #     mdl,
    #     max_new=args.max_new,
    #     temperature=args.temp,
    #     top_p=args.top_p
    # )

    # print("──── Prompt ──────────────────────────────────────")
    # print(prompt)

    # print("\n──── Normal Reasoning ─────────────────────────────")
    # print(normal_text)
    # print("Length of normal reasoning (in tokens):", len(tok(normal_text).input_ids))

    # print("\n──── LoT ─────────────────────────────────────────")
    # print(lot_text)
    # print("Length of LoT reasoning (in tokens):", len(tok(lot_text).input_ids))

    # Read all JSONL lines
    # with open(args.data, "r", encoding="utf-8") as f:
    #     data = [json.loads(l) for l in f]

    n_examples = args.num_samples
    loader = GSM8KLoader(split="train", num_samples=args.num_samples)
    correct_normal = 0
    correct_lot = 0
    sum_tokens_normal = 0
    sum_tokens_lot = 0

    # (Optional) You can store per-sample details in a list if you want to inspect later
    records = []

    for sample in tqdm(loader, total=n_examples, desc="Evaluating"):
        question = sample.get("question", "").strip()
        gold_answers = [a.strip() for a in sample.get("answers", [])]

        # Generate both reasoning paths
        prompt, normal_text, lot_text = stream_with_leap(
            question,
            tok,
            mdl,
            max_new=args.max_new,
            temperature=args.temp,
            top_p=args.top_p
        )

        # Extract just the answer portion from each generated text
        ans_normal = extract_answer(normal_text)
        ans_lot = extract_answer(lot_text)

        # Normalize all answers for robust comparison
        norm_normal = normalize_answer(ans_normal)
        norm_lot = normalize_answer(ans_lot)
        norm_gold = [normalize_answer(ga) for ga in gold_answers]

        # Compare against any of the gold answers (case-insensitive)
        is_correct_normal = any(norm_normal.lower() == ga.lower() for ga in norm_gold)
        is_correct_lot = any(norm_lot.lower() == ga.lower() for ga in norm_gold)

        # Count tokens of the entire generated string (including CoT + answer)
        tok_count_normal = len(tok(normal_text).input_ids)
        tok_count_lot = len(tok(lot_text).input_ids)

        # Accumulate
        if is_correct_normal:
            correct_normal += 1
        if is_correct_lot:
            correct_lot += 1
        sum_tokens_normal += tok_count_normal
        sum_tokens_lot += tok_count_lot

        # (Optional) store details
        records.append({
            "question": question,
            "gold_answers": gold_answers,
            "ans_normal": ans_normal,
            "ans_lot": ans_lot,
            "norm_normal": norm_normal,
            "norm_lot": norm_lot,
            "correct_normal": is_correct_normal,
            "correct_lot": is_correct_lot,
            "tokens_normal": tok_count_normal,
            "tokens_lot": tok_count_lot,

            "normal_text": normal_text,
            "lot_text": lot_text
        })

        # Compute overall metrics
        acc_normal = correct_normal / n_examples if n_examples > 0 else 0.0
        acc_lot = correct_lot / n_examples if n_examples > 0 else 0.0
        avg_tok_normal = sum_tokens_normal / n_examples if n_examples > 0 else 0.0
        avg_tok_lot = sum_tokens_lot / n_examples if n_examples > 0 else 0.0

        # Print summary
        print("\n──────────────────────────────────────────────────────")
        print(f"Total examples       : {n_examples}")
        print(f"Normal Reasoning     : Accuracy = {acc_normal:.2%},  Avg Tokens = {avg_tok_normal:.1f}")
        print(f"LoT Reasoning        : Accuracy = {acc_lot:.2%},  Avg Tokens = {avg_tok_lot:.1f}")
        print("──────────────────────────────────────────────────────")

        # Optionally, write `records` to a JSONL or CSV for inspection
        with open("detailed_eval.jsonl", "w", encoding="utf-8") as outf:
            for rec in records:
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    sys.exit(main())