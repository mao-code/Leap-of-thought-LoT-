"""
Quick sanity-check for the "reasoning-output" phase using a **local
Hugging Face chat model**.

Example:
    python -m reasoning_output.quick_test \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --question "If Alice has 3 times as many apples as Bob and together they have 16, how many apples does Alice have?" \
        --max-new 1024 > output.txt
"""

"""
Quick sanity-check for the "reasoning-output" phase using a **local
Hugging Face chat model**.

Example:
    python -m reasoning_output.quick_test \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --question "If Alice has 3 times as many apples as Bob and together they have 16, how many apples does Alice have?" \
        --max-new 1024 > output.txt
"""

from __future__ import annotations
import argparse
import re
import sys
import torch
from typing import List

import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,            # HF’s dynamic cache implementation
)
from reasoning_output.perplexity import split_sentences
from input_prompt.src.prompt_engine import build_plain_prompt

#───────────────────────────────────────────────────────────────────────────────
#  Helpers
#───────────────────────────────────────────────────────────────────────────────

DEVICE = None  # will be set once the model is loaded
DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ANSWER_RE = re.compile(r"\*\*Answer:\*\*", re.I)

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
    max_new: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> tuple[str, str]:
    """
    Generate a normal reasoning chain, prune the cache at a pivot, inject a <leap>
    snippet, and continue with Leap-of-Thought (LoT) generation.
    """

    # Build and encode the prompt
    prompt = build_plain_prompt(question=question, leap=True, fewshot=True)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(DEVICE)

    # Generate normal reasoning with cache
    generated_ids: List[int] = []
    cache = DynamicCache()  # Explicitly initialize cache

    out = mdl(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    for _ in range(max_new):
        nxt = top_p_sample(last_logits[0], temperature, top_p)
        generated_ids.append(nxt)
        new_id_tensor = torch.tensor([[nxt]], device=DEVICE)
        out = mdl(input_ids=new_id_tensor, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

    normal_text = tok.decode(generated_ids, skip_special_tokens=True)

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
    keep_token_len = len(prefix_ids)  # Only prefix tokens, not prompt

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
        lot_gen_ids.append(nxt)
        new_id_tensor = torch.tensor([[nxt]], device=DEVICE)
        out = mdl(input_ids=new_id_tensor, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

    # Construct final LoT text
    lot_text = prefix_text + leap_text + tok.decode(lot_gen_ids, skip_special_tokens=True)

    return normal_text, lot_text

#───────────────────────────────────────────────────────────────────────────────
#  CLI
#───────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--max-new",  type=int,   default=1024)
    ap.add_argument("--temp",     type=float, default=0.7)
    ap.add_argument("--top-p",    type=float, default=0.9)
    args = ap.parse_args()

    tok, mdl = load_model(args.model)

    normal_text, lot_text = stream_with_leap(
        args.question,
        tok,
        mdl,
        max_new=args.max_new,
        temperature=args.temp,
        top_p=args.top_p
    )

    print("\n──── Normal Reasoning ─────────────────────────────")
    print(normal_text)
    print("Length of normal reasoning (in tokens):", len(tok(normal_text).input_ids))

    print("\n──── LoT ─────────────────────────────────────────")
    print(lot_text)
    print("Length of LoT reasoning (in tokens):", len(tok(lot_text).input_ids))

if __name__ == "__main__":
    sys.exit(main())