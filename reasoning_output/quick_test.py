"""
Quick sanity-check for the "reasoning-output" phase using a **local
Hugging Face chat model**.

Example:
    python -m reasoning_output.quick_test \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --question "If Alice has 3 times as many apples as Bob and together they have 16, how many apples does Alice have?" \
        --max-new 1024
"""

from __future__ import annotations
import argparse, re, sys, torch, math
from typing import List
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from reasoning_output.perplexity import split_sentences
from input_prompt.src.prompt_engine   import build_plain_prompt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ANSWER_RE = re.compile(r"\*\*Answer:\*\*", re.I)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_model(repo_or_path: str):
    tok = AutoTokenizer.from_pretrained(repo_or_path, use_fast=True)
    SPECIAL_TOKENS = ["<leap>", "</leap>"]
    added = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    mdl = AutoModelForCausalLM.from_pretrained(repo_or_path, torch_dtype=DTYPE, device_map="auto")
    if added: # > 0 the first time you run this
        mdl.resize_token_embeddings(len(tok))

    return tok, mdl


def top_p_sample(logits, temperature: float = 0.7, top_p: float = 0.9):
    """Very small top-p sampler (no batching)."""
    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)

    # sort
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask   = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()  # renorm

    next_token = torch.multinomial(sorted_probs, 1)
    return sorted_idx[next_token]


def truncate_past(past, keep_len: int):
    """Slice key & value tensors in-place so only the first `keep_len`
       tokens remain in the cache.  Works for HF GPT-like models."""
    new_past = []
    for k, v in past: # (key, value)
        new_past.append((
            k[..., :keep_len, :].contiguous(),
            v[..., :keep_len, :].contiguous()
        ))
    return tuple(new_past)


# ─────────────────────────────────────────────────────────────────────────────
#  Token-stream generation with LoT injection
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def stream_with_leap(question: str, tok, mdl,
                     max_new: int = 512,
                     temperature: float = 0.7,
                     top_p: float = 0.9):

    # 0) build plain prompt (few-shot, no leap) and encode once
    prompt = build_plain_prompt(question=question, leap=True, fewshot=True)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(mdl.device)

    # 1) run **token-stream** to get the COMPLETE normal reasoning
    past, last_logits, generated = None, None, []
    ids = prompt_ids
    for _ in range(max_new):
        out   = mdl(input_ids=ids, past_key_values=past, use_cache=True)
        past  = out.past_key_values
        last_logits = out.logits[:, -1, :]
        nxt   = top_p_sample(last_logits[0], temperature, top_p)
        generated.append(nxt.item())
        char  = tok.decode([nxt])
        if ANSWER_RE.search(char):                      # crude answer stop
            break
        ids = nxt.unsqueeze(0).unsqueeze(0)

    normal_text = tok.decode(generated, skip_special_tokens=True)

    # 2) decide pivot - half the sentences
    sents = split_sentences(ANSWER_RE.split(normal_text)[0])
    half  = math.ceil(len(sents)/2)
    cut_idx = normal_text.find(sents[half])
    keep_ids = tok(normal_text[:cut_idx], add_special_tokens=False).input_ids

    # 3) Truncate the KV cache and the context
    pivot_tok_len = prompt_ids.size(1) + len(keep_ids)
    past = truncate_past(past, pivot_tok_len)

    # 4) inject <leap> ... without re-encoding the prefix
    leap_text = "<leap>I have a new idea to make it quick and clever. "
    leap_ids  = tok(leap_text, add_special_tokens=False).input_ids
    ids = torch.tensor([leap_ids], device=mdl.device)
    out = mdl(input_ids=ids, past_key_values=past, use_cache=True)
    
    past = out.past_key_values
    last_logits = out.logits[:, -1, :]

    # 5) continue sampling until answer or budget
    lot_gen = []
    len_up_to_now = len(keep_ids) + len(leap_ids)
    for _ in range(max_new - len_up_to_now):
        nxt = top_p_sample(last_logits[0], temperature, top_p)
        lot_gen.append(nxt.item())
        char = tok.decode([nxt])

        out = mdl(input_ids=nxt.unsqueeze(0).unsqueeze(0), past_key_values=past, use_cache=True)
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]

    lot_text = (
        tok.decode(keep_ids, skip_special_tokens=True) +
        leap_text +
        tok.decode(lot_gen, skip_special_tokens=True)
    )
    return normal_text, lot_text


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
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
        args.question, tok, mdl,
        max_new=args.max_new,
        temperature=args.temp,
        top_p=args.top_p
    )

    print("\n──── Normal Resoning ─────────────────────────────")
    print(normal_text)
    print("Length of normal reasoning:", len(tok(normal_text).input_ids))

    print("\n──── LoT ────────────────")
    print(lot_text)
    print("Length of LoT reasoning:", len(tok(lot_text).input_ids))


if __name__ == "__main__":
    sys.exit(main())
