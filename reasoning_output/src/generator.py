from __future__ import annotations

"""Generation utilities for LoT-2 using a token-by-token loop."""

import json
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import settings
from input_prompt.src.prompt_engine import build_plain_prompt
from reasoning_output.src.perplexity import sentence_pp, rwp_from_pp
from reasoning_output.src.criteria import should_trigger_leap

DEVICE = None  # will be set once the model is loaded
DTYPE  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LEAP_HINT = "<leap>Aha! I have a new idea to make it quick and clever. "

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


class LeapGenerator:
    """Generate baseline reasoning and optionally a leap extension."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.tok, self.mdl = load_model(model_name)
        self.device = self.mdl.device

    def _top_p_sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_token = torch.multinomial(sorted_probs, 1)
        return sorted_idx[next_token].item()

    @torch.inference_mode()
    def generate(self, question: str, fewshot: bool = False, use_window_rwp=False) -> dict:
        """Generate reasoning and dynamically insert a <leap> once RWP is high."""

        prompt = build_plain_prompt(question=question, fewshot=fewshot, leap=True)
        prompt_ids = self.tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

        # --- first pass: generate until RWP exceeds threshold ---
        out = self.mdl(input_ids=prompt_ids, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

        gen_ids: List[int] = []
        sent_start = 0
        pps: List[float] = []
        rwps: List[float] = []
        cur_logps: List[float] = []
        pivot_token = None
        has_leaped = False

        for _ in range(settings.max_tokens):
            nxt = self._top_p_sample(last_logits[0], settings.temperature, 0.9)
            logp = F.log_softmax(last_logits[0], dim=-1)[nxt].item()
            if nxt == self.tok.eos_token_id:
                break
            gen_ids.append(nxt)
            cur_logps.append(logp)
            nxt_tensor = torch.tensor([[nxt]], device=self.device)
            out = self.mdl(input_ids=nxt_tensor, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last_logits = out.logits[:, -1, :]

            text_piece = self.tok.decode(gen_ids[sent_start:], skip_special_tokens=False)
            if text_piece.endswith((".", "!", "?")) and not has_leaped:
                pp = sentence_pp(cur_logps)
                pps.append(pp)
                sent_start = len(gen_ids)
                cur_logps = []

                # RWP scheme
                should_leap, rwp =  should_trigger_leap(pps, use_window_rwp)

                # Budget Control Scheme
                should_leap = len(gen_ids) > 64

                rwps.append(rwp)
                if should_leap:
                    pivot_token = sent_start - len(self.tok(text_piece, add_special_tokens=False).input_ids)
                    has_leaped = True
                    
        normal_ids = gen_ids[:]
        normal_reasoning_text = self.tok.decode(normal_ids, skip_special_tokens=False)
        
        cut = pivot_token if pivot_token is not None else sent_start
        normal_ids_cut = gen_ids[:cut]
        normal_reasoning_cut_text = self.tok.decode(normal_ids_cut, skip_special_tokens=False)

        record = {
            "question": question,
            "normal_reasoning_text": normal_reasoning_text,
            "trigger_leap": pivot_token is not None,
            "leap_from_sentence": text_piece,
            "pps_trajectory": pps,
            "rwp_trajectory": rwps,
        }
        if pivot_token is None:
            return record

        # --- run model up to prefix ---
        prefix_ids = prompt_ids[0].tolist() + normal_ids_cut
        keep_prefix_len = len(prefix_ids)
        cache.crop(keep_prefix_len)

        # --- inject leap and continue generation ---
        leap_text = LEAP_HINT
        leap_ids = self.tok(leap_text, add_special_tokens=False).input_ids
        leap_tensor = torch.tensor([leap_ids], device=self.device)
        out = self.mdl(input_ids=leap_tensor, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

        lot_ids: List[int] = []
        for _ in range(settings.max_tokens - len(prefix_ids)):
            nxt = self._top_p_sample(last_logits[0], settings.temperature, 0.9)
            if nxt == self.tok.eos_token_id:
                break
            lot_ids.append(nxt)
            new_tensor = torch.tensor([[nxt]], device=self.device)
            out = self.mdl(input_ids=new_tensor, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last_logits = out.logits[:, -1, :]

        lot_text = normal_reasoning_cut_text + leap_text + self.tok.decode(lot_ids, skip_special_tokens=False)
        record["leap_reasoning_text"] = lot_text

        return record


def dump_record(rec: dict) -> None:
    ts = datetime.utcnow().isoformat()
    out_path = Path(settings.log_dir) / "reasoning_records.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, **rec}) + "\n")

