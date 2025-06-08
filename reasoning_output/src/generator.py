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
import math
from reasoning_output.src.perplexity import sentence_perplexity

from model_provider import get_model, BaseModel


class LeapGenerator:
    """Generate baseline reasoning and optionally a leap extension."""

    def __init__(self):
        self.base_model: BaseModel = get_model()
        if not hasattr(self.base_model, "model"):
            raise RuntimeError("LeapGenerator currently requires a local HF model")
        self.mdl: AutoModelForCausalLM = self.base_model.model
        self.tok: AutoTokenizer = self.base_model.tokenizer
        self.device = next(self.mdl.parameters()).device

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


    def generate(self, question: str, fewshot: bool = False) -> dict:
        """Generate reasoning and dynamically insert a <leap> once RWP is high."""

        prompt = build_plain_prompt(question=question, fewshot=fewshot, leap=True)
        prompt_ids = self.tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

        # --- first pass: generate until RWP exceeds threshold ---
        out = self.mdl(input_ids=prompt_ids, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

        gen_ids: List[int] = []
        sent_start = 0
        context = ""
        pps: List[float] = []
        pivot_token = None

        for _ in range(settings.max_tokens):
            nxt = self._top_p_sample(last_logits[0], settings.temperature, 0.9)
            if nxt == self.tok.eos_token_id:
                break
            gen_ids.append(nxt)
            nxt_tensor = torch.tensor([[nxt]], device=self.device)
            out = self.mdl(input_ids=nxt_tensor, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            last_logits = out.logits[:, -1, :]

            text_piece = self.tok.decode(gen_ids[sent_start:], skip_special_tokens=False)
            if text_piece.endswith(('.', '!', '?')):
                sent = text_piece
                pp = sentence_perplexity(sent, context, self.base_model)
                pps.append(pp)
                context += sent + " "
                sent_start = len(gen_ids)
                window = pps[-settings.window:]
                rwp = math.prod(window) ** (1 / len(window))
                if rwp >= settings.rwp_threshold:
                    pivot_token = len(gen_ids) - len(self.tok(sent, add_special_tokens=False).input_ids)
                    break

        normal_ids = gen_ids[:sent_start]
        normal_text = self.tok.decode(normal_ids, skip_special_tokens=False)

        record = {
            "question": question,
            "first_answer": normal_text,
            "trigger_leap": pivot_token is not None,
        }
        if pivot_token is None:
            return record

        # --- run model up to prefix ---
        prefix_ids = prompt_ids[0].tolist() + normal_ids
        prefix_tensor = torch.tensor([prefix_ids], device=self.device)
        out = self.mdl(input_ids=prefix_tensor, use_cache=True)
        cache = out.past_key_values
        last_logits = out.logits[:, -1, :]

        # --- inject leap and continue generation ---
        leap_text = "<leap>I have a new idea to make it quick and clever. "
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

        lot_text = normal_text + leap_text + self.tok.decode(lot_ids, skip_special_tokens=False)
        record["leap_answer"] = lot_text
        return record


def dump_record(rec: dict) -> None:
    ts = datetime.utcnow().isoformat()
    out_path = Path(settings.log_dir) / "reasoning_records.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, **rec}) + "\n")
