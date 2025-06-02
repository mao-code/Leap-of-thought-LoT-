"""Abstraction layer between local‑HF and OpenAI‑compatible chat endpoints."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# OpenAI‑compatible sdk (works for Groq / TogetherAI)
import openai

from config import settings

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Minimal interface each connector must satisfy."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Return OpenAI-style response dict incl. `choices[0].message.content` & `usage`."""

    @abstractmethod
    def sentence_logprobs(self, sentence: str, context: str = "") -> List[float]:
        """Return token-level log-probs for *sentence* conditioned on *context*."""


# ─────────────────────────────────────────────────────────────────────────────
# Local HuggingFace implementation
# ─────────────────────────────────────────────────────────────────────────────
class LocalHFModel(BaseModel):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(settings.model_name)
            .to(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            .eval()
        )
        self.eot = self.tokenizer.eos_token or "<|endoftext|>"

    # Simple plain‑text prompt builder (role tags → //USER//, //SYS//, etc.)
    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        out_lines = []
        for m in messages:
            role = m["role"].upper()
            out_lines.append(f"//{role}//: {m['content']}")
        out_lines.append("//ASSISTANT//:")
        return "\n".join(out_lines)

    @torch.no_grad()
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        prompt = self._to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", settings.max_tokens),
            temperature=settings.temperature,
            do_sample=True,
        )
        generated = out_ids[:, inputs["input_ids"].shape[1] :]
        content = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        # We approximate usage statistics
        usage = {
            "prompt_tokens": int(inputs["input_ids"].numel()),
            "completion_tokens": int(generated.numel()),
            "total_tokens": int(inputs["input_ids"].numel() + generated.numel()),
        }

        return {
            "choices": [{"message": {"content": content}}],
            "usage": usage,
        }

    @torch.no_grad()
    def sentence_logprobs(self, sentence: str, context: str = "") -> List[float]:
        # Build combined sequence but mask loss to *sentence* only
        full_text = context + sentence
        enc_full = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        enc_ctx = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        ctx_len = enc_ctx["input_ids"].shape[1]

        # labels: we only care about sentence tokens (shifted by ctx_len)
        labels = enc_full["input_ids"].clone()
        labels[:, :ctx_len] = -100  # ignore context tokens in loss

        out = self.model(**enc_full, labels=labels)
        # Get per‑token log‑probabilities
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        sent_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        # Remove ignored positions (‑100)
        mask = shift_labels != -100
        return sent_logps[0][mask[0]].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI‑compatible REST implementation (Groq / TogetherAI)
# ─────────────────────────────────────────────────────────────────────────────
class OpenAICompatibleModel(BaseModel):
    def __init__(self):
        openai.api_key = settings.groq_api_key or settings.together_api_key
        if settings.api_base_url:
            openai.base_url = settings.api_base_url
        self.model_name = settings.model_name

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        # Ask for logprobs so we can re‑use for first assistant draft
        resp = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=settings.temperature,
            max_tokens=kwargs.get("max_tokens", settings.max_tokens),
            logprobs=True,  # ← togetherAI & groq both support this flag
        )
        return resp.model_dump()

    def sentence_logprobs(self, sentence: str, context: str = "") -> List[float]:
        # We proxy through the completions endpoint with echo=True to get logprobs
        prompt = context + sentence
        resp = openai.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        # The last choice contains logprobs for prompt tokens; extract sentence piece
        token_data = resp.choices[0].logprobs
        # Convert into list[float]
        logps = token_data.token_logprobs[-len(token_data.tokens) :]
        return [lp for lp in logps if lp is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model() -> BaseModel:
    if settings.provider.lower() == "local":
        logger.info("Using local HuggingFace model: %s", settings.model_name)
        return LocalHFModel()
    elif settings.provider.lower() == "api":
        logger.info("Using OpenAI-compatible API model: %s", settings.model_name)
        return OpenAICompatibleModel()
    else:
        raise ValueError(f"Unknown provider: {settings.provider}")