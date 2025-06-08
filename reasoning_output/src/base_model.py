from __future__ import annotations

"""Minimal local model wrapper used for LoT reasoning."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import settings


class BaseModel:
    """Simple HuggingFace causal LM wrapper with log-prob utilities."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
        special = ["<leap>", "</leap>"]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": special})

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=dtype,
            device_map="auto",
            use_cache=True,
        )
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = next(self.model.parameters()).device
        self.model.eval()

_model: BaseModel | None = None


def get_model() -> BaseModel:
    """Return a singleton :class:`BaseModel` instance."""
    global _model
    if _model is None:
        _model = BaseModel()
    return _model
