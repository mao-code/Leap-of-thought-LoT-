"""Run-time settings shared across the sub-package.

You can override every field via environment variables, e.g.
`export LOT_PROVIDER=api`, `export GROQ_API_KEY=...`.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Execution back‑end ────────────────────────────────────────────────────
    provider: str = Field(
        "local", env="LOT_PROVIDER",
        description="One of {'local', 'api'}.  Local ⇒ HuggingFace,  api ⇒ OpenAI-compatible"
    )

    # ── Model choices ────────────────────────────────────────────────────────
    # Together-AI models
        # deepseek-ai/DeepSeek-R1-Distill-Llama-70B (cost $2.00 per 1M tokens)
        # deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
        # deepseek-ai/DeepSeek-R1 (671B, cost $3/$7 per 1M input/output tokens)
    model_name: str = Field(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", env="LOT_MODEL_NAME"
    )
    # CUDA / MPS / CPU
    device: str | None = Field('cuda', env="LOT_DEVICE")

    # ── API keys (only required for provider=="api") ───────────────────────
    groq_api_key: str | None = Field(None, env="GROQ_API_KEY")
    together_api_key: str | None = Field(None, env="TOGETHER_API_KEY")
    # You may point to a custom base‑url if needed (Groq -> https://api.groq.com/openai/v1)
    api_base_url: str | None = Field(None, env="LOT_API_BASE_URL")

    # ── Generation & bookkeeping ─────────────────────────────────────────────
    temperature: float = Field(0.6, env="LOT_TEMPERATURE")
    max_tokens: int = Field(2048, env="LOT_MAX_TOKENS")
    log_dir: str = Field("logs", env="LOT_LOG_DIR")

    # ── RWP‑criterion hyper‑parameters ───────────────────────────────────────
    rwp_threshold: float = Field(50.0, env="LOT_RWP_THRESHOLD")
    window: int = Field(3, env="LOT_RWP_WINDOW", description="Sliding-window for RWP^{(w)}")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Ensure log directory exists
Path(settings.log_dir).mkdir(parents=True, exist_ok=True)