"""Two-pass generation pipeline (baseline → criterion → optional leap)."""
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

from config import settings
from input_prompt.src.prompt_engine import build_prompt
from model_provider import get_model
from reasoning_output.criteria import should_trigger_leap

# We reuse ANSWER_RE from the old evaluator so users can diff easily
import re
ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*(.+)", re.I)


class LeapGenerator:
    def __init__(self):
        self.model = get_model()

    def _call(self, messages):
        resp = self.model.chat(messages, max_tokens=settings.max_tokens)
        content = resp["choices"][0]["message"]["content"]
        return content, resp["usage"]

    def generate(self, question: str, fewshot: bool = False):
        # ── Pass‑1  (no leap) ──────────────────────────
        msgs = build_prompt(question, leap=False, fewshot=fewshot)
        first_content, first_usage = self._call(msgs)

        # Decide whether to trigger leap
        need_leap = should_trigger_leap(first_content, self.model)

        record = {
            "question": question,
            "first_answer": first_content,
            "first_usage": first_usage,
            "trigger_leap": need_leap,
        }

        if not need_leap:
            return record  # good enough

        # ── Pass‑2  (with <leap>) ──────────────────────
        leap_msgs = build_prompt(question, leap=True, fewshot=fewshot)
        leap_content, leap_usage = self._call(leap_msgs)
        record.update({
            "leap_answer": leap_content,
            "leap_usage": leap_usage,
        })
        return record


def dump_record(rec):
    ts = datetime.utcnow().isoformat()
    out_path = Path(settings.log_dir) / "reasoning_records.jsonl"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, **rec}) + "\n")