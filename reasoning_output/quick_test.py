"""
Quick sanity-check for the "reasoning-output" phase using a **local
Hugging Face chat model**.

Example:
    python -m reasoning_output.quick_test \
        --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --question "If Alice has 3 times as many apples as Bob and together they have 16, how many apples does Alice have?"
"""

from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

# ─────────────────────────────  Local config  ────────────────────────────── #

DEFAULT_MODEL_PATH = os.getenv("HF_CHAT_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE             = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# Prompt-builder from your repo
from input_prompt.src.prompt_engine import build_prompt
from reasoning_output.perplexity import split_sentences


ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*(.+)", re.I)


# ─────────────────────────  Minimal chat wrapper  ───────────────────────── #

class LocalChat:
    """
    Very small wrapper that mimics `client.chat.completions.create(...)`
    so you only touch the I/O layer, not the algorithmic core.
    """

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            device_map="auto",
        )
        # We rely on the tokenizer's chat template if it exists
        self.has_template = getattr(self.tokenizer, "chat_template", None) is not None

    def _render_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.has_template:                       # ◀ most modern chat models
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # ↓ Fallback: very simple role-tagged format
        rendered = []
        for msg in messages:
            role = msg["role"].capitalize()
            rendered.append(f"### {role}:\n{msg['content']}\n")
        rendered.append("### Assistant:\n")         # generation starts here
        return "".join(rendered)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        prompt = self._render_prompt(messages)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_cfg = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            output = self.model.generate(**input_ids, generation_config=gen_cfg)

        generated_text = self.tokenizer.decode(
            output[0, input_ids.input_ids.shape[-1]:], skip_special_tokens=True
        )
        return generated_text.strip()


# ─────────────────────────  Two-pass reasoning code  ────────────────────── #

def first_pass(
    question: str, chat: LocalChat, max_tokens: int, temperature: float
) -> tuple[List[dict], str]:
    """
    1st call: get full chain of thought **without** the final answer.
    """
    messages = build_prompt(question, fewshot=True, leap=True)
    reasoning_text = chat.chat_completion(messages, max_tokens, temperature)

    # Strip any accidental answer
    answer_match = ANSWER_RE.search(reasoning_text)
    if answer_match:
        reasoning_text = reasoning_text[:answer_match.start()].rstrip()

    return messages, reasoning_text


def _inject_leap(text: str) -> str:
    leap_snippet = (
        "<leap>I have a new idea. I can solve this in a more clever way. "
    )

    sents = split_sentences(text)
    if len(sents) > 1:
        leap_idx   = len(sents) // 2
        leap_sent  = sents[leap_idx]
        leap_start = text.find(leap_sent)
        text       = text[:leap_start] + leap_snippet + text[leap_start:]

    return text


def second_pass(
    base_messages: List[dict],
    reasoning_text: str,
    chat: LocalChat,
    max_tokens: int,
    temperature: float,
) -> tuple[List[dict], str]:
    """
    2nd call: feed back the (possibly partial) chain-of-thought with a
    <leap> idea, then request the model to finish and output only **Answer:** ...
    """
    assistant_msg = _inject_leap(reasoning_text)
    followup_user = (
        "Please continue from here, complete the reasoning, and output **only** "
        "the final answer in the form **Answer:**<number_or_word>."
    )

    messages = (
        base_messages
        + [{"role": "assistant", "content": assistant_msg}]
        + [{"role": "user", "content": followup_user}]
    )

    final_ans = chat.chat_completion(messages, max_tokens, temperature)
    return messages, final_ans


# ───────────────────────────────  CLI  ──────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Local path or HF repo-id of a chat-tuned model")
    parser.add_argument("--question", required=True, help="Problem statement")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    chat = LocalChat(args.model_path)

    base_msgs, reasoning = first_pass(
        args.question, chat, args.max_tokens, args.temperature
    )
    print("\n── First pass (reasoning only) ────────────────────────────────")
    print(reasoning)

    leap_msgs, final_answer = second_pass(
        base_msgs, reasoning, chat, args.max_tokens, args.temperature
    )

    print("\n── Leap Messages ─────────────────────────")
    for msg in leap_msgs:
        role = msg["role"].ljust(9)
        print(f"{role}: {msg['content']}")

    print("\n── Second pass (leap + final answer) ─────────────────────────")
    print(final_answer)
    print("Total words:", len(final_answer.split()))


if __name__ == "__main__":
    sys.exit(main())
