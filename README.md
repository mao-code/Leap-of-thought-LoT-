# Leap of Thought (LoT)

This project explores **Leap‑of‑Thought** reasoning with language models.  A model produces a normal chain‑of‑thought and, once a criterion is met, leaps to a new idea using the special `<leap>` token.

## Repository layout
- `input_prompt/` – utilities for LoT‑1 style prompt experiments
- `reasoning_output/` – implementation of LoT‑2 with reasoning‑wise perplexity (RWP)
- `dataset/` – small helpers to stream GSM8K examples
- `config.py` – runtime settings

## Quick sanity check
`reasoning_output/quick_test.py` performs a simple cut‑and‑paste leap.  It does **not** use RWP but is handy to verify your model setup.

```bash
python -m reasoning_output.quick_test --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --num_samples 1
```

## Criterion‑based evaluation
The real LoT‑2 pipeline monitors reasoning‑wise perplexity as it generates.  When perplexity spikes, it injects a `<leap>` snippet and continues.  Try it interactively via `criterion_test.py` or run a small dataset evaluation:

```bash
# Run a handful of samples and show accuracy
LOT_PROVIDER=local python -m reasoning_output.criterion_test --num_samples 3

# Evaluate a JSONL dataset (see `input_prompt/data/gsm8k_demo.jsonl`)
LOT_PROVIDER=local python -m reasoning_output.main --fewshot
```

Evaluation summaries and detailed records are stored under `logs/`.
