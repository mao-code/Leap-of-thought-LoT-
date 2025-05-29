import argparse, json, pprint
from input_prompt.src.evaluator import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="input_prompt/data/gsm8k_demo.jsonl")
    parser.add_argument("--fewshot", action="store_true")
    args = parser.parse_args()

    summary, _ = evaluate(args.data, fewshot=args.fewshot)
    pprint.pp(summary)

if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m input_prompt.src.main --fewshot
    """
