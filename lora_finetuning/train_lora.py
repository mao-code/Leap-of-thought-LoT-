import argparse
import os

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import wandb


def load_data(path: str, tokenizer, max_length: int = 1024):
    data = load_dataset("json", data_files=path, split="train")

    def _format(example):
        text = f"Problem: {example['question']}\n\n{example['solution']}"
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = data.map(_format, remove_columns=data.column_names)
    return tokenized


def create_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    special_tokens = ["<leap>", "</leap>"]
    if any(t not in tokenizer.get_vocab() for t in special_tokens):
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if len(special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description="Train LoRA on ALoT dataset")
    parser.add_argument("--dataset", type=str, default="alot_dataset.jsonl")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--output", type=str, default="lora_adapter")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wandb-project", type=str, default="ALoT")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=os.path.basename(args.output))

    tokenizer, model = create_model(args.model, args.lora_r, args.lora_alpha, args.lora_dropout)

    dataset = load_data(args.dataset, tokenizer, args.max_length)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        report_to=["wandb"],
        gradient_checkpointing=True
    )
    model.config.use_cache = False # Disable cache for gradient checkpointing

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    wandb.finish()


if __name__ == "__main__":
    main()

    """
    Please set WANDB_API_KEY in your environment variables to enable Weights & Biases logging.

    Example usage:
    python -m lora_finetuning.train_lora \
        --dataset dataset/distilled_data/alot_dataset_o3.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --output lora_finetuning/lora_adapter/1_5B_o3/ \
        --batch 4 \
        --epochs 3 \
        --lr 5e-5 \
        --wandb-project ALoT \
        --max-length 1024 \
        --lora-r 8 \
        --lora-alpha 32 \
        --lora-dropout 0.1
    """
