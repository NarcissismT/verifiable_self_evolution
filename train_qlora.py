#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


MODEL_DEFAULTS = {
    "qwen2.5-7b-instruct": {
        "repository": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "d1e200f",
        "learning_rate": 1e-4,
    },
    "qwen2.5-14b-instruct": {
        "repository": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "336cd7f",
        "learning_rate": 8e-5,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT on verifier-approved records")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-profile", choices=sorted(MODEL_DEFAULTS))
    parser.add_argument("--revision")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--effective-nonpad-tokens-per-batch", type=int, default=32768)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = args.model_profile
    if profile is None:
        profile = next(
            (name for name, spec in MODEL_DEFAULTS.items() if spec["repository"] == args.model),
            None,
        )
    if profile is None:
        raise ValueError("--model-profile is required for an unregistered model")
    defaults = MODEL_DEFAULTS[profile]
    if args.revision is None:
        args.revision = defaults["revision"]
    if args.learning_rate is None:
        args.learning_rate = defaults["learning_rate"]
    configured_tokens = args.batch_size * args.gradient_accumulation * args.max_length
    if configured_tokens != args.effective_nonpad_tokens_per_batch:
        raise ValueError(
            "batch-size * gradient-accumulation * max-length must equal "
            "effective-nonpad-tokens-per-batch"
        )
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {args.output}")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if not rows:
        raise ValueError("empty training dataset")
    required = {"prompt", "completion", "task_id", "verification_digest"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"dataset row {index} is missing {sorted(missing)}")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise RuntimeError(
            "install the optional training dependencies with `pip install -e '.[train]'`"
        ) from error

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError(
            f"tokenizer EOS mismatch: expected <|im_end|>, got {tokenizer.eos_token!r}"
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )

    class VerifiedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = rows[index]
            prompt = f"<task>\n{row['prompt']}\n</task>\n<answer>\n"
            completion = row["completion"] + "\n</answer>" + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full = tokenizer(
                prompt + completion,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_length,
            )["input_ids"]
            prompt_length = min(len(prompt_ids), len(full))
            return {
                "input_ids": full,
                "attention_mask": [1] * len(full),
                "labels": [-100] * prompt_length + full[prompt_length:],
            }

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [], "attention_mask": [], "labels": []
        }
        for item in features:
            padding = max_length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        weight_decay=args.weight_decay,
        optim="paged_adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.95,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=VerifiedDataset(),
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(args.output / "adapter"))
    tokenizer.save_pretrained(str(args.output / "adapter"))
    receipt = {
        "model": args.model,
        "model_profile": profile,
        "revision": args.revision,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "row_count": len(rows),
        "arguments": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)},
    }
    (args.output / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
