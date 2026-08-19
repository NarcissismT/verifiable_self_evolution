#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

from vse.freeze import check_freeze
from vse.hashing import content_hash, file_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-config QLoRA on verifier-approved records"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-profile", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--adapter-seed", type=int, required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _verify_file_manifest(root: Path, expected: dict[str, str], name: str) -> None:
    if not expected:
        raise ValueError(f"frozen {name} file hashes are missing")
    for relative, digest in sorted(expected.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"{name} file is outside model root: {relative}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_hash(path) != digest:
            raise ValueError(f"{name} file hash mismatch: {relative}")


def _load_training_rows(
    manifest_path: Path, mixture: dict[str, float], seed: int
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    declared_digest = manifest.pop("manifest_digest", "")
    if content_hash(manifest) != declared_digest:
        raise ValueError("dataset manifest digest mismatch")
    if manifest.get("mixture") != mixture:
        raise ValueError("dataset replay mixture differs from frozen configuration")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for name, relative in manifest["files"].items():
        path = (manifest_path.parent / relative).resolve()
        path.relative_to(manifest_path.parent.resolve())
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        if len(rows) != int(manifest["counts"][name]):
            raise ValueError(f"dataset row count mismatch: {name}")
        if any(row.get("bucket") != name for row in rows):
            raise ValueError(f"dataset bucket label mismatch: {name}")
        if mixture.get(name, 0.0) > 0.0 and not rows:
            raise ValueError(f"frozen replay bucket is empty: {name}")
        buckets[name] = rows
    integer_weights = {
        name: int(round(weight * 10)) for name, weight in mixture.items()
    }
    cycles = max(
        (len(buckets[name]) + integer_weights[name] - 1) // integer_weights[name]
        for name in mixture
    )
    rng = random.Random(seed)
    scheduled: list[dict[str, Any]] = []
    for name in sorted(mixture):
        source = buckets[name]
        scheduled.extend(
            source[index % len(source)]
            for index in range(cycles * integer_weights[name])
        )
    rng.shuffle(scheduled)
    return scheduled


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    freeze_report = check_freeze(config, args.run_root)
    if not freeze_report.ready:
        raise RuntimeError(
            "formal QLoRA is blocked by freeze-check: "
            + ", ".join(freeze_report.failures)
        )
    profiles = {item["model_id"]: item for item in config["models"]}
    if args.model_profile not in profiles:
        raise ValueError("model profile is not registered in the frozen config")
    model_spec = profiles[args.model_profile]
    adapter_seeds = tuple(int(seed) for seed in config["seeds"]["adapter"])
    if args.adapter_seed not in adapter_seeds:
        raise ValueError("adapter seed is outside the frozen registry")
    expected_container = config["containers"]["train_image_digest"]
    if args.container_digest != expected_container:
        raise ValueError("runtime container digest differs from frozen config")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    _verify_file_manifest(
        args.model_path, model_spec["checkpoint_file_hashes"], "checkpoint"
    )
    _verify_file_manifest(
        args.model_path, model_spec["tokenizer_file_hashes"], "tokenizer"
    )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {args.output}")

    qlora = config["qlora"]
    tokenizer_config = config["tokenizer"]
    max_length = int(tokenizer_config["max_sequence_length"])
    effective_tokens = int(qlora["effective_nonpad_tokens_per_batch"])
    if effective_tokens % max_length:
        raise ValueError("effective token batch must be divisible by sequence length")
    gradient_accumulation = effective_tokens // max_length
    rows = _load_training_rows(
        args.dataset_manifest,
        {key: float(value) for key, value in qlora["replay_mixture"].items()},
        args.adapter_seed,
    )

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
        raise RuntimeError("frozen training dependencies are unavailable") from error

    random.seed(args.adapter_seed)
    torch.manual_seed(args.adapter_seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=model_spec["revision"],
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = tokenizer_config["train_padding_side"]
    if tokenizer.eos_token != tokenizer_config["eos_token"]:
        raise ValueError("tokenizer EOS differs from frozen config")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    stream_ids: list[int] = []
    stream_labels: list[int] = []
    for row in rows:
        prompt = f"<task>\n{row['prompt']}\n</task>\n<answer>\n"
        completion = row["completion"] + "\n</answer>" + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        stream_ids.extend(prompt_ids + completion_ids)
        stream_labels.extend([-100] * len(prompt_ids) + completion_ids)
    block_count = len(stream_ids) // max_length
    if block_count < gradient_accumulation:
        raise ValueError("not enough verified tokens for one frozen optimizer update")
    usable = block_count * max_length
    stream_ids = stream_ids[:usable]
    stream_labels = stream_labels[:usable]

    class PackedVerifiedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return block_count

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            start = index * max_length
            end = start + max_length
            return {
                "input_ids": stream_ids[start:end],
                "attention_mask": [1] * max_length,
                "labels": stream_labels[start:end],
            }

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor([item[key] for item in features], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")
        }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        revision=model_spec["revision"],
        local_files_only=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(qlora["rank"]),
            lora_alpha=int(qlora["alpha"]),
            lora_dropout=float(qlora["dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=qlora["target_modules"],
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=float(qlora["epochs"]),
        learning_rate=float(qlora["learning_rate_by_model"][args.model_profile]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation,
        weight_decay=float(qlora["weight_decay"]),
        optim=qlora["optimizer"],
        adam_beta1=float(qlora["betas"][0]),
        adam_beta2=float(qlora["betas"][1]),
        lr_scheduler_type=qlora["scheduler"],
        warmup_ratio=float(qlora["warmup_ratio"]),
        max_grad_norm=float(qlora["gradient_clipping"]),
        gradient_checkpointing=bool(qlora["gradient_checkpointing"]),
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
        seed=args.adapter_seed,
        data_seed=args.adapter_seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=PackedVerifiedDataset(),
        data_collator=collate,
    )
    trainer.train()
    adapter_path = args.output / "adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    adapter_hashes = {
        str(path.relative_to(adapter_path)): file_hash(path)
        for path in sorted(adapter_path.rglob("*"))
        if path.is_file()
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pip_freeze": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines(),
    }
    receipt = {
        "experiment_id": config["experiment_id"],
        "config_digest": config["config_digest"],
        "model_profile": args.model_profile,
        "model_revision": model_spec["revision"],
        "adapter_seed": args.adapter_seed,
        "container_digest": args.container_digest,
        "dataset_manifest_sha256": file_hash(args.dataset_manifest),
        "packed_block_count": block_count,
        "sequence_length": max_length,
        "nonpad_tokens_per_optimizer_update": max_length * gradient_accumulation,
        "adapter_file_hashes": adapter_hashes,
        "environment": environment,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = content_hash(receipt)
    (args.output / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
