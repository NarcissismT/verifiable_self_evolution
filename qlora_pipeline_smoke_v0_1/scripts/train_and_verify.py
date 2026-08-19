#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any


LOSS_ON = {
    "assistant_plan",
    "assistant_action",
    "structured_result_explanation",
    "belief_update",
}
LOSS_OFF = {"system", "user", "tool_observation"}
ROLE_BY_SEGMENT = {
    "system": "system",
    "user": "user",
    "tool_observation": "tool",
    "assistant_plan": "assistant",
    "assistant_action": "assistant",
    "structured_result_explanation": "assistant",
    "belief_update": "assistant",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sealed_json(path: Path, digest_field: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    declared = str(value.get(digest_field, ""))
    blank = dict(value)
    blank[digest_field] = ""
    if not declared or hashlib.sha256(canonical_json(blank).encode("utf-8")).hexdigest() != declared:
        raise ValueError(f"{path.name}:{digest_field}_mismatch")
    return value


def verify_file_manifest(model_root: Path, expected_digest: str) -> dict[str, Any]:
    manifest = json.loads((model_root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_digest") != expected_digest:
        raise ValueError("model manifest/config digest mismatch")
    for item in manifest.get("files", []):
        path = (model_root / item["path"]).resolve()
        path.relative_to(model_root.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["bytes"]) or file_hash(path) != item["sha256"]:
            raise ValueError(f"model file mismatch: {item['path']}")
    return manifest


def verify_inputs(
    config_path: Path,
    freeze_path: Path,
    dataset_manifest_path: Path,
    model_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze = verify_sealed_json(freeze_path, "receipt_digest")
    dataset_manifest = verify_sealed_json(dataset_manifest_path, "manifest_digest")
    if config.get("study_kind") != "qlora_pipeline_smoke":
        raise ValueError("wrong study kind")
    for key in ("scientific_claims_allowed", "eligible_for_champion", "eligible_for_training_library"):
        if config.get(key) is not False or freeze.get(key) is not False:
            raise ValueError(f"smoke eligibility violation: {key}")
    if config["training"]["adapter_seeds"] != [17]:
        raise ValueError("only adapter seed 17 is allowed")
    if int(config["training"]["max_optimizer_updates"]) != 100:
        raise ValueError("smoke requires exactly 100 optimizer updates")
    if config["dataset"]["kind"] != "deterministic_synthetic_toy":
        raise ValueError("non-synthetic dataset rejected")
    if dataset_manifest.get("dataset_kind") != "deterministic_synthetic_toy":
        raise ValueError("dataset manifest is not synthetic")
    if dataset_manifest.get("formal_split_eligible") is not False:
        raise ValueError("formal split eligible data rejected")
    if dataset_manifest.get("eligible_for_training_library") is not False:
        raise ValueError("training-library eligible data rejected")
    if file_hash(config_path) != freeze["config_sha256"]:
        raise ValueError("config/freeze hash mismatch")
    if file_hash(dataset_manifest_path) != freeze["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest/freeze hash mismatch")
    runtime_digest = os.environ.get("VSE_RUNTIME_IMAGE_DIGEST", "")
    if runtime_digest != freeze["train_image_digest"]:
        raise ValueError("runtime image digest differs from frozen digest")
    if os.environ.get("VSE_NETWORK_POLICY") != "none":
        raise ValueError("VSE_NETWORK_POLICY must be none")
    model_manifest = verify_file_manifest(model_root, config["model"]["manifest_digest"])
    if model_manifest.get("revision") != config["model"]["revision"]:
        raise ValueError("model revision mismatch")
    dataset_path = (dataset_manifest_path.parent / dataset_manifest["dataset_file"]).resolve()
    dataset_path.relative_to(dataset_manifest_path.parent.resolve())
    if file_hash(dataset_path) != dataset_manifest["dataset_sha256"]:
        raise ValueError("dataset file hash mismatch")
    return config, dataset_manifest, dataset_path


def render_segment(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def encode_segmented_example(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    input_ids: list[int] = []
    labels: list[int] = []
    ranges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for segment in row.get("segments", []):
        kind = str(segment.get("segment_type", ""))
        role = str(segment.get("role", ""))
        if kind not in LOSS_ON | LOSS_OFF:
            raise ValueError(f"unknown segment type: {kind}")
        if role != ROLE_BY_SEGMENT[kind]:
            raise ValueError(f"role mismatch for segment: {kind}")
        token_ids = list(tokenizer.encode(
            render_segment(role, str(segment.get("content", ""))),
            add_special_tokens=False,
        ))
        if not token_ids:
            raise ValueError(f"empty tokenized segment: {kind}")
        start = len(input_ids)
        input_ids.extend(token_ids)
        labels.extend(token_ids if kind in LOSS_ON else [-100] * len(token_ids))
        ranges.append({"segment_type": kind, "start": start, "end": len(input_ids), "loss_on": kind in LOSS_ON})
        seen.add(kind)
    if seen != LOSS_ON | LOSS_OFF:
        raise ValueError("example does not contain the complete frozen segment set")
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if len(input_ids) != len(labels) or not input_ids:
        raise ValueError("invalid encoded example")
    trainable = sum(value != -100 for value in labels)
    masked = len(labels) - trainable
    if trainable <= 0 or masked <= 0:
        raise ValueError("segmented loss mask must contain trainable and masked tokens")
    for item in ranges:
        start = min(item["start"], len(labels))
        end = min(item["end"], len(labels))
        if end <= start:
            continue
        values = labels[start:end]
        if item["loss_on"] and any(value == -100 for value in values):
            raise ValueError(f"loss-on segment was masked: {item['segment_type']}")
        if not item["loss_on"] and any(value != -100 for value in values):
            raise ValueError(f"loss-off segment was unmasked: {item['segment_type']}")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "trainable_tokens": trainable,
        "masked_tokens": masked,
        "ranges": ranges,
    }


def load_rows(path: Path, expected_count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_count:
        raise ValueError("synthetic dataset row count mismatch")
    if any(row.get("source_kind") != "deterministic_synthetic_toy" for row in rows):
        raise ValueError("non-synthetic row rejected")
    if any(row.get("formal_split_eligible") is not False for row in rows):
        raise ValueError("formal split eligible row rejected")
    return rows


def adapter_hashes(adapter_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(adapter_root)): file_hash(path)
        for path in sorted(adapter_root.rglob("*"))
        if path.is_file()
    }


def load_quantized_base(model_root: Path, config: dict[str, Any]) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if config["acceptance"]["require_4bit_loaded"] and not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("base model was not loaded in 4-bit")
    return model


def tokenizer_from_local(model_root: Path, config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def command_validate(args: argparse.Namespace) -> int:
    config, manifest, dataset_path = verify_inputs(
        args.config, args.freeze, args.dataset_manifest, args.model_root
    )
    tokenizer = tokenizer_from_local(args.model_root, config)
    rows = load_rows(dataset_path, int(manifest["row_count"]))
    encoded = [
        encode_segmented_example(tokenizer, row, int(config["training"]["max_sequence_length"]))
        for row in rows
    ]
    result = {
        "passed": True,
        "rows": len(encoded),
        "trainable_tokens": sum(item["trainable_tokens"] for item in encoded),
        "masked_tokens": sum(item["masked_tokens"] for item in encoded),
        "loss_on": sorted(LOSS_ON),
        "loss_off": sorted(LOSS_OFF),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config, manifest, dataset_path = verify_inputs(
        args.config, args.freeze, args.dataset_manifest, args.model_root
    )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {args.output}")
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import get_cosine_schedule_with_warmup

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one visible CUDA GPU")
    seed = int(config["training"]["adapter_seeds"][0])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = tokenizer_from_local(args.model_root, config)
    rows = load_rows(dataset_path, int(manifest["row_count"]))
    encoded = [
        encode_segmented_example(tokenizer, row, int(config["training"]["max_sequence_length"]))
        for row in rows
    ]
    order = list(range(len(encoded)))
    random.Random(seed).shuffle(order)
    model = load_quantized_base(args.model_root, config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
    )
    model = get_peft_model(model, LoraConfig(
        r=int(config["training"]["lora_rank"]),
        lora_alpha=int(config["training"]["lora_alpha"]),
        lora_dropout=float(config["training"]["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["training"]["target_modules"]),
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count <= 0:
        raise RuntimeError("QLoRA has no trainable parameters")
    linear4bit_count = sum(1 for module in model.modules() if isinstance(module, bnb.nn.Linear4bit))
    if linear4bit_count <= 0:
        raise RuntimeError("no bitsandbytes Linear4bit modules found")
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    steps = int(config["training"]["max_optimizer_updates"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["training"]["warmup_steps"]),
        num_training_steps=steps,
    )
    losses: list[float] = []
    grad_norms: list[float] = []
    token_counts: list[int] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        item = encoded[order[step % len(order)]]
        batch = {
            key: torch.tensor([item[key]], dtype=torch.long, device="cuda:0")
            for key in ("input_ids", "attention_mask", "labels")
        }
        output = model(**batch)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}")
        loss.backward()
        squared = 0.0
        for parameter in trainable:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"non-finite gradient at step {step + 1}")
            squared += float(torch.sum(grad * grad).item())
        grad_norm = math.sqrt(squared)
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError(f"zero or non-finite gradient norm at step {step + 1}")
        torch.nn.utils.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(grad_norm)
        token_counts.append(int(item["trainable_tokens"]))
        if step == 0 or (step + 1) % 10 == 0:
            print(json.dumps({"step": step + 1, "loss": losses[-1], "grad_norm": grad_norm}, sort_keys=True), flush=True)
    if len(losses) != steps:
        raise RuntimeError("optimizer update count mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    adapter_root = args.output / "adapter"
    model.save_pretrained(adapter_root, safe_serialization=True)
    tokenizer.save_pretrained(adapter_root)
    hashes = adapter_hashes(adapter_root)
    if not hashes or not any(name.endswith(".safetensors") for name in hashes):
        raise RuntimeError("adapter safetensors file missing")
    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
        "bitsandbytes": bnb.__version__,
    }
    receipt = {
        "schema_version": 1,
        "study_kind": "qlora_pipeline_smoke",
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
        "freeze_receipt_digest": verify_sealed_json(args.freeze, "receipt_digest")["receipt_digest"],
        "adapter_seed": seed,
        "optimizer_updates": len(losses),
        "four_bit_loaded": True,
        "linear4bit_module_count": linear4bit_count,
        "trainable_parameter_count": trainable_count,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "gradient_norm_min": min(grad_norms),
        "gradient_norm_max": max(grad_norms),
        "trainable_tokens_total": sum(token_counts),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "adapter_file_hashes": hashes,
        "versions": versions,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = hashlib.sha256(canonical_json(receipt).encode("utf-8")).hexdigest()
    (args.output / "training_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"training_passed": True, "receipt_digest": receipt["receipt_digest"]}, sort_keys=True))
    return 0


def command_reload(args: argparse.Namespace) -> int:
    config, _, _ = verify_inputs(
        args.config, args.freeze, args.dataset_manifest, args.model_root
    )
    import torch
    from peft import PeftModel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("reload requires exactly one visible CUDA GPU")
    training_receipt = verify_sealed_json(args.output / "training_receipt.json", "receipt_digest")
    adapter_root = args.output / "adapter"
    actual_hashes = adapter_hashes(adapter_root)
    if actual_hashes != training_receipt["adapter_file_hashes"]:
        raise ValueError("adapter files changed after training receipt")
    tokenizer = tokenizer_from_local(args.model_root, config)
    model = load_quantized_base(args.model_root, config)
    model = PeftModel.from_pretrained(model, adapter_root, is_trainable=False, local_files_only=True)
    model.eval()
    prompt = render_segment("system", "Operate only on synthetic arithmetic fixtures.") + render_segment(
        "user", "Compute 17 * 9 + 4 and report a structured verification."
    ) + "<|im_start|>assistant\n"
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=bool(config["decoding"]["do_sample"]),
            num_beams=int(config["decoding"]["num_beams"]),
            max_new_tokens=int(config["decoding"]["max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_ids = generated[0, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
    if not new_ids:
        raise RuntimeError("offline reload inference produced no tokens")
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    inference = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "generated_token_ids": new_ids,
        "generated_text": text,
        "network_policy": "none",
    }
    (args.output / "offline_inference.json").write_text(json.dumps(inference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "study_kind": "qlora_pipeline_smoke",
        "scientific_claims_allowed": False,
        "eligible_for_champion": False,
        "eligible_for_training_library": False,
        "fresh_process_reload": True,
        "adapter_file_hashes_verified": True,
        "offline_inference_passed": True,
        "training_receipt_digest": training_receipt["receipt_digest"],
        "inference_sha256": file_hash(args.output / "offline_inference.json"),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = hashlib.sha256(canonical_json(receipt).encode("utf-8")).hexdigest()
    (args.output / "reload_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reload_passed": True, "receipt_digest": receipt["receipt_digest"]}, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-mask", "train", "reload"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command in {"train", "reload"} and args.output is None:
        parser.error("--output is required for train/reload")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "validate-mask":
        return command_validate(args)
    if args.command == "train":
        return command_train(args)
    return command_reload(args)


if __name__ == "__main__":
    raise SystemExit(main())
