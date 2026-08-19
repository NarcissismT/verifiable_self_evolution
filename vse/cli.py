from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .contamination import ProbeObservation, evaluate_contamination
from .formal import bind_formal_freeze, initialize_formal_run, seal_formal_capsules
from .evaluation_io import read_evaluation_cells, write_decision_once
from .freeze import check_freeze, write_report
from .hashing import content_hash
from .ledger import RunLedger
from .paper_capsule import (
    audit_capsule,
    capsule_from_mapping,
    sealed_target_from_mapping,
)
from .promotion import (
    decide_final,
    decide_promotion,
    final_policy_from_config,
    promotion_policy_from_config,
)
from .registry import build_manifest, write_manifest_once
from .state import PromotionStateMachine
from .store import export_sft, export_training_datasets
from .toy import make_tasks, run_training_smoke
from .vertical_slice import run_vertical_slice, write_vertical_slice_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vse")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--root", type=Path, required=True)
    toy = commands.add_parser("run-toy")
    toy.add_argument("--root", type=Path, required=True)
    formal = commands.add_parser("init-formal")
    formal.add_argument("--config", type=Path, required=True)
    formal.add_argument("--candidates", type=Path, required=True)
    formal.add_argument("--root", type=Path, required=True)
    seal = commands.add_parser("seal-capsules")
    seal.add_argument("--root", type=Path, required=True)
    seal.add_argument("--index", type=Path, required=True)
    seal.add_argument("--public-root", type=Path, required=True)
    seal.add_argument("--sealed-root", type=Path, required=True)
    bind = commands.add_parser("bind-freeze")
    bind.add_argument("--root", type=Path, required=True)
    bind.add_argument("--evaluator-receipt", type=Path, required=True)
    bind.add_argument("--contamination-receipt", type=Path, required=True)
    bind.add_argument("--base-checkpoint-receipt", type=Path, required=True)
    freeze = commands.add_parser("freeze-check")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--root", type=Path)
    freeze.add_argument("--output", type=Path)
    capsule = commands.add_parser("audit-capsule")
    capsule.add_argument("--capsule", type=Path, required=True)
    capsule.add_argument("--target", type=Path, required=True)
    capsule.add_argument("--capsule-root", type=Path, required=True)
    capsule.add_argument("--sealed-root", type=Path, required=True)
    capsule.add_argument("--output", type=Path, required=True)
    contamination = commands.add_parser("audit-contamination")
    contamination.add_argument("--observations", type=Path, required=True)
    contamination.add_argument("--expected-model", action="append", required=True)
    contamination.add_argument("--output", type=Path, required=True)
    ledger = commands.add_parser("ledger-check")
    ledger.add_argument("--ledger", type=Path, required=True)
    promote = commands.add_parser("promotion-decision")
    promote.add_argument("--root", type=Path, required=True)
    promote.add_argument("--candidate", type=Path, required=True)
    promote.add_argument("--champion", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("final-decision")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--candidate", type=Path, required=True)
    final.add_argument("--base", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    vertical = commands.add_parser("run-vertical-slice")
    vertical.add_argument("--manifest", type=Path, required=True)
    vertical.add_argument("--public-root", type=Path, required=True)
    vertical.add_argument("--sealed-root", type=Path, required=True)
    vertical.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export-sft")
    export.add_argument("--library", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export_all = commands.add_parser("export-training-data")
    export_all.add_argument("--success-library", type=Path, required=True)
    export_all.add_argument("--counterexample-library", type=Path, required=True)
    export_all.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        config = json.loads(args.config.read_text())
        tasks = make_tasks(config["splits"], seed=config["seeds"]["task_seed"])
        manifest = build_manifest(config["experiment_id"], tasks, config)
        write_manifest_once(args.root / "manifests" / "tasks.json", manifest)
        destination = args.root / "config.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sealed = dict(config)
        sealed["config_digest"] = content_hash(config)
        serialized = json.dumps(sealed, indent=2, sort_keys=True) + "\n"
        if destination.exists() and destination.read_text() != serialized:
            raise FileExistsError(f"refusing to replace frozen config: {destination}")
        destination.write_text(serialized)
        print(
            json.dumps(
                {"manifest_digest": manifest.manifest_digest, "task_count": len(tasks)}
            )
        )
        return 0
    if args.command == "run-toy":
        config = json.loads((args.root / "config.json").read_text())
        outcomes = run_training_smoke(args.root, config["splits"])
        print(json.dumps(outcomes, indent=2, sort_keys=True))
        return 0
    if args.command == "init-formal":
        config = json.loads(args.config.read_text())
        print(
            json.dumps(
                initialize_formal_run(config, args.candidates, args.root),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "seal-capsules":
        print(
            json.dumps(
                seal_formal_capsules(
                    args.root,
                    args.index,
                    public_root=args.public_root,
                    sealed_root=args.sealed_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "bind-freeze":
        print(
            json.dumps(
                bind_formal_freeze(
                    args.root,
                    evaluator_receipt=args.evaluator_receipt,
                    contamination_receipt=args.contamination_receipt,
                    base_checkpoint_receipt=args.base_checkpoint_receipt,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "freeze-check":
        config = json.loads(args.config.read_text())
        report = check_freeze(config, args.root)
        if args.output is not None:
            write_report(args.output, report)
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0 if report.ready else 2
    if args.command == "audit-capsule":
        capsule_value = capsule_from_mapping(json.loads(args.capsule.read_text()))
        target_value = sealed_target_from_mapping(json.loads(args.target.read_text()))
        report = audit_capsule(
            capsule_value,
            target_value,
            args.capsule_root,
            args.sealed_root,
        )
        serialized = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
        if args.output.exists() and args.output.read_text() != serialized:
            raise FileExistsError(f"refusing to replace capsule audit: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(serialized, end="")
        return 0 if report.passed else 2
    if args.command == "audit-contamination":
        observations: list[ProbeObservation] = []
        for line_number, line in enumerate(args.observations.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                observations.append(ProbeObservation(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid contamination row at line {line_number}") from error
        report = evaluate_contamination(
            tuple(observations), expected_models=frozenset(args.expected_model)
        )
        serialized = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
        if args.output.exists() and args.output.read_text() != serialized:
            raise FileExistsError(f"refusing to replace contamination audit: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(serialized, end="")
        return 0 if report.passed else 2
    if args.command == "ledger-check":
        entries = RunLedger(args.ledger).validate()
        print(
            json.dumps(
                {"entry_count": len(entries), "head": entries[-1].entry_hash if entries else ""}
            )
        )
        return 0
    if args.command in {"promotion-decision", "final-decision"}:
        config = json.loads((args.root / "config.json").read_text())
        bindings = json.loads(
            (args.root / "manifests" / "freeze_bindings.json").read_text()
        )
        required_bindings = {
            "config_digest",
            "paper_selection_digest",
            "task_manifest_digest",
            "evaluator_digest",
            "contamination_audit_digest",
            "base_checkpoint_digest",
        }
        missing = required_bindings - set(bindings)
        if missing or any(not bindings[key] for key in required_bindings):
            raise ValueError(f"freeze bindings are incomplete: {sorted(missing)}")
        if bindings["config_digest"] != config["config_digest"]:
            raise ValueError("run config digest does not match freeze bindings")
        ledger_value = RunLedger(args.root / "ledger" / "events.jsonl")
        state_machine = PromotionStateMachine(
            ledger_value,
            maximum_attempts=int(config["evolution"]["maximum_promotion_attempts"]),
            frozen_bindings={key: str(bindings[key]) for key in required_bindings},
            initial_champion_digest=str(bindings["base_checkpoint_digest"]),
        )
        if args.command == "promotion-decision":
            candidate = read_evaluation_cells(args.candidate)
            champion = read_evaluation_cells(args.champion)
            policy = promotion_policy_from_config(config)
            policy = type(policy)(
                **{
                    **policy.__dict__,
                    "expected_manifest_digest": bindings["task_manifest_digest"],
                    "expected_evaluator_digest": bindings["evaluator_digest"],
                    "expected_contamination_digest": bindings["contamination_audit_digest"],
                }
            )
            decision = decide_promotion(
                candidate,
                champion,
                policy,
                int(config["seeds"]["power_simulation"]),
            )
            attempt = next(iter({cell.promotion_attempt for cell in candidate}))
            state_machine.record_promotion(decision, promotion_attempt=attempt)
        else:
            candidate = read_evaluation_cells(args.candidate)
            base = read_evaluation_cells(args.base)
            policy = final_policy_from_config(config)
            policy = type(policy)(
                **{
                    **policy.__dict__,
                    "reference_checkpoint_id": "base",
                    "reference_checkpoint_digest": bindings["base_checkpoint_digest"],
                    "expected_manifest_digest": bindings["task_manifest_digest"],
                    "expected_evaluator_digest": bindings["evaluator_digest"],
                    "expected_contamination_digest": bindings["contamination_audit_digest"],
                }
            )
            decision = decide_final(
                candidate,
                base,
                policy,
                int(config["seeds"]["power_simulation"]),
            )
            state_machine.record_final(decision)
        write_decision_once(args.output, decision)
        print(json.dumps({"promoted": decision.promoted, "decision_digest": decision.decision_digest}))
        return 0 if decision.promoted else 3
    if args.command == "run-vertical-slice":
        report = run_vertical_slice(
            args.manifest,
            public_root=args.public_root,
            sealed_root=args.sealed_root,
        )
        write_vertical_slice_report(args.output, report)
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0 if report.passed else 2
    if args.command == "export-sft":
        print(export_sft(args.library, args.output))
        return 0
    if args.command == "export-training-data":
        print(
            json.dumps(
                export_training_datasets(
                    args.success_library,
                    args.counterexample_library,
                    args.output_dir,
                ),
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
