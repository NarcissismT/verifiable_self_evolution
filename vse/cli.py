from __future__ import annotations

import argparse
import json
from pathlib import Path

from .hashing import content_hash
from .registry import build_manifest, write_manifest_once
from .store import export_sft
from .toy import make_tasks, run_training_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vse")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--root", type=Path, required=True)
    toy = commands.add_parser("run-toy")
    toy.add_argument("--root", type=Path, required=True)
    export = commands.add_parser("export-sft")
    export.add_argument("--library", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
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
    if args.command == "export-sft":
        print(export_sft(args.library, args.output))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
