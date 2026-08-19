from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from vse.contracts import Split
from vse.formal import initialize_formal_run
from vse.freeze import check_freeze
from vse.hashing import content_hash, file_hash
from vse.ledger import RunLedger
from vse.promotion import PromotionDecision
from vse.state import PromotionStateMachine
from vse.paper_capsule import (
    ArtifactKind,
    CapsuleResearchContext,
    PublicPaperCapsule,
    SealedTarget,
    TemporalArtifact,
)
from vse.store import export_sft
from vse.toy import run_training_smoke
from vse.vertical_slice import run_vertical_slice


REPO_ROOT = Path(__file__).parents[1]


def _candidate(paper_id: str, stratum: str) -> dict:
    return {
        "paper_id": paper_id,
        "stratum": stratum,
        "venue": "ICLR",
        "proceedings_year": 2025,
        "official_main_or_proceedings": True,
        "public_paper": True,
        "public_code": True,
        "public_data": True,
        "requires_commercial_api": False,
        "requires_private_data": False,
        "requires_real_robot": False,
        "estimated_gpu_hours": 1.0,
        "estimated_cpu_hours": 1.0,
        "public_timestamps_utc": {"arxiv_v1": "2025-06-01T00:00:00Z"},
    }


def _sealed_receipt(payload: dict) -> dict:
    value = {**payload, "receipt_digest": ""}
    value["receipt_digest"] = content_hash(value)
    return value


class HardeningTests(unittest.TestCase):
    def test_freeze_check_blocks_current_partial_freeze(self) -> None:
        config = json.loads(
            (REPO_ROOT / "configs" / "paper_rediscovery_v0_1.json").read_text()
        )
        report = check_freeze(config)
        self.assertFalse(report.ready)
        self.assertIn(
            "freeze_status:blocked_pending_implementation_and_artifacts",
            report.failures,
        )
        self.assertIn("final_sample_sizes_pending_power_confirmation", report.failures)

    def test_formal_init_freezes_all_five_splits_and_reserve(self) -> None:
        config = json.loads(
            (REPO_ROOT / "configs" / "paper_rediscovery_v0_1.json").read_text()
        )
        strata = tuple(config["selection_quotas"]["train"]) + tuple(
            key
            for key in config["selection_quotas"]["ood"]
            if key not in config["selection_quotas"]["train"]
        )
        rows = [
            _candidate(f"paper-{stratum}-{index:03d}", stratum)
            for stratum in strata
            for index in range(60)
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            candidates = root / "candidates.jsonl"
            candidates.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            run_root = root / "run"
            result = initialize_formal_run(config, candidates, run_root)
            self.assertEqual(result["assigned_count"], 161)
            self.assertEqual(result["reserve_count"], len(rows) - 161)
            selection = json.loads(
                (run_root / "manifests" / "paper_selection.json").read_text()
            )
            self.assertEqual(set(selection["assignments"]), {
                "train", "dev", "promotion", "heldout", "ood"
            })
            assigned = [
                paper_id
                for values in selection["assignments"].values()
                for paper_id in values
            ]
            self.assertEqual(len(assigned), len(set(assigned)))

            cli_run_root = root / "cli-run"
            completed = subprocess.run(
                [
                    "python",
                    "-m",
                    "vse.cli",
                    "init-formal",
                    "--config",
                    str(REPO_ROOT / "configs" / "paper_rediscovery_v0_1.json"),
                    "--candidates",
                    str(candidates),
                    "--root",
                    str(cli_run_root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(
                json.loads(completed.stdout)["assigned_count"],
                161,
            )

    def test_ledger_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "events.jsonl"
            ledger = RunLedger(path)
            ledger.append("one", {"x": 1}, bindings={"config": "a"})
            ledger.append("two", {"x": 2}, bindings={"config": "a"})
            self.assertEqual(len(ledger.validate()), 2)
            rows = path.read_text().splitlines()
            value = json.loads(rows[0])
            value["payload"]["x"] = 99
            rows[0] = json.dumps(value, sort_keys=True)
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaises(ValueError):
                ledger.validate()

    def test_promotion_state_machine_is_contiguous_and_final_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = RunLedger(Path(raw) / "events.jsonl")
            machine = PromotionStateMachine(
                ledger,
                maximum_attempts=3,
                frozen_bindings={"config": "c", "manifest": "m"},
                initial_champion_digest="base",
            )
            decision = PromotionDecision(
                promoted=True,
                decision_kind="promotion",
                candidate_checkpoint_digest="candidate-1",
                champion_checkpoint_digest="base",
                manifest_digest="m",
                evaluator_digest="e",
                split_decisions=(),
                failures=(),
            ).sealed()
            state = machine.record_promotion(decision, promotion_attempt=1)
            self.assertEqual(state.champion_checkpoint_digest, "candidate-1")
            with self.assertRaises(ValueError):
                machine.record_promotion(decision, promotion_attempt=3)
            final = PromotionDecision(
                promoted=True,
                decision_kind="final",
                candidate_checkpoint_digest="candidate-1",
                champion_checkpoint_digest="base",
                manifest_digest="m",
                evaluator_digest="e",
                split_decisions=(),
                failures=(),
            ).sealed()
            machine.record_final(final)
            with self.assertRaises(ValueError):
                machine.record_final(final)

    def test_training_export_rechecks_report_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_training_smoke(
                root,
                {"train": 2, "dev": 1, "promotion": 1, "heldout": 1, "ood": 1},
            )
            library = root / "libraries" / "success.jsonl"
            export_sft(library, root / "valid.jsonl")
            rows = library.read_text().splitlines()
            value = json.loads(rows[0])
            value["verification"]["quality_score"] = 0.123
            tampered = root / "tampered.jsonl"
            tampered.write_text(json.dumps(value) + "\n")
            with self.assertRaises(ValueError):
                export_sft(tampered, root / "invalid.jsonl")

    def test_three_case_vertical_slice_and_negative_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            public = base / "public"
            sealed = base / "sealed"
            public.mkdir()
            sealed.mkdir()
            cases: list[dict] = []
            for index in range(3):
                case_id = f"pilot-{index}"
                public_case = public / case_id
                sealed_case = sealed / case_id
                public_case.mkdir()
                sealed_case.mkdir()
                evidence = public_case / "prior.txt"
                evidence.write_text(f"pre-cutoff evidence {index}")
                environment = public_case / "environment.lock"
                environment.write_text("python=3.11")
                snapshot = sealed_case / "target.txt"
                snapshot.write_text(f"hidden target {index}")
                target = SealedTarget(
                    capsule_id=case_id,
                    target_id=f"target-{index}",
                    title=f"Hidden title {index}",
                    identifiers=(f"hidden-id-{index}",),
                    first_public_at_utc="2025-06-01T00:00:00Z",
                    target_snapshot_path=f"{case_id}/target.txt",
                    target_snapshot_sha256=file_hash(snapshot),
                    hidden_claims=(),
                    hidden_result_spec={},
                    salt=f"salt-{index}",
                )
                artifact = TemporalArtifact(
                    artifact_id=f"prior-{index}",
                    kind=ArtifactKind.PAPER,
                    title=f"Prior paper {index}",
                    version="v1",
                    available_at_utc="2025-01-01T00:00:00Z",
                    snapshot_path=f"{case_id}/prior.txt",
                    snapshot_sha256=file_hash(evidence),
                    search_text_path=f"{case_id}/prior.txt",
                    search_text_sha256=file_hash(evidence),
                    provenance_url=f"https://example.invalid/prior-{index}",
                )
                capsule = PublicPaperCapsule(
                    capsule_id=case_id,
                    split=Split.DEV,
                    field="bilevel_stackelberg_alignment_optimization",
                    cutoff_utc="2025-05-01T00:00:00Z",
                    public_problem="Find a stationary constrained solution.",
                    research_context=CapsuleResearchContext(
                        research_question="Which update is stable?",
                        formal_problem="A constrained bilevel objective.",
                        assumptions=("Smooth objective",),
                        known_results=(),
                        known_failures_and_conflicts=(),
                        candidate_metrics=("kkt_residual",),
                        available_datasets=(),
                        available_environments=(),
                        baseline_code=({"baseline_id": "base", "artifact_id": f"prior-{index}"},),
                        compute_budget={"cpu_hours": 1},
                        claim_evidence_graph=(
                            {"claim": "Smoothness", "source_id": f"prior-{index}", "locator": "p.1"},
                        ),
                        source_manifest=(
                            {"artifact_id": f"prior-{index}", "sha256": file_hash(evidence)},
                        ),
                    ),
                    artifacts=(artifact,),
                    environment_lock_path=f"{case_id}/environment.lock",
                    environment_lock_sha256=file_hash(environment),
                    allowed_tools=("python",),
                    target_commitment=target.commitment,
                    semantic_leak_review_digest=f"review-{index}",
                )
                (public_case / "capsule.json").write_text(
                    json.dumps(capsule.payload(), sort_keys=True)
                )
                (sealed_case / "target.json").write_text(
                    json.dumps(asdict(target), sort_keys=True)
                )
                generation = _sealed_receipt({
                    "capsule_digest": capsule.digest,
                    "model_digest": "model",
                    "proposal_digest": f"proposal-{index}",
                })
                execution = _sealed_receipt({
                    "capsule_digest": capsule.digest,
                    "proposal_digest": f"proposal-{index}",
                    "network_policy": "none",
                    "container_digest": "container",
                    "execution_digest": f"execution-{index}",
                })
                evaluation = _sealed_receipt({
                    "capsule_digest": capsule.digest,
                    "execution_digest": f"execution-{index}",
                    "trusted_evaluator_digest": "evaluator",
                    "hard_pass": True,
                    "vds_score": 0.8,
                })
                for name, value in (
                    ("generation", generation),
                    ("execution", execution),
                    ("evaluation", evaluation),
                ):
                    (public_case / f"{name}.json").write_text(json.dumps(value))
                cases.append({
                    "case_id": case_id,
                    "independent_pilot": True,
                    "excluded_from_formal_splits": True,
                    "capsule_json": f"{case_id}/capsule.json",
                    "target_json": f"{case_id}/target.json",
                    "generation_receipt": f"{case_id}/generation.json",
                    "execution_receipt": f"{case_id}/execution.json",
                    "evaluation_receipt": f"{case_id}/evaluation.json",
                })
            manifest = base / "manifest.json"
            manifest.write_text(json.dumps({"pilot_id": "pilot", "cases": cases}))
            report = run_vertical_slice(manifest, public_root=public, sealed_root=sealed)
            self.assertTrue(report.passed, report.failures)

            bad_path = public / "pilot-0" / "evaluation.json"
            bad = json.loads(bad_path.read_text())
            bad["hard_pass"] = False
            bad["receipt_digest"] = ""
            bad["receipt_digest"] = content_hash(bad)
            bad_path.write_text(json.dumps(bad))
            failed = run_vertical_slice(manifest, public_root=public, sealed_root=sealed)
            self.assertFalse(failed.passed)
            self.assertTrue(
                any("hard_failure_did_not_zero_vds" in item for item in failed.failures)
            )


if __name__ == "__main__":
    unittest.main()
