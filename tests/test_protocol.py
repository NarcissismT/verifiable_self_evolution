from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import json

from vse.candidate_pool import PoolMember, validate_candidate_pool
from vse.contamination import ProbeObservation, evaluate_contamination
from vse.contracts import CandidateSource, Split, Task, TeacherPurpose
from vse.model_provenance import ModelProvenance
from vse.paper_capsule import (
    ArtifactKind,
    CapsuleResearchContext,
    PublicPaperCapsule,
    SealedTarget,
    TemporalArtifact,
    audit_capsule,
)
from vse.promotion import (
    EvaluationCell,
    FinalPolicy,
    PromotionPolicy,
    decide_final,
    decide_promotion,
    final_policy_from_config,
    promotion_policy_from_config,
)
from vse.paper_selection import PaperCandidate, eligible_candidates, select_frozen_papers
from vse.proposal_io import parse_model_proposal
from vse.registry import build_manifest
from vse.runner import CodeRunner, RunnerConfig
from vse.teacher_policy import TeacherPolicy, TeacherRequest
from vse.toy import attach_trusted_metrics, make_tasks, proposal, verifier


class ProtocolTests(unittest.TestCase):
    def test_freeze_config_builds_matching_policies(self) -> None:
        config = json.loads(
            (Path(__file__).parents[1] / "configs" / "paper_rediscovery_v0_1.json").read_text()
        )
        promotion = promotion_policy_from_config(config)
        final = final_policy_from_config(config)
        self.assertEqual(promotion.minimum_tasks, 12)
        self.assertEqual((promotion.promotion_id_tasks, promotion.promotion_ood_tasks), (8, 4))
        self.assertEqual((final.minimum_id_tasks, final.minimum_ood_tasks), (24, 16))
        self.assertEqual(final.minimum_id_vds_delta, 0.05)

    def test_capsule_target_identifier_leak_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            evidence = root / "prior.txt"
            evidence.write_text("pre-cutoff evidence")
            environment_lock = root / "environment.lock"
            environment_lock.write_text("python=3.11")
            sealed_root = root / "private"
            sealed_root.mkdir()
            sealed_snapshot = sealed_root / "target.pdf"
            sealed_snapshot.write_text("hidden target")
            target = SealedTarget(
                capsule_id="cap-1",
                target_id="target-1",
                title="Hidden Target Paper",
                identifiers=("arxiv2608.02163",),
                first_public_at_utc="2024-02-01T00:00:00Z",
                target_snapshot_path="target.pdf",
                target_snapshot_sha256=hashlib.sha256(
                    sealed_snapshot.read_bytes()
                ).hexdigest(),
                hidden_claims=(),
                hidden_result_spec={},
                salt="secret",
            )
            artifact = TemporalArtifact(
                artifact_id="a1",
                kind=ArtifactKind.PAPER,
                title="Prior evidence",
                version="v1",
                available_at_utc="2024-01-01T00:00:00Z",
                snapshot_path="prior.txt",
                snapshot_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                search_text_path="prior.txt",
                search_text_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
                provenance_url="https://example.invalid/prior",
            )
            capsule = PublicPaperCapsule(
                capsule_id="cap-1",
                split=Split.TRAIN,
                field="testing",
                cutoff_utc="2024-01-15T00:00:00Z",
                public_problem="Study the hidden target arxiv2608.02163.",
                research_context=CapsuleResearchContext(
                    research_question="A question",
                    formal_problem="A formal problem",
                    assumptions=("An assumption",),
                    known_results=(),
                    known_failures_and_conflicts=(),
                    candidate_metrics=("metric",),
                    available_datasets=(),
                    available_environments=(),
                    baseline_code=({"baseline_id": "b", "artifact_id": "a1"},),
                    compute_budget={"cpu_hours": 1},
                    source_manifest=(
                        {
                            "artifact_id": "a1",
                            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                            "git_commit": None,
                        },
                    ),
                    claim_evidence_graph=(
                        {"claim": "Pre-cutoff fact", "source_id": "a1", "locator": "p.1"},
                    ),
                ),
                artifacts=(artifact,),
                environment_lock_path="environment.lock",
                environment_lock_sha256=hashlib.sha256(
                    environment_lock.read_bytes()
                ).hexdigest(),
                allowed_tools=("python",),
                target_commitment=target.commitment,
            )
            audit = audit_capsule(capsule, target, root, sealed_root)
            self.assertFalse(audit.passed)
            self.assertIn("target_identifier_leak:arxiv2608.02163", audit.failures)

    def test_model_cutoff_must_be_before_capsule(self) -> None:
        model = ModelProvenance("m", "c", "2024-01-10T00:00:00Z", "a", True)
        self.assertTrue(
            model.permits_capsule(
                "2024-01-15T00:00:00Z", "2024-02-01T00:00:00Z"
            )
        )
        late = ModelProvenance("m", "c", "2024-01-20T00:00:00Z", "a", True)
        self.assertFalse(
            late.permits_capsule(
                "2024-01-15T00:00:00Z", "2024-02-01T00:00:00Z"
            )
        )
        unknown = ModelProvenance("m", "c", None, "a", True)
        self.assertFalse(
            unknown.permits_capsule(
                "2024-01-15T00:00:00Z", "2024-02-01T00:00:00Z"
            )
        )

    def test_teacher_permissions_and_candidate_pool(self) -> None:
        policy = TeacherPolicy()
        policy.authorize(
            TeacherRequest(
                "r", "t", Split.TRAIN, TeacherPurpose.BOOTSTRAP, 0, 0
            )
        )
        with self.assertRaises(PermissionError):
            policy.authorize(
                TeacherRequest(
                    "r", "t", Split.HELDOUT, TeacherPurpose.BOOTSTRAP, 0, 0
                )
            )
        with self.assertRaises(PermissionError):
            policy.authorize(
                TeacherRequest(
                    "r", "t", Split.TRAIN, TeacherPurpose.HARD_STATE_REPAIR, 1, 1
                )
            )
        members = (
            PoolMember(CandidateSource.CHAMPION, "champ"),
            PoolMember(CandidateSource.LEGACY, "legacy"),
            PoolMember(CandidateSource.VARIANT, "v1"),
            PoolMember(CandidateSource.VARIANT, "v2"),
            PoolMember(CandidateSource.CLOSED_TEACHER, "teacher"),
        )
        validate_candidate_pool(members, "t", Split.TRAIN, 0, 0, policy)

    def test_manifest_is_content_frozen(self) -> None:
        tasks = make_tasks({"train": 2, "dev": 1, "promotion": 1, "heldout": 1, "ood": 1})
        manifest = build_manifest(
            "e",
            tasks,
            {"x": 1, "split_constraints": {"train": 2, "dev": 1, "promotion": 1, "heldout": 1, "ood": 1}},
        )
        self.assertEqual(manifest.sealed().manifest_digest, manifest.manifest_digest)
        changed = Task(**{**tasks[0].__dict__, "statement": "changed"})
        with self.assertRaises(ValueError):
            manifest.verify_task(changed)

    def test_toy_verifier_accepts_exact_and_rejects_counterexample(self) -> None:
        task = make_tasks({"train": 1, "dev": 1, "promotion": 1, "heldout": 1, "ood": 1})[0]
        runner = CodeRunner(RunnerConfig(mode="local_test", timeout_seconds=5.0))
        exact = proposal(
            task,
            float(task.instance["optimum"]),
            CandidateSource.CHAMPION,
            "exact",
        )
        bad = proposal(task, 0.0, CandidateSource.VARIANT, "bad")
        exact_report = verifier().verify(
            task,
            tuple(
                attach_trusted_metrics(task, runner.execute(task, exact, seed))
                for seed in exact.seeds
            ),
        )
        bad_report = verifier().verify(
            task,
            tuple(
                attach_trusted_metrics(task, runner.execute(task, bad, seed))
                for seed in bad.seeds
            ),
        )
        self.assertTrue(exact_report.accepted, exact_report.hard_failures)
        self.assertFalse(bad_report.accepted)
        self.assertEqual(bad_report.vds_score, 0.0)
        self.assertTrue(
            any("hard_metric_failed" in item for item in bad_report.hard_failures)
        )

    def test_promotion_uses_only_distinct_promotion_split(self) -> None:
        policy = PromotionPolicy(
            minimum_tasks=1,
            promotion_id_tasks=1,
            promotion_ood_tasks=1,
            minimum_vds_delta=0.05,
            alpha=0.10,
            bootstrap_replicates=100,
            minimum_hard_pass_rate=0.95,
            maximum_component_drop=0.02,
            maximum_cost_ratio=1.10,
            minimum_adapter_seed_passes=2,
        )
        rows: list[tuple[EvaluationCell, EvaluationCell]] = []
        for task_index in range(2):
            stratum = "ood" if task_index == 1 else "id"
            for adapter_seed in (17, 29, 43):
                for seed in (1, 2):
                    common = {
                        "evaluation_run_id": "run-1",
                        "evaluation_phase": "promotion",
                        "promotion_attempt": 1,
                        "adapter_seed": adapter_seed,
                        "task_id": f"promotion-{task_index}",
                        "task_digest": f"task-{task_index}",
                        "split": Split.PROMOTION,
                        "stratum": stratum,
                        "seed": seed,
                        "hard_pass": True,
                        "executable_pass": True,
                        "fabricated_result": False,
                        "target_leakage": False,
                        "runtime_seconds": 1.0,
                        "cost_units": 1.0,
                        "manifest_digest": "manifest",
                        "evaluator_digest": "evaluator",
                        "contamination_audit_digest": "audit",
                        "vds_components": {
                            "empirical": 1.0,
                            "hypothesis": 1.0,
                            "experiment": 1.0,
                            "novelty": 1.0,
                            "calibration": 1.0,
                        },
                    }
                    rows.append(
                        (
                            EvaluationCell(
                                checkpoint_id="candidate",
                                checkpoint_digest="cand",
                                vds_score=1.0,
                                **common,
                            ),
                            EvaluationCell(
                                checkpoint_id="champion",
                                checkpoint_digest="champ",
                                vds_score=0.5,
                                **common,
                            ),
                        )
                    )
        candidate = [pair[0] for pair in rows]
        champion = [pair[1] for pair in rows]
        decision = decide_promotion(candidate, champion, policy, 42)
        self.assertTrue(decision.promoted, decision.failures)
        train_cell = EvaluationCell(
            **{**candidate[0].__dict__, "split": Split.TRAIN}
        )
        with self.assertRaises(ValueError):
            decide_promotion(candidate + [train_cell], champion, policy, 42)

        final_cell = EvaluationCell(
            **{
                **candidate[0].__dict__,
                "evaluation_phase": "final",
                "split": Split.HELDOUT,
                "task_id": "final-id-0",
            }
        )
        with self.assertRaises(ValueError):
            decide_promotion(candidate + [final_cell], champion, policy, 42)

    def test_final_gate_uses_final_splits_and_id_effect_threshold(self) -> None:
        components = {
            "empirical": 1.0,
            "hypothesis": 1.0,
            "experiment": 1.0,
            "novelty": 1.0,
            "calibration": 1.0,
        }
        candidate: list[EvaluationCell] = []
        champion: list[EvaluationCell] = []
        for split, count in ((Split.HELDOUT, 24), (Split.OOD, 16)):
            for index in range(count):
                common = {
                    "evaluation_run_id": "final-run",
                    "evaluation_phase": "final",
                    "promotion_attempt": 1,
                    "adapter_seed": 17,
                    "task_id": f"{split.value}-{index}",
                    "task_digest": f"digest-{split.value}-{index}",
                    "split": split,
                    "stratum": "id" if split is Split.HELDOUT else "ood",
                    "seed": 101,
                    "hard_pass": True,
                    "executable_pass": True,
                    "fabricated_result": False,
                    "target_leakage": False,
                    "vds_components": components,
                    "cost_units": 1.0,
                    "runtime_seconds": 1.0,
                    "manifest_digest": "manifest",
                    "evaluator_digest": "evaluator",
                    "contamination_audit_digest": "audit",
                }
                candidate.append(
                    EvaluationCell(
                        checkpoint_id="candidate",
                        checkpoint_digest="candidate-digest",
                        vds_score=1.0 if split is Split.HELDOUT else 0.8,
                        **common,
                    )
                )
                champion.append(
                    EvaluationCell(
                        checkpoint_id="champion",
                        checkpoint_digest="champion-digest",
                        vds_score=0.5 if split is Split.HELDOUT else 0.8,
                        **common,
                    )
                )
        decision = decide_final(
            candidate,
            champion,
            FinalPolicy(bootstrap_replicates=100),
            42,
        )
        self.assertTrue(decision.promoted, decision.failures)

    def test_contamination_audit_requires_balanced_probe_grid(self) -> None:
        rows: list[ProbeObservation] = []
        for model_id in ("7b", "14b"):
            for family_index, family in enumerate(
                ("bibliographic", "method", "result", "algorithm_name")
            ):
                for probe_index in range(3):
                    probe_id = f"{family_index}-{probe_index}"
                    for seed in (1, 2, 3):
                        rows.append(
                            ProbeObservation(
                                target_id="paper",
                                model_id=model_id,
                                model_digest=f"digest-{model_id}",
                                probe_id=probe_id,
                                probe_family=family,
                                seed=seed,
                                output_digest=f"{model_id}-{probe_id}-{seed}",
                                remembered_title_or_algorithm=False,
                                target_specific_contributions=0,
                                max_exact_phrase_words=0,
                                target_specific_numeric_results=0,
                            )
                        )
        decision = evaluate_contamination(
            tuple(rows), expected_models=frozenset({"7b", "14b"})
        )
        self.assertTrue(decision.passed, decision.failures)
        contaminated = replace(rows[0], max_exact_phrase_words=13)
        failed = evaluate_contamination(
            tuple([contaminated, *rows[1:]]), expected_models=frozenset({"7b", "14b"})
        )
        self.assertFalse(failed.passed)
        self.assertEqual(failed.excluded_models, ("7b",))

    def test_frozen_paper_selection_is_disjoint_and_quota_exact(self) -> None:
        strata = (
            "constrained_safe_rl",
            "general_sum_equilibrium_marl",
            "bilevel_stackelberg_alignment_optimization",
            "ood_constrained_bilevel_differentiable_optimization",
            "ood_safe_control_learning_to_optimize",
        )
        candidates = tuple(
            PaperCandidate(
                paper_id=f"paper-{stratum}-{index}",
                stratum=stratum,
                venue="ICLR",
                proceedings_year=2025,
                official_main_or_proceedings=True,
                public_paper=True,
                public_code=True,
                public_data=True,
                requires_commercial_api=False,
                requires_private_data=False,
                requires_real_robot=False,
                estimated_gpu_hours=1.0,
                estimated_cpu_hours=1.0,
                public_timestamps_utc={"arxiv_v1": "2025-06-01T00:00:00Z"},
            )
            for stratum in strata
            for index in range(14)
        )
        eligible = eligible_candidates(
            candidates,
            publication_start_utc="2025-03-01T00:00:00Z",
            publication_end_utc="2026-03-31T23:59:59Z",
            max_gpu_hours=4.0,
            max_cpu_hours=32.0,
        )
        selection = select_frozen_papers(
            eligible,
            {
                "promotion": {
                    "constrained_safe_rl": 3,
                    "general_sum_equilibrium_marl": 3,
                    "bilevel_stackelberg_alignment_optimization": 2,
                    "ood_constrained_bilevel_differentiable_optimization": 2,
                    "ood_safe_control_learning_to_optimize": 2,
                },
                "heldout": {
                    "constrained_safe_rl": 8,
                    "general_sum_equilibrium_marl": 8,
                    "bilevel_stackelberg_alignment_optimization": 8,
                },
                "ood": {
                    "ood_constrained_bilevel_differentiable_optimization": 8,
                    "ood_safe_control_learning_to_optimize": 8,
                },
            },
            sampling_seed=7,
        )
        all_ids = [paper_id for split in selection.assignments.values() for paper_id in split]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(selection.assignments["promotion"]), 12)
        self.assertEqual(len(selection.assignments["heldout"]), 24)
        self.assertEqual(len(selection.assignments["ood"]), 16)

    def test_model_cannot_inject_proposal_identity_or_extra_fields(self) -> None:
        task = make_tasks({"train": 1, "dev": 1, "promotion": 1, "heldout": 1, "ood": 1})[0]
        known = proposal(task, float(task.instance["optimum"]), CandidateSource.CHAMPION, "known")
        payload = known.payload()
        model_value = {
            key: payload[key]
            for key in (
                "hypothesis",
                "solution",
                "experiment_code",
                "seeds",
                "baselines",
                "primary_metric",
                "secondary_metrics",
                "expected_effect",
                "power_assumptions",
                "stopping_rule",
                "resource_schedule",
            )
        }
        parsed = parse_model_proposal(
            json.dumps(model_value),
            task=task,
            source=CandidateSource.STUDENT,
            model_id="student",
            model_digest="digest",
            round_index=1,
            frozen_seeds=known.seeds,
            mandatory_baselines=("zero",),
            allowed_baselines=frozenset({"zero"}),
            allowed_metrics=frozenset(
                {"objective_gap", "solution_error", "kkt_residual"}
            ),
            frozen_resource_schedule=known.resource_schedule,
        )
        self.assertEqual(parsed.source, CandidateSource.STUDENT)
        model_value["source"] = "closed_teacher"
        with self.assertRaises(ValueError):
            parse_model_proposal(
                json.dumps(model_value),
                task=task,
                source=CandidateSource.STUDENT,
                model_id="student",
                model_digest="digest",
                round_index=1,
                frozen_seeds=known.seeds,
                mandatory_baselines=("zero",),
                allowed_baselines=frozenset({"zero"}),
                allowed_metrics=frozenset(
                    {"objective_gap", "solution_error", "kkt_residual"}
                ),
                frozen_resource_schedule=known.resource_schedule,
            )


if __name__ == "__main__":
    unittest.main()
