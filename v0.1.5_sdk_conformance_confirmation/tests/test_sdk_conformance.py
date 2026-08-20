from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
REPO = ROOT.parent
MANIFEST = json.loads((ROOT / "cases" / "case_manifest.json").read_text())
sys.path.insert(0, str(ROOT / "scripts"))
from sdk_api_linter import lint_code  # noqa: E402
from run_sdk_gate import BASELINES, EVALUATOR_VERSION, METRICS, SCHEDULE, SEEDS, frozen_view, proposal_binding, task_for  # noqa: E402
from vse.contracts import CandidateSource  # noqa: E402
from vse.hashing import canonical_json  # noqa: E402
from vse.proposal_io import parse_model_proposal  # noqa: E402


class SDKConformanceTests(unittest.TestCase):
    def test_card_has_frozen_signatures_and_binding_inputs(self) -> None:
        card = json.loads((ROOT / "sdk" / "sdk_card.json").read_text())
        self.assertEqual(card, {
            "initialize_seed": "initialize_seed(seed: int) -> None",
            "finite_vector": "finite_vector(values, dimension: int) -> list[float]",
            "project_box": "project_box(values, bounds) -> list[float]",
            "Budget": "Budget(limit: int); consume(amount=1); remaining",
        })
        lock = json.loads((ROOT / "sdk" / "sdk_lock.json").read_text())
        self.assertEqual(lock["candidate_public_imports"], ["Budget", "project_box"])
        self.assertEqual(lock["numpy_version"], "2.1.2")
        self.assertFalse(lock["scipy_allowed"])

    def test_wrapper_owns_seed_and_rejects_candidate_metrics(self) -> None:
        candidate = """import random
from execution_sdk import project_box
def solve(problem, seed, budget):
    point = project_box([random.uniform(-2, 2) for _ in range(int(problem['dimension']))], problem['bounds'])
    return {'point': point, 'oracle_calls': 0}
"""
        extra = """def solve(problem, seed, budget):
    return {'point': [0.0] * int(problem['dimension']), 'oracle_calls': 0, 'regret': 0.0}
"""
        problem = MANIFEST["cases"][0]["problem"]
        for code, expected in ((candidate, True), (extra, False)):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "candidate.py"
                path.write_text(code)
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "sdk" / "trusted_wrapper.py"), str(path), str(ROOT / "sdk")],
                    input=json.dumps({"problem": problem, "seed": 1103, "budget": 1000}),
                    text=True, capture_output=True, check=False)
                payload = json.loads(completed.stdout.strip())
                self.assertEqual(payload["ok"], expected)

    def test_linter_catches_the_four_v014_failures(self) -> None:
        samples = (
            "def solve(problem, seed, budget):\n return {'point':[random.random()], 'oracle_calls':0}\n",
            "from execution_sdk import finite_vector\ndef solve(problem, seed, budget):\n return {'point':finite_vector(problem['target']), 'oracle_calls':0}\n",
            "from execution_sdk import initialize_seed\ndef solve(problem, seed, budget):\n initialize_seed(seed, 1, 2, 3)\n return {'point':[0.0], 'oracle_calls':0}\n",
            "def solve(problem, seed, budget):\n return {'point':np.zeros(problem['dimension']).tolist(), 'oracle_calls':0}\n",
        )
        issues = [{item.code for item in lint_code(code)} for code in samples]
        self.assertIn("undefined_name", issues[0])
        self.assertIn("sdk_arity", issues[1])
        self.assertIn("non_public_sdk_api", issues[2])
        self.assertIn("numpy_alias_missing", issues[3])

    def test_new_cases_are_permanently_excluded_and_not_v014(self) -> None:
        ids = [case["case_id"] for case in MANIFEST["cases"]]
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)
        self.assertTrue(all(item.startswith("sdk_confirm_") for item in ids))
        self.assertFalse(MANIFEST["formal_split_eligible"])
        old = json.loads((REPO / "v0.1.4_executable_proposal_bridge" / "cases" / "case_manifest.json").read_text())
        self.assertTrue(set(ids).isdisjoint(item["case_id"] for item in old["cases"]))
        self.assertIn("target + offset", MANIFEST["public_math_contract"])

    def test_generator_has_no_code_placeholders_and_only_hydrates_frozen_fields(self) -> None:
        source = (ROOT / "scripts" / "generate_sdk_proposals.py").read_text()
        self.assertNotIn('"normal multi-line Python source encoded as a JSON string"', source)
        self.assertNotIn('"replacement multi-line Python source"', source)
        self.assertIn("hydrate_frozen_protocol_fields", source)
        self.assertIn("raw_schema_complete_rate", source)
        self.assertNotIn('FROZEN_PROTOCOL_FIELDS["hypothesis"]', source)
        self.assertNotIn('FROZEN_PROTOCOL_FIELDS["solution"]', source)
        self.assertNotIn('FROZEN_PROTOCOL_FIELDS["experiment_code"]', source)

    def test_archived_set_c_pass_is_execution_only_and_hash_bound(self) -> None:
        root = REPO / "artifacts" / "sdk_conformance_qwen7b_set_c"
        report = json.loads((root / "sdk_conformance_report.json").read_text())
        receipt = json.loads((root / "sdk_conformance_receipt.json").read_text())
        generation = json.loads((root / "generation_report.json").read_text())
        self.assertTrue(report["sdk_contract_gate_passed"])
        self.assertEqual(report["parser_valid_rate"], 1.0)
        self.assertEqual(report["raw_schema_complete_rate"], 0.0)
        self.assertEqual(report["valid_solver_result_cases"], 4)
        self.assertEqual(report["scientific_hard_pass_cases"], 0)
        self.assertTrue(report["negative_controls"]["passed"])
        self.assertEqual(generation["model_calls_during_replay"], 0)
        self.assertEqual(receipt["report_digest"], report["report_digest"])
        self.assertFalse(receipt["scientific_claims_allowed"])
        self.assertFalse(receipt["eligible_for_training_library"])
        for artifact in (REPO / "artifacts" / "sdk_conformance_qwen7b",
                         REPO / "artifacts" / "sdk_conformance_qwen7b_set_b", root):
            manifest = json.loads((artifact / "receipt_manifest.json").read_text())
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((artifact / name).read_bytes()).hexdigest(), digest)

    def test_scripts_compile_and_gate_synthetic_confirmations(self) -> None:
        subprocess.run([sys.executable, "-m", "py_compile", *map(str, (ROOT / "scripts").glob("*.py"))], check=True)
        model_digest = "a" * 64
        candidate_code = """from execution_sdk import project_box
def solve(problem, seed, budget):
    del seed, budget
    offset = problem.get('offset', [0.0] * int(problem['dimension']))
    center = [float(a) + float(b) for a, b in zip(problem['target'], offset)]
    return {'point': project_box(center, problem['bounds']), 'oracle_calls': 0}
"""
        hypothesis = {"claim": "The public coordinatewise projection reaches the constrained minimizer.",
                      "mechanism": "Projection clips each coordinate to its public interval.",
                      "assumptions": ["The objective is separable and convex."],
                      "alternative_explanations": ["A zero baseline may be sufficient."],
                      "null_hypothesis": "Projection does not improve over zero.",
                      "predicted_failure_mode": "A wrong shift sign causes residual error.",
                      "discriminating_observation": "Residual is zero when the public center is feasible."}
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            proposals = temp_root / "proposals"
            run = temp_root / "run"
            for case in MANIFEST["cases"]:
                task = task_for(case)
                value = {"hypothesis": hypothesis, "solution": {"api": "solve_v1", "algorithm_family": "projection"},
                         "experiment_code": candidate_code, "seeds": list(SEEDS), "baselines": list(BASELINES),
                         "primary_metric": "regret", "secondary_metrics": ["lower_residual", "kkt_ni_gap", "oracle_calls", "seed_reproducibility"],
                         "expected_effect": {"direction": "lower", "minimum_delta": 0.0},
                         "power_assumptions": {"alpha": 0.05, "target_power": 0.8, "unit": "seed"},
                         "stopping_rule": "return within the supplied budget", "resource_schedule": list(SCHEDULE)}
                proposal = parse_model_proposal(json.dumps(value), task=task, source=CandidateSource.STUDENT,
                                                model_id="synthetic", model_digest=model_digest, round_index=0,
                                                frozen_seeds=SEEDS, mandatory_baselines=BASELINES,
                                                allowed_baselines=frozenset(BASELINES), allowed_metrics=METRICS,
                                                frozen_resource_schedule=SCHEDULE)
                root = proposals / case["case_id"]
                root.mkdir(parents=True)
                (root / "proposal.json").write_text(json.dumps(proposal.payload(), indent=2, sort_keys=True) + "\n")
                (root / "proposal_binding.json").write_text(json.dumps(proposal_binding(proposal), indent=2, sort_keys=True) + "\n")
                chain = {"schema_version": 1, "initial_digest": proposal.digest, "final_digest": proposal.digest,
                         "repairs": [], "frozen_fields_digest": hashlib.sha256(canonical_json(frozen_view(proposal.payload())).encode()).hexdigest()}
                (root / "proposal_chain.json").write_text(json.dumps(chain, indent=2, sort_keys=True) + "\n")
            generation = temp_root / "generation_report.json"
            generation.write_text(json.dumps({"initial_execution_rate": 1.0, "execution_rate_after_repair": 1.0}))
            binding = temp_root / "sdk_binding.json"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "bind_sdk.py"), "--sdk-root", str(ROOT / "sdk"),
                            "--execution-image-digest", "sha256:" + "b" * 64, "--output", str(binding)], check=True)
            completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_sdk_gate.py"),
                                        "--case-manifest", str(ROOT / "cases" / "case_manifest.json"),
                                        "--proposals-root", str(proposals), "--output-root", str(run),
                                        "--model-digest", model_digest, "--execution-container-digest", "sha256:" + "b" * 64,
                                        "--sdk-binding", str(binding), "--generation-report", str(generation)],
                                       cwd=REPO, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((run / "sdk_conformance_report.json").read_text())
            self.assertEqual(report["parser_valid_rate"], 1.0)
            self.assertEqual(report["valid_solver_result_cases"], 5)
            self.assertTrue(report["negative_controls"]["passed"])
            self.assertFalse(report["scientific_hard_pass_launch_gate"])


if __name__ == "__main__":
    unittest.main()
