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


class BridgeContractTests(unittest.TestCase):
    def test_sdk_is_target_neutral_and_locked(self) -> None:
        source = (ROOT / "sdk" / "execution_sdk.py").read_text()
        self.assertNotIn("confirm_", source)
        lock = json.loads((ROOT / "sdk" / "sdk_lock.json").read_text())
        self.assertFalse(lock["scipy_allowed"])
        self.assertEqual(lock["candidate_entrypoint"], "solve(problem, seed, budget)")

    def test_wrapper_accepts_valid_candidate_and_rejects_extra_metrics(self) -> None:
        candidate = """from execution_sdk import finite_vector\ndef solve(problem, seed, budget):\n    return {'point': finite_vector(problem['target'], problem['dimension']), 'oracle_calls': 1}\n"""
        extra = """def solve(problem, seed, budget):\n    return {'point': [0.0] * problem['dimension'], 'oracle_calls': 1, 'trusted_metrics': {}}\n"""
        for code, expected in ((candidate, True), (extra, False)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / "candidate.py"
                path.write_text(code)
                completed = subprocess.run(
                    [sys.executable, str(ROOT / "sdk" / "trusted_wrapper.py"), str(path), str(ROOT / "sdk")],
                    input=json.dumps({"problem": MANIFEST["cases"][0]["problem"], "seed": 1103, "budget": 1000}),
                    text=True, capture_output=True, check=False,
                )
                payload = json.loads(completed.stdout)
                self.assertEqual(payload["ok"], expected)

    def test_confirmation_cases_are_permanently_excluded(self) -> None:
        self.assertEqual(len(MANIFEST["cases"]), 5)
        self.assertFalse(MANIFEST["formal_split_eligible"])
        self.assertTrue(all(case["case_id"].startswith("confirm_") for case in MANIFEST["cases"]))

    def test_scripts_compile(self) -> None:
        subprocess.run([sys.executable, "-m", "py_compile", *map(str, (ROOT / "scripts").glob("*.py"))], check=True)

    def test_generator_repairs_only_public_runtime_errors(self) -> None:
        source = (ROOT / "scripts" / "generate_bridge_proposals.py").read_text()
        self.assertIn("public_runtime_check", source)
        self.assertIn("public_runtime_error", source)
        self.assertNotIn("bridge_metrics", source)
        self.assertNotIn("conditional_vds", source)

    def test_archived_qwen_result_is_fail_closed_and_bound(self) -> None:
        root = REPO / "artifacts" / "causal_bridge_qwen7b"
        report = json.loads((root / "bridge_report.json").read_text())
        receipt = json.loads((root / "bridge_receipt.json").read_text())
        self.assertEqual(report["parser_valid_rate"], 1.0)
        self.assertEqual(report["valid_solver_result_cases"], 1)
        self.assertTrue(report["negative_controls"]["passed"])
        self.assertEqual(receipt["report_digest"], report["report_digest"])
        self.assertIs(receipt["scientific_claims_allowed"], False)
        self.assertIs(receipt["eligible_for_champion"], False)
        self.assertIs(receipt["eligible_for_training_library"], False)

    def test_archived_fixture_manifest(self) -> None:
        root = REPO / "artifacts" / "causal_bridge_qwen7b"
        manifest = json.loads((root / "receipt_manifest.json").read_text())
        for name, digest in manifest["files"].items():
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
