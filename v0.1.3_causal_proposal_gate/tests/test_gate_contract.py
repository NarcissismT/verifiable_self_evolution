from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).parents[1]


class GateContractTests(unittest.TestCase):
    def test_generator_has_no_template_fallback(self) -> None:
        source = (ROOT / "scripts" / "generate_causal_proposals.py").read_text()
        self.assertNotIn("template_implementation", source)
        self.assertIn("parse_model_proposal", source)
        self.assertIn("experiment_code", source)

    def test_gate_is_permanently_non_scientific(self) -> None:
        source = (ROOT / "scripts" / "run_gate.py").read_text()
        self.assertIn('"scientific_claims_allowed": False', source)
        self.assertIn('"eligible_for_champion": False', source)
        self.assertIn('"eligible_for_training_library": False', source)
        self.assertIn("negative-controls", source)

    def test_smoke_receipts_are_archived_outside_ignored_runs(self) -> None:
        artifact = Path(__file__).parents[2] / "artifacts" / "qlora_smoke_seed17" / "SMOKE_REPORT.json"
        value = json.loads(artifact.read_text())
        self.assertEqual(value["status"], "passed")
        self.assertIs(value["scientific_claims_allowed"], False)
        self.assertIs(value["eligible_for_champion"], False)
        self.assertIs(value["eligible_for_training_library"], False)

    def test_scripts_parse(self) -> None:
        subprocess.run(["python", "-m", "py_compile", *map(str, (ROOT / "scripts").glob("*.py"))], check=True)

    def test_real_run_is_fail_closed_and_non_scientific(self) -> None:
        result = Path(__file__).parents[2] / "artifacts" / "causal_gate_qwen7b" / "gate_environment.json"
        value = json.loads(result.read_text())
        self.assertEqual(value["positive_status"], "failed")
        self.assertEqual(value["negative_controls_status"], "passed")
        self.assertIs(value["scientific_claims_allowed"], False)
        self.assertIs(value["eligible_for_champion"], False)
        self.assertIs(value["eligible_for_training_library"], False)
        self.assertTrue(all(item[1] is False and item[2] == 0.0 for item in value["positive_cases"]))

    def test_historical_fixture_manifest_is_self_contained(self) -> None:
        root = Path(__file__).parents[2] / "artifacts" / "causal_gate_qwen7b"
        manifest = json.loads((root / "receipt_manifest.json").read_text())
        for name, digest in manifest["files"].items():
            import hashlib
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest)

    def test_protocol_fix_uses_dynamic_digest_and_continues_negatives(self) -> None:
        runner = (ROOT / "scripts" / "run_gate.py").read_text()
        launcher = (ROOT / "scripts" / "run_qwen_gate.sh").read_text()
        self.assertNotIn("EXECUTION_CONTAINER_DIGEST =", runner)
        self.assertIn("--execution-container-digest", runner)
        self.assertIn("POSITIVE_STATUS=$?", launcher)
        self.assertIn("NEGATIVE_STATUS=$?", launcher)
        self.assertIn("overall_report.json", launcher)

    def test_unit_tests_are_evaluator_owned(self) -> None:
        path = ROOT / "scripts" / "run_gate.py"
        spec = importlib.util.spec_from_file_location("v0131_gate", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        passed, total = module.independent_unit_tests(
            {"point": [0.0, 0.0], "oracle_calls": 1, "unit_tests": {"passed": 0, "total": 999}},
            [0.0, 0.0], 1, {"lower_residual": 0.0}, 2, 1000,
        )
        self.assertEqual((passed, total), (5, 5))


if __name__ == "__main__":
    unittest.main()
