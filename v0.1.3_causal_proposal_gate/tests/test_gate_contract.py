from __future__ import annotations

import json
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
        result = ROOT / "runs" / "qwen7b_seed17_gate" / "gate_environment.json"
        value = json.loads(result.read_text())
        self.assertEqual(value["positive_status"], "failed")
        self.assertEqual(value["negative_controls_status"], "passed")
        self.assertIs(value["scientific_claims_allowed"], False)
        self.assertIs(value["eligible_for_champion"], False)
        self.assertIs(value["eligible_for_training_library"], False)
        self.assertTrue(all(item[1] is False and item[2] == 0.0 for item in value["positive_cases"]))


if __name__ == "__main__":
    unittest.main()
