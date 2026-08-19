from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "smoke_train", ROOT / "scripts" / "train_and_verify.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [index + 1 for index, _ in enumerate(text.split())]


class SmokeContractTests(unittest.TestCase):
    def test_config_is_permanently_ineligible(self) -> None:
        config = json.loads((ROOT / "config" / "smoke_config.json").read_text())
        self.assertEqual(config["study_kind"], "qlora_pipeline_smoke")
        self.assertIs(config["scientific_claims_allowed"], False)
        self.assertIs(config["eligible_for_champion"], False)
        self.assertIs(config["eligible_for_training_library"], False)
        self.assertEqual(config["model"]["model_id"], "qwen2.5-7b-instruct")
        self.assertEqual(config["training"]["adapter_seeds"], [17])
        self.assertEqual(config["training"]["max_optimizer_updates"], 100)

    def test_segmented_mask_obeys_frozen_roles(self) -> None:
        row = {
            "segments": [
                {"segment_type": "system", "role": "system", "content": "system text"},
                {"segment_type": "user", "role": "user", "content": "user text"},
                {"segment_type": "assistant_plan", "role": "assistant", "content": "plan text"},
                {"segment_type": "assistant_action", "role": "assistant", "content": "action text"},
                {"segment_type": "tool_observation", "role": "tool", "content": "tool text"},
                {"segment_type": "structured_result_explanation", "role": "assistant", "content": "result text"},
                {"segment_type": "belief_update", "role": "assistant", "content": "belief text"},
            ]
        }
        encoded = MODULE.encode_segmented_example(FakeTokenizer(), row, 512)
        for item in encoded["ranges"]:
            labels = encoded["labels"][item["start"]:item["end"]]
            if item["loss_on"]:
                self.assertTrue(all(value != -100 for value in labels))
            else:
                self.assertTrue(all(value == -100 for value in labels))

    def test_generator_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            import subprocess
            command = [
                "python", str(ROOT / "scripts" / "generate_synthetic_data.py"),
                "--config", str(ROOT / "config" / "smoke_config.json"),
            ]
            subprocess.run(command + ["--output-root", first], check=True, capture_output=True)
            subprocess.run(command + ["--output-root", second], check=True, capture_output=True)
            self.assertEqual(
                (Path(first) / "synthetic_train.jsonl").read_bytes(),
                (Path(second) / "synthetic_train.jsonl").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
