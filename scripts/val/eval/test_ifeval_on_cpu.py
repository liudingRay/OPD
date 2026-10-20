from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ifeval  # noqa: E402
import grade_ifeval  # noqa: E402


@dataclass
class FakeInputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict]


def fake_strict(example: FakeInputExample, response: str) -> SimpleNamespace:
    followed = response == "strict"
    values = [followed] * len(example.instruction_id_list)
    return SimpleNamespace(follow_all_instructions=all(values), follow_instruction_list=values)


def fake_loose(example: FakeInputExample, response: str) -> SimpleNamespace:
    followed = response in {"strict", "loose"}
    values = [followed] * len(example.instruction_id_list)
    return SimpleNamespace(follow_all_instructions=all(values), follow_instruction_list=values)


class IFEvalPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_file = Path(__file__).resolve().parent.parent / "data" / "IFEval" / "ifeval_input_data.jsonl"
        cls.dataset = grade_ifeval.load_dataset(cls.data_file)

    def test_pinned_dataset_shape_and_identity(self) -> None:
        self.assertEqual(len(self.dataset), 541)
        self.assertEqual(sum(len(row["instruction_id_list"]) for row in self.dataset), 834)
        self.assertEqual(len({row["key"] for row in self.dataset}), 541)
        self.assertEqual(len({row["prompt"] for row in self.dataset}), 541)

    def test_zero_shot_message_preserves_prompt_exactly(self) -> None:
        prompt = self.dataset[0]["prompt"]
        self.assertEqual(eval_ifeval.build_chat_messages(prompt), [{"role": "user", "content": prompt}])

    def test_cli_defaults_match_primary_protocol(self) -> None:
        argv = [
            "eval_ifeval.py",
            "--model-name",
            "model",
            "--model-path",
            "/model",
            "--output-root",
            "/output",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = eval_ifeval.parse_args()
        self.assertFalse(args.enable_thinking)
        self.assertEqual(args.num_samples, 1)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_p, 1.0)
        self.assertEqual(args.top_k, -1)
        self.assertEqual(args.min_p, 0.0)
        self.assertEqual(args.max_tokens, 4096)

    def test_prompt_metrics_use_all_prompt_sample_pairs(self) -> None:
        dataset = [
            {
                "key": 1,
                "prompt": "one",
                "instruction_id_list": ["group:a"],
                "kwargs": [{}],
            },
            {
                "key": 2,
                "prompt": "two",
                "instruction_id_list": ["group:a", "group:b"],
                "kwargs": [{}, {}],
            },
        ]
        generations = [
            {"key": 1, "prompt": "one", "response": "strict", "sample_id": 0},
            {"key": 2, "prompt": "two", "response": "loose", "sample_id": 0},
            {"key": 1, "prompt": "one", "response": "fail", "sample_id": 1},
            {"key": 2, "prompt": "two", "response": "strict", "sample_id": 1},
        ]
        summary, details = grade_ifeval.score_records(
            dataset,
            generations,
            scorer=(FakeInputExample, fake_strict, fake_loose),
        )
        self.assertEqual(len(details), 4)
        self.assertEqual(summary["main_metrics"]["prompt_level_strict_acc"], 0.5)
        self.assertEqual(summary["main_metrics"]["prompt_level_loose_acc"], 0.75)
        self.assertEqual(summary["counts"]["samples_per_prompt"], 2)
        self.assertFalse(summary["partial_evaluation"])

    def test_partial_results_require_explicit_opt_in(self) -> None:
        row = self.dataset[0]
        generations = [{"key": row["key"], "prompt": row["prompt"], "response": "strict"}]
        with self.assertRaisesRegex(ValueError, "allow-partial"):
            grade_ifeval.score_records(
                self.dataset,
                generations,
                scorer=(FakeInputExample, fake_strict, fake_loose),
            )
        summary, _ = grade_ifeval.score_records(
            self.dataset,
            generations,
            allow_partial=True,
            scorer=(FakeInputExample, fake_strict, fake_loose),
        )
        self.assertTrue(summary["partial_evaluation"])
        self.assertEqual(summary["counts"]["evaluated_prompts_per_sample"], 1)

    def test_prompt_mismatch_is_rejected(self) -> None:
        row = self.dataset[0]
        generations = [{"key": row["key"], "prompt": row["prompt"] + " changed", "response": "strict"}]
        with self.assertRaisesRegex(ValueError, "does not match"):
            grade_ifeval.score_records(
                self.dataset,
                generations,
                allow_partial=True,
                scorer=(FakeInputExample, fake_strict, fake_loose),
            )

    def test_leonardo_launcher_uses_the_standard_protocol(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        launcher = (project_root / "leonardo" / "eval-ifeval-qwen3-model-1xa100").read_text(encoding="utf-8")
        expected_defaults = (
            "ENABLE_THINKING=${ENABLE_THINKING:-0}",
            "NUM_SAMPLES=${NUM_SAMPLES:-1}",
            "TEMPERATURE=${TEMPERATURE:-0.0}",
            "TOP_P=${TOP_P:-1.0}",
            "TOP_K=${TOP_K:--1}",
            "MIN_P=${MIN_P:-0.0}",
            "MAX_TOKENS=${MAX_TOKENS:-4096}",
            "SEED=${SEED:-0}",
        )
        for assignment in expected_defaults:
            with self.subTest(assignment=assignment):
                self.assertIn(assignment, launcher)


if __name__ == "__main__":
    unittest.main()
