from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade_multiple_choice  # noqa: E402


class MultipleChoiceAnswerExtractionTest(unittest.TestCase):
    def test_accepts_letter_with_optional_boxed_answer_text(self) -> None:
        cases = {
            r"Final: \boxed{A}": "A",
            r"Final: \boxed{\text{B}}": "B",
            r"Final: \boxed{C. Cold atomic interstellar medium}": "C",
            r"Final: \boxed{\text{D. answer text}}": "D",
            r"Final: \boxed{\mathbf{Option A: answer text}}": "A",
            r"Final: \boxed{\text{I. MMLU-Pro answer text}}": "I",
            r"First \boxed{A}, corrected to \boxed{\mathrm{B. final}}": "B",
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(grade_multiple_choice.extract_boxed_option(response), expected)

    def test_rejects_missing_ambiguous_or_non_letter_answers(self) -> None:
        responses = (
            "Final answer: C",
            r"Final: \boxed{1.5}",
            r"Final: \boxed{Carbon dioxide}",
            r"Final: \boxed{\text{None of the options}}",
            r"Final: \boxed{C",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(grade_multiple_choice.extract_boxed_option(response))

    def test_scores_grouped_rollouts_and_counts_extraction_errors(self) -> None:
        records = (
            {"example_id": 0, "answer": "C", "response": r"\boxed{C}"},
            {"example_id": 1, "answer": "A", "response": r"\boxed{B. wrong}"},
            {"example_id": 0, "answer": "C", "response": r"\boxed{\text{C. correct text}}"},
            {"example_id": 1, "answer": "A", "response": "Final answer: A"},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "gpqa_t0.7_p0.8_n2-MNT100.jsonl"
            with result_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            result = grade_multiple_choice.grade_file(result_path)

        self.assertEqual(result["mean_score"], 0.5)
        self.assertEqual(result["best_score"], 0.5)
        self.assertEqual(result["solve_all"], 1)
        self.assertEqual(result["solve_none"], 1)
        self.assertEqual(result["format_error_rollouts"], 1)
        self.assertEqual(result["parsed_rollouts"], 3)
        self.assertEqual(result["scorer"], "boxed_multiple_choice_v1")

    def test_rejects_inconsistent_rollout_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "mmlu-pro_t0.7_p0.8_n2-MNT100.jsonl"
            result_path.write_text(
                json.dumps({"example_id": 0, "answer": "A", "response": r"\boxed{A}"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 2 responses per question"):
                grade_multiple_choice.grade_file(result_path)

    def test_mmlu_pro_accepts_options_a_through_j(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "mmlu-pro_t0.7_p0.8_n1-MNT100.jsonl"
            result_path.write_text(
                json.dumps({"example_id": 0, "answer": "I", "response": r"\boxed{\text{I. answer text}}"})
                + "\n",
                encoding="utf-8",
            )
            result = grade_multiple_choice.grade_file(result_path)

        self.assertEqual(result["mean_score"], 1.0)
        self.assertEqual(result["format_error_rollouts"], 0)

    def test_gpqa_rejects_options_outside_a_through_d(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "gpqa_t0.7_p0.8_n1-MNT100.jsonl"
            result_path.write_text(
                json.dumps({"example_id": 0, "answer": "A", "response": r"\boxed{I. invalid for GPQA}"}) + "\n",
                encoding="utf-8",
            )
            result = grade_multiple_choice.grade_file(result_path)

        self.assertEqual(result["mean_score"], 0.0)
        self.assertEqual(result["format_error_rollouts"], 1)


if __name__ == "__main__":
    unittest.main()
