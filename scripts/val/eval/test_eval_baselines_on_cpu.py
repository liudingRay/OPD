from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_baselines  # noqa: E402


class EvalBaselinePromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]

    def load_first_prompt(self, task_name: str) -> str:
        path = self.project_root / "datasets" / "test_data" / task_name / "test.parquet_first_row.json"
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        return row["prompt"][0]["content"]

    def test_new_multiple_choice_tasks_are_registered(self) -> None:
        self.assertIn("GPQA", eval_baselines.SUPPORTED_TASKS)
        self.assertIn("MMLU-Pro", eval_baselines.SUPPORTED_TASKS)

    def test_multiple_choice_prompts_are_preserved_verbatim(self) -> None:
        for task_name in ("GPQA", "MMLU-Pro"):
            with self.subTest(task_name=task_name):
                question = self.load_first_prompt(task_name)
                self.assertEqual(eval_baselines.build_user_prompt(task_name, question), question)

    def test_existing_math_prompt_protocol_is_unchanged(self) -> None:
        question = "What is 1 + 1?"
        expected = eval_baselines.PROMPT_TEMPLATE.format(problem=question)
        self.assertEqual(eval_baselines.build_user_prompt("AIME24", question), expected)
        self.assertEqual(eval_baselines.build_user_prompt("MATH-500", question), expected)

    def test_alex_multiple_choice_launchers_use_independent_sample_counts(self) -> None:
        alex_dir = self.project_root / "alex"
        gpqa_launcher = (alex_dir / "eval-gpqa-qwen3-model-1xa100").read_text(encoding="utf-8")
        mmlupro_launcher = (alex_dir / "eval-mmlupro-qwen3-model-1xa100").read_text(encoding="utf-8")
        self.assertIn("export TASKS=GPQA", gpqa_launcher)
        self.assertIn("export NUM_SAMPLES=8", gpqa_launcher)
        self.assertIn("export MAX_EXAMPLES=1", gpqa_launcher)
        self.assertIn("export TASKS=MMLU-Pro", mmlupro_launcher)
        self.assertIn("export NUM_SAMPLES=1", mmlupro_launcher)
        self.assertIn("export MAX_EXAMPLES=4", mmlupro_launcher)

    def test_alex_multiple_choice_launchers_match_aime_sampling_defaults(self) -> None:
        alex_dir = self.project_root / "alex"
        launchers = (
            alex_dir / "eval-gpqa-qwen3-model-1xa100",
            alex_dir / "eval-mmlupro-qwen3-model-1xa100",
        )
        expected_common_assignments = (
            "export ENABLE_THINKING=0",
            "export TEMPERATURE=0.7",
            "export TOP_P=0.8",
            "export TOP_K=20",
            "export MIN_P=0.0",
            "export MAX_TOKENS=38912",
            "export SEED=0",
        )
        for launcher_path in launchers:
            launcher = launcher_path.read_text(encoding="utf-8")
            for assignment in expected_common_assignments:
                with self.subTest(launcher=launcher_path.name, assignment=assignment):
                    self.assertIn(assignment, launcher)

    def test_leonardo_multiple_choice_launchers_match_the_alex_protocols(self) -> None:
        leonardo_dir = self.project_root / "leonardo"
        expected = {
            "eval-gpqa-qwen3-model-1xa100": ("GPQA", 8, 1),
            "eval-mmlupro-qwen3-model-1xa100": ("MMLU-Pro", 1, 4),
        }
        for filename, (task, samples, smoke_examples) in expected.items():
            launcher = (leonardo_dir / filename).read_text(encoding="utf-8")
            with self.subTest(launcher=filename):
                self.assertIn(f"export TASKS={task}", launcher)
                self.assertIn(f"export NUM_SAMPLES={samples}", launcher)
                self.assertIn(f"export MAX_EXAMPLES={smoke_examples}", launcher)
                self.assertIn("export ENABLE_THINKING=0", launcher)
                self.assertIn("export TEMPERATURE=0.7", launcher)
                self.assertIn("export TOP_P=0.8", launcher)
                self.assertIn("export TOP_K=20", launcher)
                self.assertIn("export MIN_P=0.0", launcher)
                self.assertIn("export MAX_TOKENS=38912", launcher)
                self.assertIn("export SEED=0", launcher)
                self.assertIn("EVAL_ACCOUNT=${EVAL_ACCOUNT:-AIFAC_S07_051}", launcher)
                self.assertIn('--account="$EVAL_ACCOUNT"', launcher)
                self.assertIn("submit_eval_qwen3_model_1xa100.sbatch", launcher)

        shared_launcher = (leonardo_dir / "submit_eval_qwen3_model_1xa100.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --account=AIFAC_S07_051", shared_launcher)

    def test_leonardo_vllm_evaluations_use_project_cache(self) -> None:
        leonardo_dir = self.project_root / "leonardo"
        launchers = (
            "eval-ifeval-qwen3-model-1xa100",
            "submit_eval_qwen3_model_1xa100.sbatch",
            "submit_eval_qwen3_thinking_baselines_1xa100.sbatch",
            "submit_eval_qwen3_1p7b_opd_1xa100.sbatch",
            "submit_eval_qwen3_14b_task_2xa100.sbatch",
        )
        for filename in launchers:
            launcher = (leonardo_dir / filename).read_text(encoding="utf-8")
            with self.subTest(launcher=filename):
                self.assertIn("VLLM_CACHE_ROOT", launcher)
                self.assertIn("$OPD_CACHE_ROOT/vllm/", launcher)


if __name__ == "__main__":
    unittest.main()
