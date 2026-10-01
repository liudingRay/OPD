"""CPU checks: /path/to/python -m unittest discover -s scripts/val/eval -p test_fixed_overlap.py"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("fixed_overlap", Path(__file__).with_name("eval_fixed_overlap.py"))
overlap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlap)


class OverlapTests(unittest.TestCase):
    def test_repeated_response_ids_seeds_and_question_mean(self):
        samples = [
            {"uid": "AIME24:1", "task": "AIME24", "question": "q1"},
            {"uid": "AIME24:2", "task": "AIME24", "question": "q2"},
        ]
        expanded = overlap.expand_response_samples(samples, 4, 42)
        self.assertEqual(len(expanded), 8)
        self.assertEqual([row["seed"] for row in expanded], list(range(42, 50)))
        self.assertEqual(len({row["uid"] for row in expanded}), 8)
        self.assertEqual({row["question_uid"] for row in expanded}, {"AIME24:1", "AIME24:2"})
        rows = [
            {"uid": "a:1", "question_uid": "a", "value": 1.0},
            {"uid": "a:2", "question_uid": "a", "value": 3.0},
            {"uid": "b:1", "question_uid": "b", "value": 10.0},
        ]
        self.assertEqual(overlap._question_mean(rows, "value"), 6.0)

    def test_mass_is_not_renormalized_or_shared_between_models(self):
        result = overlap.token_overlap([0, 1], [.6, .2], [1, 2], [.5, .3], 2)
        np.testing.assert_allclose(result, [.5, .2, .5])
        self.assertEqual(overlap.token_overlap([0], [.9], [1], [.8], 1), (0, 0, 0))
        np.testing.assert_allclose(overlap.token_overlap([0], [.9], [0], [.8], 1), [1, .9, .8])

    def test_topk_mechanism_metrics_distinguish_sparse_shared_and_exact(self):
        student_ids = np.asarray([[0, 1], [0, 1]])
        student_probs = np.asarray([[.6, .2], [.7, .1]])
        teacher_ids = np.asarray([[0, 2], [2, 3]])
        teacher_probs = np.asarray([[.5, .3], [.6, .2]])
        teacher_on_student = np.log(np.asarray([[.5, .1], [.05, .02]]))
        result = overlap.topk_mechanism_metrics(
            student_ids, student_probs, teacher_ids, teacher_probs, 2, teacher_on_student
        )
        expected_cosine = .6 * .5 / np.sqrt((.6 ** 2 + .2 ** 2) * (.5 ** 2 + .3 ** 2))
        self.assertAlmostEqual(result["sparse_union_cosine"][0], expected_cosine)
        self.assertEqual(result["sparse_union_cosine"][1], 0)
        self.assertAlmostEqual(
            result["shared_weighted_abs_logprob_gap"][0], abs(np.log(.6) - np.log(.5))
        )
        self.assertTrue(np.isnan(result["shared_weighted_abs_logprob_gap"][1]))
        weights = np.asarray([.6, .2]) / .8
        expected_gap = np.sum(weights * np.abs(np.log([.6, .2]) - teacher_on_student[0]))
        self.assertAlmostEqual(result["student_topk_weighted_abs_logprob_gap"][0], expected_gap)

    def test_historical_analysis_is_cpu_only_and_marks_exact_gap_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            output = Path(folder) / "analysis"
            pair = source / "stage1"
            pair.mkdir(parents=True)
            config = {"pairs": [{"label": "stage1"}], "ks": [2]}
            overlap.write_json(source / "config.json", config)
            overlap.write_json(source / "COMPLETED.json", {"complete": True})
            records = []
            for index, task in enumerate(overlap.TASKS):
                records.append({"uid": f"{task}:0", "task": task, "response_token_ids": [0],
                                "finish_reason": "stop"})
                np.savez(pair / f"student-topk-{index:03d}.npz",
                         ids=np.asarray([[0, 1]]), probabilities=np.asarray([[.6, .2]]))
                np.savez(pair / f"teacher-topk-{index:03d}.npz",
                         ids=np.asarray([[0, 2]]), probabilities=np.asarray([[.5, .3]]))
            overlap.write_json(pair / "trajectories.json", records)
            overlap.analyze_existing(config, source, output)
            summary = [row for row in overlap.read_json(output / "summary.json")
                       if row["task"] == "ALL"][0]
            self.assertGreater(summary["token_mean_sparse_union_cosine"], 0)
            self.assertGreater(summary["token_mean_shared_weighted_abs_logprob_gap"], 0)
            self.assertIsNone(summary["token_mean_student_topk_weighted_abs_logprob_gap"])
            self.assertFalse(summary["exact_gap_available"])
            self.assertTrue((output / "stage1" / "mechanism-k2-000.npz").is_file())
            with self.assertRaises(ValueError):
                overlap.analyze_existing(config, source, output)

            for index in range(len(overlap.TASKS)):
                np.savez(pair / f"teacher-on-student-topk-{index:03d}.npz",
                         ids=np.asarray([[0, 1]]), log_probabilities=np.log(np.asarray([[.5, .1]])))
            exact_output = Path(folder) / "exact-analysis"
            overlap.analyze_existing(config, source, exact_output)
            exact_summary = [row for row in overlap.read_json(exact_output / "summary.json")
                             if row["task"] == "ALL"][0]
            self.assertTrue(exact_summary["exact_gap_available"])
            self.assertGreater(exact_summary["token_mean_student_topk_weighted_abs_logprob_gap"], 0)

    def test_summary_matches_full_vocab_reference_and_weighting(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "stage1"
            directory.mkdir()
            records, reference = [], []
            for index, (task, n) in enumerate(zip(overlap.TASKS, [1, 3, 7])):
                records.append({"uid": f"{task}:0", "task": task,
                                "response_token_ids": list(range(n)), "finish_reason": "length"})
                distributions = []
                for role in ("student", "teacher"):
                    logits = rng.normal(size=(n, 20))
                    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
                    probs /= probs.sum(axis=1, keepdims=True)
                    ids = np.argsort(-probs, axis=1)[:, :16]
                    masses = np.take_along_axis(probs, ids, axis=1)
                    np.savez(directory / f"{role}-topk-{index:03d}.npz", ids=ids, probabilities=masses)
                    distributions.append((ids, masses))
                for k in [4, 8, 16]:
                    values = [overlap.token_overlap(*[a[pos].tolist() for pair in distributions for a in pair], k)
                              for pos in range(n)]
                    reference.append((k, n, np.asarray(values).mean(axis=0)))
            overlap.write_json(directory / "trajectories.json", records)
            overlap.summarize({"pairs": [{"label": "stage1"}], "ks": [4, 8, 16]}, root)
            summary = overlap.read_json(root / "summary.json")
            for row in summary:
                if row["task"] != "ALL":
                    continue
                expected = [r for r in reference if r[0] == row["k"]]
                for col, name in enumerate(("ratio", "student_mass", "teacher_mass")):
                    token_mean = sum(n * values[col] for _, n, values in expected) / 11
                    question_mean = sum(values[col] for _, _, values in expected) / 3
                    self.assertAlmostEqual(row[f"token_mean_{name}"], token_mean)
                    self.assertAlmostEqual(row[f"question_mean_{name}"], question_mean)
                self.assertEqual(row["length_limited"], 3)
            self.assertTrue((root / "summary.csv").is_file())

    def test_fixed_selection_on_real_benchmarks(self):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        first = overlap.select_samples(data_dir, 5, 42)
        second = overlap.select_samples(data_dir, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first["samples"]), 15)
        self.assertEqual(len({s["uid"] for s in first["samples"]}), 15)
        for task in overlap.TASKS:
            self.assertEqual(sum(s["task"] == task for s in first["samples"]), 5)
        with self.assertRaises(ValueError):
            overlap.select_samples(data_dir, 10000, 42)

    def test_position_bins_partial_suffix_and_empty_intersections(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "stage1"
            directory.mkdir()
            records = []
            for index, (task, n) in enumerate(zip(overlap.TASKS, [1025, 1024, 1])):
                records.append({"uid": f"{task}:0", "task": task,
                                "response_token_ids": [0] * n, "finish_reason": "stop"})
                student_ids = np.tile(np.arange(4), (n, 1))
                teacher_ids = student_ids.copy()
                if index == 0:
                    teacher_ids[-1] += 4  # Empty overlap in the partial final bin.
                if index == 1:
                    teacher_ids += 4  # Entire second question has empty overlap.
                np.savez(directory / f"student-topk-{index:03d}.npz",
                         ids=student_ids, probabilities=np.full((n, 4), .1))
                np.savez(directory / f"teacher-topk-{index:03d}.npz",
                         ids=teacher_ids, probabilities=np.full((n, 4), .05))
            overlap.write_json(directory / "trajectories.json", records)
            overlap.summarize({"pairs": [{"label": "stage1"}], "ks": [4]}, root)
            chunks = [r for r in overlap.read_json(root / "summary_chunks.json") if r["task"] == "ALL"]
            self.assertEqual(len(chunks), 2)
            first, last = chunks
            self.assertEqual((first["start"], first["end"], first["tokens"], first["questions"]),
                             (0, 1024, 2049, 3))
            self.assertEqual(first["responses"], 3)
            self.assertEqual(first["nonempty_tokens"], 1025)
            self.assertAlmostEqual(first["token_mean_ratio"], 1025 / 2049)
            self.assertAlmostEqual(first["token_mean_student_mass"], .4 * 1025 / 2049)
            self.assertAlmostEqual(first["token_mean_student_mass_nonempty"], .4)
            self.assertAlmostEqual(first["token_mean_teacher_mass_nonempty"], .2)
            self.assertEqual((last["start"], last["end"], last["tokens"], last["questions"]),
                             (1024, 1025, 1, 1))
            self.assertEqual(last["token_mean_student_mass"], 0)
            self.assertIsNone(last["token_mean_student_mass_nonempty"])

    def test_safe_pair_labels(self):
        self.assertEqual(overlap.parse_pair("stage1=/model/student,/model/teacher")["label"], "stage1")
        with self.assertRaises(Exception):
            overlap.parse_pair("../escape=/student,/teacher")

    def test_trajectory_reuse_requires_earlier_exact_student(self):
        pairs = [
            {"label": "base4", "student": "/student", "teacher": "/teacher4"},
            {"label": "base8", "student": "/student", "teacher": "/teacher8"},
        ]
        overlap.apply_trajectory_reuse(pairs, [overlap.parse_reuse_spec("base8=base4")])
        self.assertEqual(pairs[1]["trajectory_source"], "base4")
        with self.assertRaises(ValueError):
            overlap.apply_trajectory_reuse(pairs, [("base4", "base8")])
        different = [
            {"label": "a", "student": "/student-a", "teacher": "/teacher4"},
            {"label": "b", "student": "/student-b", "teacher": "/teacher8"},
        ]
        with self.assertRaises(ValueError):
            overlap.apply_trajectory_reuse(different, [("b", "a")])

    def test_copy_reused_student_outputs_excludes_teacher_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "base4", root / "base8"
            source.mkdir(); target.mkdir()
            for name in ("trajectories.json", "trajectory-000.json", "student-topk-000.npz",
                         "teacher-topk-000.npz", "overlap-k4-000.npz"):
                (source / name).write_bytes(name.encode())
            overlap.copy_reused_student_outputs(root, {"label": "base8", "trajectory_source": "base4"})
            self.assertEqual(
                sorted(path.name for path in target.iterdir()),
                ["student-topk-000.npz", "trajectories.json", "trajectory-000.json"],
            )


if __name__ == "__main__":
    unittest.main()
