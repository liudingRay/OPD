"""CPU-only tests for text-rollout reconstruction and EU report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import eval_student_rollout_eu as evaluator
import numpy as np


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt
        rendered = f"<chat thinking={int(enable_thinking)}>{messages[0]['content']}<assistant>"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return bytes(ids).decode("utf-8")

    def convert_ids_to_tokens(self, ids):
        return [bytes([value]).decode("utf-8", errors="replace") for value in ids]


def source_rows():
    return [
        {
            "example_id": 0,
            "question": "What is 1+1?",
            "prompt": "What is 1+1? Explain.",
            "answer": "2",
            "seed": 0,
            "response": "Two: 2",
            "_source_line": 1,
        },
        {
            "example_id": 0,
            "question": "What is 1+1?",
            "prompt": "What is 1+1? Explain.",
            "answer": "2",
            "seed": 1,
            "response": "It is 2",
            "_source_line": 2,
        },
    ]


def test_reconstructs_visible_text_and_stable_identities():
    tokenizer = FakeTokenizer()
    records = evaluator.build_trajectories(
        source_rows(), tokenizer, {"enable_thinking": False}, "AMC23"
    )
    assert [record["uid"] for record in records] == [
        "AMC23:0:response-00",
        "AMC23:0:response-01",
    ]
    assert records[0]["question_uid"] == "AMC23:0"
    assert tokenizer.decode(records[0]["response_token_ids"]) == "Two: 2"
    assert records[0]["response_mask"] == [1] * len(records[0]["response_token_ids"])
    assert records[0]["finish_reason"] == "not_recorded_by_eval_baselines"
    assert len(evaluator.build_trajectories(source_rows(), tokenizer, {}, "AMC23", 1)) == 1


def test_rejects_duplicate_example_response_pairs():
    rows = source_rows()
    rows[1]["seed"] = rows[0]["seed"]
    try:
        evaluator.build_trajectories(rows, FakeTokenizer(), {}, "AMC23")
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Duplicate rollout identity was accepted")


def fixture_output(root: Path):
    tokenizer = FakeTokenizer()
    records = evaluator.build_trajectories(source_rows(), tokenizer, {}, "AMC23")
    names = list(evaluator.token_eu.TEACHER_NAMES)
    evaluator.write_json(root / "config.json", {"teachers": [{"name": name} for name in names]})
    evaluator.write_json(root / "trajectories.json", records)
    tokens = root / "tokens"
    tokens.mkdir()
    for index, record in enumerate(records):
        n = len(record["response_token_ids"])
        base = np.linspace(0.1, 1.0, n, dtype=np.float64)
        eu = np.stack([base, base[::-1] + 0.3, base * 2 + index], axis=1)
        np.savez_compressed(
            tokens / f"trajectory-{index:06d}.npz",
            eu=eu,
            response_mask=np.ones(n, dtype=bool),
            response_token_ids=np.asarray(record["response_token_ids"], dtype=np.int64),
            teacher_names=np.asarray(names),
        )
    return tokenizer, records


def test_analysis_exports_stats_high_tokens_and_self_contained_visuals(tmp_path):
    tokenizer, records = fixture_output(tmp_path)
    evaluator.analyze_output(tmp_path, tokenizer, ["0:0"], top_n=2)
    analysis = tmp_path / "analysis"
    expected = (
        "teacher_summary.json",
        "per_trajectory_teacher_stats.json",
        "per_trajectory_teacher_stats.csv",
        "high_eu_tokens.csv",
        "teacher_distribution.svg",
        "report.html",
        "ANALYSIS_COMPLETED.json",
        "trajectories/trajectory-000000.svg",
        "trajectories/trajectory-000000.html",
    )
    assert all((analysis / name).is_file() for name in expected)
    summary = json.loads((analysis / "teacher_summary.json").read_text())
    assert len(summary) == 3 and all(row["finite_tokens"] > 0 for row in summary)
    with (analysis / "high_eu_tokens.csv").open() as handle:
        high = list(csv.DictReader(handle))
    assert high and {row["trajectory_uid"] for row in high} == {record["uid"] for record in records}
    report = (analysis / "report.html").read_text()
    assert "raw-logit scales are not calibrated" in report.lower()
    assert "trajectory-000000.html" in report
    assert not (analysis / "trajectories/trajectory-000001.html").exists()


def test_extended_statistics_keeps_nonfinite_counts_and_quantiles():
    stats = evaluator.extended_statistics([1.0, 2.0, 3.0, np.inf, np.nan])
    assert stats["finite_tokens"] == 3
    assert stats["nonfinite_count"] == 2
    assert stats["mean"] == 2.0
    assert stats["p50"] == 2.0
