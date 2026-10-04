"""CPU tests for saved-prefix EU scoring and complete export, without model downloads."""

import copy
import csv
import gzip
import math
from pathlib import Path
from types import SimpleNamespace

import eval_token_eu as evaluator
import numpy as np
import pytest
import torch


def records():
    rows = []
    for i, response in enumerate(([6, 8, 2], [7, 4, 3, 2])):
        rows.append(
            {
                "uid": f"q1:response-{i}",
                "question_uid": "q1",
                "response_index": i,
                "task": "AIME24",
                "question": "example question",
                "answer": "42",
                "prompt_token_ids": [1, 3, 5, 9],
                "response_token_ids": list(response),
                "response": "thinking and answer",
                "finish_reason": "stop",
                "seed": 42 + i,
            }
        )
    return rows


def source_file(tmp_path, rows=None):
    path = tmp_path / "source" / "pair" / "trajectories.json"
    path.parent.mkdir(parents=True)
    evaluator.write_json(path, rows if rows is not None else records())
    return path


class CachedTeacher:
    """Logits depend on the entire prefix, not only the chunk's last token."""

    def __init__(self, offset=0.0):
        self.offset = offset
        self.inputs = []

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        self.inputs.extend(input_ids[0].tolist())
        prefix_sum = input_ids.float().cumsum(-1) + (past_key_values or 0)
        vocab = torch.arange(32).float()
        logits = prefix_sum[..., None] * (vocab + 1) / 10 + self.offset
        return SimpleNamespace(logits=logits, past_key_values=float(prefix_sum[0, -1]))


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 128])
@pytest.mark.parametrize("prompt_length", [1, 4])
def test_cached_chunks_match_full_prefix_and_predict_current_token(
    chunk_size, prompt_length
):
    record = records()[0]
    record["prompt_token_ids"] = record["prompt_token_ids"][:prompt_length]
    model = CachedTeacher(4)
    result = evaluator.score_trajectory(
        model, record, chunk_size, "cpu", evaluator.load_eu_function()
    )
    full_input = record["prompt_token_ids"] + record["response_token_ids"][:-1]
    raw = CachedTeacher(4)(torch.tensor([full_input])).logits[0, prompt_length - 1 :]
    expected_top16 = torch.sort(raw, dim=-1, descending=True).values[:, :16]
    expected = (16 / (expected_top16 + 1).sum(-1)).numpy()
    np.testing.assert_allclose(result["eu"], expected)
    np.testing.assert_allclose(result["top16_logits"], expected_top16.numpy())
    np.testing.assert_array_equal(
        result["top16_ids"], np.tile(np.arange(31, 15, -1), (3, 1))
    )
    assert (
        model.inputs == full_input
    )  # no generation, duplicate chunks, or extra final token
    assert result["eu"].shape == (len(record["response_token_ids"]),)
    wrong_raw = CachedTeacher(4)(torch.tensor([full_input + [2]])).logits[
        0, prompt_length:
    ]
    wrong_eu = evaluator.load_eu_function()(wrong_raw).numpy()
    assert not np.allclose(result["eu"], wrong_eu)


def test_legacy_records_and_validation(tmp_path):
    rows = records()[:1]
    rows[0].pop("question_uid")
    rows[0].pop("response_index")
    path = source_file(tmp_path, rows)
    _, loaded = evaluator.load_trajectories(path.parent)
    assert loaded[0]["question_uid"] == rows[0]["uid"]
    assert loaded[0]["response_index"] == 0
    assert loaded[0]["response_mask"] == [1, 1, 1]
    assert evaluator.read_json(path) == rows  # original file unchanged
    for mutate in (
        lambda r: r[0].update(prompt_token_ids=[]),
        lambda r: r[0].update(response_token_ids=[-1]),
        lambda r: r[0].update(response_mask=[1]),
        lambda r: r.append(copy.deepcopy(r[0])),
    ):
        bad = copy.deepcopy(rows)
        mutate(bad)
        evaluator.write_json(path, bad)
        with pytest.raises(ValueError):
            evaluator.load_trajectories(path)


def fixture_scores(tmp_path):
    path = source_file(tmp_path)
    before = path.read_bytes()
    _, rows = evaluator.load_trajectories(path)
    root = tmp_path / "output"
    root.mkdir()
    evaluator.write_json(root / "trajectories.json", rows)
    evaluator.write_json(
        root / "config.json",
        {"teachers": [{"name": n} for n in evaluator.TEACHER_NAMES]},
    )
    for j, name in enumerate(evaluator.TEACHER_NAMES):
        directory = root / "teachers" / name
        directory.mkdir(parents=True)
        for i, row in enumerate(rows):
            result = evaluator.score_trajectory(
                CachedTeacher(j * 30), row, 2, "cpu", evaluator.load_eu_function()
            )
            evaluator.save_score(directory / f"trajectory-{i:06d}.npz", row, result)
        evaluator.write_json(directory / "COMPLETED.json", {"responses": len(rows)})
    return root, rows, path, before


def test_export_keeps_question_trajectory_and_three_teacher_token_axes(tmp_path):
    root, rows, path, before = fixture_scores(tmp_path)
    tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda ids: [f"token-{i}" for i in ids]
    )
    evaluator.summarize(root, tokenizer)
    with gzip.open(root / "tokens.csv.gz", "rt", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 7
    assert {r["question_uid"] for r in table} == {"q1"}
    assert {r["response_index"] for r in table} == {"0", "1"}
    for i, row in enumerate(rows):
        with np.load(
            root / "tokens" / f"trajectory-{i:06d}.npz", allow_pickle=False
        ) as joined:
            assert joined["eu"].shape == (len(row["response_token_ids"]), 3)
            assert joined["top16_logits"].shape == (
                len(row["response_token_ids"]),
                3,
                16,
            )
            np.testing.assert_array_equal(
                joined["teacher_names"], evaluator.TEACHER_NAMES
            )
            np.testing.assert_allclose(
                joined["eu"], 16 / (joined["top16_logits"] + 1).sum(-1)
            )
            assert np.all(joined["eu"][:, 0] > joined["eu"][:, 1])
            subset = [r for r in table if int(r["trajectory_index"]) == i]
            np.testing.assert_allclose(
                [float(r["eu_Qwen3-14B"]) for r in subset], joined["eu"][:, 2]
            )
    summary = evaluator.read_json(root / "summary.json")
    assert (
        summary["questions"] == 1
        and summary["responses"] == 2
        and summary["tokens"] == 7
    )
    assert len(summary["pairwise"]) == 3
    assert path.read_bytes() == before
    assert (root / "COMPLETED.json").exists()
    with pytest.raises(FileExistsError):
        evaluator.summarize(root, tokenizer)


def test_export_rejects_mismatched_prefix_even_when_lengths_match(tmp_path):
    root, _, _, _ = fixture_scores(tmp_path)
    path = root / "teachers" / "Qwen3-8B" / "trajectory-000000.npz"
    with np.load(path, allow_pickle=False) as content:
        arrays = {k: content[k].copy() for k in content.files}
    arrays["prefix_sha256"] = np.asarray("wrong-prefix")
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError, match="prefix/token identity"):
        evaluator.summarize(root, SimpleNamespace(convert_ids_to_tokens=lambda x: x))
    assert not (root / "COMPLETED.json").exists()


def test_statistics_reports_nonfinite_and_preserves_negative_values():
    stats = evaluator.statistics([1, 3, -1, np.inf, np.nan])
    assert (
        stats["mean"] == 1
        and stats["nonfinite_count"] == 2
        and stats["nonpositive_count"] == 1
    )
    assert stats["std"] == pytest.approx(math.sqrt(8 / 3))
    assert evaluator.statistics([])["mean"] is None


def test_preflight_tokenizer_and_context_checks(tmp_path, monkeypatch):
    import sys

    source, rows = evaluator.load_trajectories(source_file(tmp_path))
    tokenizer = SimpleNamespace(
        get_vocab=lambda: {str(i): i for i in range(32)}, special_tokens_map={}
    )
    signature = evaluator.tokenizer_signature(tokenizer)
    evaluator.write_json(
        source.parent.parent / "config.json",
        {
            "thinking": True,
            "pairs": [
                {
                    "label": "pair",
                    "tokenizer_sha256": signature,
                    "student": "/old/student",
                }
            ],
        },
    )
    teachers = []
    for name in evaluator.TEACHER_NAMES:
        path = tmp_path / name
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"fixture, not real weights")
        (path / "config.json").write_text("{}")
        teachers.append({"name": name, "path": str(path)})
    config = SimpleNamespace(
        vocab_size=32, model_type="qwen3", max_position_embeddings=100
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
        ),
    )
    result = evaluator.preflight(source, rows, teachers)
    assert result["thinking"] is True
    config.max_position_embeddings = 3
    with pytest.raises(ValueError, match="context"):
        evaluator.preflight(source, rows, teachers)
    config.max_position_embeddings = 100
    tokenizer.special_tokens_map = {"eos_token": "different"}
    with pytest.raises(ValueError, match="token-ID mapping"):
        evaluator.preflight(source, rows, teachers)


def test_alex_launcher_preflight_and_submission_contract(tmp_path):
    import os
    import subprocess

    workspace = tmp_path / "workspace"
    project = Path(__file__).resolve().parents[3]
    environment = workspace / "envs/opd-py312"
    binaries = environment / "bin"
    binaries.mkdir(parents=True)
    (binaries / "activate").write_text(f'export PATH="{binaries}:$PATH"\n')
    calls = tmp_path / "calls.txt"
    for name, body in {
        "module": "exit 0",
        "nvcc": "exit 0",
        "python": 'printf "python %s\\n" "$*" >> "$TEST_CALLS"',
        "sbatch": 'printf "sbatch %s\\n" "$*" >> "$TEST_CALLS"',
    }.items():
        path = binaries / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    for name in evaluator.TEACHER_NAMES:
        path = workspace / "models" / name
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"fixture")
    source = source_file(tmp_path)
    output = tmp_path / "new-eu-output"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "OPD_WS": str(workspace),
        "PROJECT_DIR": str(project),
        "TRAJECTORIES_FILE": str(source),
        "EU_OUTPUT_ROOT": str(output),
        "DRY_RUN": "1",
        "TEST_CALLS": str(calls),
    }
    env.pop("SLURM_JOB_ID", None)
    script = project / "alex/eval-token-eu"
    subprocess.run(
        ["bash", str(script)], env=env, check=True, capture_output=True, text=True
    )
    assert "--dry-run" in calls.read_text() and "sbatch" not in calls.read_text()
    assert not output.exists()
    env["DRY_RUN"] = "0"
    subprocess.run(
        ["bash", str(script)], env=env, check=True, capture_output=True, text=True
    )
    assert "sbatch --export=ALL --job-name=eval-token-eu" in calls.read_text()
    assert not output.exists()
    before = calls.read_text()
    output.mkdir()
    rejected = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, check=False
    )
    assert rejected.returncode != 0
    assert calls.read_text() == before  # rejection precedes preflight/submission
