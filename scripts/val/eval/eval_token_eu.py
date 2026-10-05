#!/usr/bin/env python3
"""Score saved student trajectories with raw top-16 EU from one student or three teachers.

No generation, re-tokenization, temperature scaling, training, or normalization.
Each scoring model runs in its own process on one GPU; completed per-trajectory scores
are retained if a later phase fails. Existing output roots are never replaced.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

from eval_fixed_overlap import file_hash, read_json, tokenizer_signature, write_json

TEACHER_NAMES = ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B")
EU_MODULE = (
    Path(__file__).resolve().parents[3] / "verl/verl/utils/epistemic_uncertainty.py"
)


def load_eu_function():
    # Import the pure torch utility without initializing verl/Ray/FSDP.
    spec = importlib.util.spec_from_file_location("opd_eu", EU_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_token_level_eu


def trajectory_hash(record):
    return hashlib.sha256(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "uid",
                    "question_uid",
                    "response_index",
                    "prompt_token_ids",
                    "response_token_ids",
                    "response_mask",
                )
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def load_trajectories(source):
    source = Path(source).resolve()
    if source.is_dir():
        source = source / "trajectories.json"
    if not source.is_file():
        raise ValueError(
            f"Select one pair directory's trajectories.json, not the overall run root: {source}"
        )
    records = read_json(source)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON list of saved trajectories")
    seen, question_responses, questions = set(), set(), {}
    for index, row in enumerate(records):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) or not row[k]
            for k in ("uid", "task", "question")
        ):
            raise ValueError(
                f"Trajectory {index} requires nonempty uid, task and question strings"
            )
        if row["uid"] in seen:
            raise ValueError(f"Duplicate trajectory UID: {row['uid']}")
        seen.add(row["uid"])
        row.setdefault("question_uid", row["uid"])
        row.setdefault("response_index", 0)
        if not isinstance(row["question_uid"], str) or not row["question_uid"]:
            raise ValueError("Invalid question_uid")
        if type(row["response_index"]) is not int or row["response_index"] < 0:
            raise ValueError("Invalid response_index")
        key = (row["task"], row["question_uid"])
        if questions.setdefault(key, row["question"]) != row["question"]:
            raise ValueError(f"Inconsistent question text for {key}")
        response_key = (*key, row["response_index"])
        if response_key in question_responses:
            raise ValueError(f"Duplicate question/response index: {response_key}")
        question_responses.add(response_key)
        for name in ("prompt_token_ids", "response_token_ids"):
            ids = row.get(name)
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(v) is not int or v < 0 for v in ids)
            ):
                raise ValueError(
                    f"{row['uid']}: {name} must contain original nonnegative integer token IDs"
                )
        n = len(row["response_token_ids"])
        row.setdefault("response_mask", [1] * n)
        if (
            not isinstance(row["response_mask"], list)
            or len(row["response_mask"]) != n
            or any(v not in (0, 1) for v in row["response_mask"])
        ):
            raise ValueError(
                f"{row['uid']}: response_mask must be binary and align with response tokens"
            )
        row["trajectory_index"] = index
        row["prefix_sha256"] = trajectory_hash(row)
    return source, records


def source_metadata(source):
    path = source.parent.parent / "config.json"
    if path.is_file():
        config = read_json(path)
        pair = next(
            (
                p
                for p in config.get("pairs", [])
                if p.get("label") == source.parent.name
            ),
            None,
        )
        if pair:
            return {
                "config_path": str(path),
                "config_sha256": file_hash(path),
                "pair": pair,
                "thinking": config.get("thinking"),
                "generation": {
                    k: config.get(k)
                    for k in (
                        "seed",
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "generation_top_k",
                        "samples_per_question",
                    )
                },
            }
    return {}


def model_files(path):
    path = Path(path)
    config = path / "config.json"
    weights = sorted(path.glob("*.safetensors"))
    if not path.is_absolute() or not config.is_file() or not weights:
        raise ValueError(
            f"Expected absolute complete Hugging Face teacher directory: {path}"
        )
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = set(read_json(index).get("weight_map", {}).values())
        if not shards or any(not (path / name).is_file() for name in shards):
            raise ValueError(f"Missing model shards: {path}")
    return {
        "config_sha256": file_hash(config),
        "weights": [
            {
                "name": w.name,
                "bytes": w.stat().st_size,
                "mtime_ns": w.stat().st_mtime_ns,
            }
            for w in weights
        ],
    }


def preflight(source, records, teachers, tokenizer_path=None):
    from transformers import AutoConfig, AutoTokenizer

    origin = source_metadata(source)
    expected_signature = origin.get("pair", {}).get("tokenizer_sha256")
    if tokenizer_path:
        signature = tokenizer_signature(
            AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        )
        if expected_signature and expected_signature != signature:
            raise ValueError(
                "Explicit source tokenizer conflicts with the saved trajectory tokenizer fingerprint"
            )
        expected_signature = signature
    if not expected_signature:
        raise ValueError(
            "Missing saved tokenizer fingerprint; specify --tokenizer-path for the original student"
        )
    max_id = max(
        max(r[k]) for r in records for k in ("prompt_token_ids", "response_token_ids")
    )
    max_context = max(
        len(r["prompt_token_ids"]) + len(r["response_token_ids"]) - 1 for r in records
    )
    for teacher in teachers:
        teacher["files"] = model_files(teacher["path"])
        config = AutoConfig.from_pretrained(teacher["path"], local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            teacher["path"], local_files_only=True
        )
        if tokenizer_signature(tokenizer) != expected_signature:
            raise ValueError(
                f"{teacher['name']}: teacher token-ID mapping differs from saved student trajectories"
            )
        if (
            config.model_type != "qwen3"
            or config.vocab_size < 16
            or max_id >= config.vocab_size
        ):
            raise ValueError(
                f"{teacher['name']}: requires Qwen3 with compatible token IDs and vocabulary"
            )
        if max_context > config.max_position_embeddings:
            raise ValueError(
                f"{teacher['name']}: saved prefixes exceed context; no silent truncation is allowed"
            )
        teacher["tokenizer_sha256"] = expected_signature
        teacher["max_position_embeddings"] = config.max_position_embeddings
        teacher["model_type"] = config.model_type
    return origin


def score_trajectory(model, record, chunk_size, device, compute_eu):
    """KV-cache streaming. Logit at P-1+t predicts response token t, including EOS."""
    import numpy as np
    import torch

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    prompt_size = len(record["prompt_token_ids"])
    inputs = record["prompt_token_ids"] + record["response_token_ids"][:-1]
    cache, eu_chunks, value_chunks, id_chunks, denominator_chunks = None, [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(inputs), chunk_size):
            end = min(start + chunk_size, len(inputs))
            batch = torch.tensor([inputs[start:end]], dtype=torch.long, device=device)
            output = model(input_ids=batch, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            first = max(prompt_size - 1 - start, 0)
            if first < end - start:
                raw_logits = output.logits[0, first:]
                # No softmax or temperature. Select from the full vocabulary once;
                # the shared utility then only reduces these 16 selected values.
                top16, ids = torch.topk(raw_logits, k=16, dim=-1)
                values = top16.float()
                eu = compute_eu(values, top_k=16)
                denominator = (values + 1.0).sum(-1)
                eu_chunks.append(eu.cpu().numpy())
                value_chunks.append(values.cpu().numpy())
                id_chunks.append(ids.cpu().numpy())
                denominator_chunks.append(denominator.cpu().numpy())
                del raw_logits, values, top16, ids, eu, denominator
            del output, batch
    result = {
        "eu": np.concatenate(eu_chunks),
        "top16_logits": np.concatenate(value_chunks),
        "top16_ids": np.concatenate(id_chunks),
        "denominator": np.concatenate(denominator_chunks),
    }
    n = len(record["response_token_ids"])
    if result["eu"].shape != (n,) or result["top16_logits"].shape != (n, 16):
        raise ValueError(f"EU/token alignment failed: {record['uid']}")
    return result


def save_score(path, record, arrays):
    import numpy as np

    with Path(path).open("xb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            prefix_sha256=np.asarray(record["prefix_sha256"]),
            response_token_ids=np.asarray(record["response_token_ids"], dtype=np.int64),
        )


def worker(config_path, name):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    root = Path(config_path).parent
    config = read_json(config_path)
    teacher = next(t for t in config["teachers"] if t["name"] == name)
    if model_files(teacher["path"]) != teacher["files"]:
        raise ValueError(f"Scoring model files changed after preflight: {name}")
    if file_hash(EU_MODULE) != config["eu_source_sha256"]:
        raise ValueError("Shared EU code changed after preflight")
    if file_hash(root / "trajectories.json") != config["trajectory_snapshot_sha256"]:
        raise ValueError("Saved trajectory snapshot changed")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("This scorer requires exactly one visible allocated GPU")
    compute_eu = load_eu_function()
    model = (
        AutoModelForCausalLM.from_pretrained(
            teacher["path"],
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    directory = root / "teachers" / name
    directory.mkdir(parents=True, exist_ok=False)
    records = read_json(root / "trajectories.json")
    for index, record in enumerate(records):
        arrays = score_trajectory(
            model, record, config["score_chunk_size"], "cuda:0", compute_eu
        )
        torch.cuda.synchronize()
        save_score(directory / f"trajectory-{index:06d}.npz", record, arrays)
        valid = np.asarray(record["response_mask"], dtype=bool)
        print(
            f"{name} [{index + 1}/{len(records)}] {record['uid']}: "
            f"{len(arrays['eu'])} tokens, nonfinite_valid={int((~np.isfinite(arrays['eu'][valid])).sum())}",
            flush=True,
        )
        if index == 0:
            t = next((i for i, is_valid in enumerate(valid) if is_valid), 0)
            write_json(
                directory / "manual_check.json",
                {
                    "uid": record["uid"],
                    "response_position": t,
                    "response_token_id": record["response_token_ids"][t],
                    "predictor_position": len(record["prompt_token_ids"]) - 1 + t,
                    "top16_ids": arrays["top16_ids"][t].tolist(),
                    "top16_raw_logits": arrays["top16_logits"][t].tolist(),
                    "denominator": float(arrays["denominator"][t]),
                    "eu": float(arrays["eu"][t]),
                },
            )
    write_json(
        directory / "COMPLETED.json", {"responses": len(records), "teacher": name}
    )


def statistics(values):
    import numpy as np

    raw = np.asarray(values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    result = {
        "tokens": len(raw),
        "nonfinite_count": int(len(raw) - len(finite)),
        "nonpositive_count": int((finite <= 0).sum()),
    }
    result.update(
        {key: None for key in ("mean", "std", "min", "p10", "p50", "p90", "max")}
    )
    if len(finite):
        result.update(
            mean=float(finite.mean()),
            std=float(finite.std()),
            min=float(finite.min()),
            max=float(finite.max()),
            **dict(
                zip(
                    ("p10", "p50", "p90"),
                    np.quantile(finite, [0.1, 0.5, 0.9]).tolist(),
                    strict=True,
                )
            ),
        )
    return result


def summarize(root, tokenizer):
    """Join scorers on saved trajectory identity/token IDs, never array length alone."""
    import numpy as np

    root = Path(root)
    config = read_json(root / "config.json")
    records = read_json(root / "trajectories.json")
    names = [t["name"] for t in config["teachers"]]
    for name in names:
        if not (root / "teachers" / name / "COMPLETED.json").is_file():
            raise ValueError(f"Scoring incomplete: {name}")
    joined = root / "tokens"
    joined.mkdir(exist_ok=False)
    summary_rows, corpus, token_counts = [], [], 0
    with gzip.open(
        root / "tokens.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "trajectory_index",
            "task",
            "question_uid",
            "trajectory_uid",
            "response_index",
            "token_position",
            "token_id",
            "token_string",
            "valid",
        ] + [f"eu_{name}" for name in names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            columns = {
                k: [] for k in ("eu", "denominator", "top16_logits", "top16_ids")
            }
            n = len(record["response_token_ids"])
            for name in names:
                path = root / "teachers" / name / f"trajectory-{index:06d}.npz"
                with np.load(path, allow_pickle=False) as scores:
                    if str(scores["prefix_sha256"].item()) != record[
                        "prefix_sha256"
                    ] or not np.array_equal(
                        scores["response_token_ids"], record["response_token_ids"]
                    ):
                        raise ValueError(
                            f"Mismatched teacher prefix/token identity: {path}"
                        )
                    for key, parts in columns.items():
                        expected = (n, 16) if key.startswith("top16_") else (n,)
                        if scores[key].shape != expected:
                            raise ValueError(
                                f"Mismatched scorer output shape: {path} {key}"
                            )
                        parts.append(scores[key].copy())
            arrays = {key: np.stack(values, axis=1) for key, values in columns.items()}
            valid = np.asarray(record["response_mask"], dtype=bool)
            save_score(
                joined / f"trajectory-{index:06d}.npz",
                record,
                {**arrays, "teacher_names": np.asarray(names), "response_mask": valid},
            )
            corpus.append(arrays["eu"][valid])
            token_strings = tokenizer.convert_ids_to_tokens(
                record["response_token_ids"]
            )
            for t, token_id in enumerate(record["response_token_ids"]):
                writer.writerow(
                    {
                        "trajectory_index": index,
                        "task": record["task"],
                        "question_uid": record["question_uid"],
                        "trajectory_uid": record["uid"],
                        "response_index": record["response_index"],
                        "token_position": t,
                        "token_id": token_id,
                        "token_string": token_strings[t],
                        "valid": int(valid[t]),
                        **{
                            f"eu_{name}": float(arrays["eu"][t, j])
                            for j, name in enumerate(names)
                        },
                    }
                )
            for j, name in enumerate(names):
                summary_rows.append(
                    {
                        "task": record["task"],
                        "question_uid": record["question_uid"],
                        "trajectory_uid": record["uid"],
                        "response_index": record["response_index"],
                        "teacher": name,
                        **statistics(arrays["eu"][valid, j]),
                    }
                )
            token_counts += n
    values = np.concatenate(corpus, axis=0)
    global_stats = [
        {"teacher": name, **statistics(values[:, j])} for j, name in enumerate(names)
    ]
    pairwise = []
    for a, b in itertools.combinations(range(len(names)), 2):
        pairs = values[np.isfinite(values[:, a]) & np.isfinite(values[:, b])][
            :, [a, b]
        ].astype(np.float64)
        corr = None
        if len(pairs) > 1 and np.std(pairs[:, 0]) > 0 and np.std(pairs[:, 1]) > 0:
            corr = float(np.corrcoef(pairs.T)[0, 1])
        pairwise.append(
            {
                "teacher_a": names[a],
                "teacher_b": names[b],
                "finite_pairs": len(pairs),
                "pearson": corr,
                "mean_abs_difference": (
                    float(np.abs(pairs[:, 0] - pairs[:, 1]).mean())
                    if len(pairs)
                    else None
                ),
            }
        )
    write_json(root / "per_trajectory_summary.json", summary_rows)
    write_json(
        root / "summary.json",
        {
            "teachers": global_stats,
            "pairwise": pairwise,
            "responses": len(records),
            "tokens": token_counts,
            "questions": len({(r["task"], r["question_uid"]) for r in records}),
        },
    )
    lines = [
        "# Raw EU on identical saved student prefixes",
        "",
        "No normalization or temperature scaling. Finite valid-response-token statistics; std is population std.",
        "Different raw logit scales can give different EU scales; correlation alone does not imply calibration.",
        "",
        "| Scoring model | Valid tokens | Mean | Std | Min | Median | p90 | Max | Nonfinite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in global_stats:
        lines.append(
            "| "
            + " | ".join(
                str(row[k])
                for k in (
                    "teacher",
                    "tokens",
                    "mean",
                    "std",
                    "min",
                    "p50",
                    "p90",
                    "max",
                    "nonfinite_count",
                )
            )
            + " |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        root / "COMPLETED.json",
        {
            "complete": True,
            "teachers": names,
            "responses": len(records),
            "tokens": token_counts,
        },
    )


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        worker(Path(sys.argv[2]), sys.argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--student-model",
        type=Path,
        help="Score with one student model instead of the three-teacher mode",
    )
    for scale in ("4b", "8b", "14b"):
        parser.add_argument(f"--teacher-{scale}", type=Path)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Original student tokenizer; only needed without saved overlap config/fingerprint",
    )
    parser.add_argument("--score-chunk-size", type=int, default=128)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU preflight only, no weights loaded or files written",
    )
    args = parser.parse_args()
    if args.score_chunk_size < 1:
        parser.error("score-chunk-size must be positive")
    root = args.output_root.resolve()
    if root.exists():
        parser.error(
            "Output root exists; choose a new directory. No overwrite or implicit resume."
        )
    source = args.trajectories.resolve()
    if source.is_dir():
        source = source / "trajectories.json"
    source_hash = file_hash(source)
    source, records = load_trajectories(source)
    if file_hash(source) != source_hash:
        raise ValueError("Source trajectory file changed while reading")
    teacher_paths = [args.teacher_4b, args.teacher_8b, args.teacher_14b]
    if args.student_model:
        if any(teacher_paths):
            parser.error("--student-model cannot be combined with --teacher-4b/8b/14b")
        teachers = [{"name": "Student", "path": str(args.student_model.resolve())}]
        scoring_mode = "student"
    else:
        if not all(teacher_paths):
            parser.error("set --student-model, or provide all of --teacher-4b/8b/14b")
        teachers = [
            {"name": name, "path": str(path.resolve())}
            for name, path in zip(TEACHER_NAMES, teacher_paths, strict=True)
        ]
        scoring_mode = "teachers"
    origin = preflight(source, records, teachers, args.tokenizer_path)
    config = {
        "schema_version": 1,
        "top_k": 16,
        "scoring_temperature": None,
        "formula": "16 / sum(top16_raw_logits + 1)",
        "dtype": "bfloat16",
        "reduction_dtype": "float32",
        "attn_implementation": "sdpa",
        "score_chunk_size": args.score_chunk_size,
        "source_trajectories": str(source),
        "source_sha256": source_hash,
        "source_metadata": origin,
        "scoring_mode": scoring_mode,
        "teachers": teachers,
        "eu_source_sha256": file_hash(EU_MODULE),
        "scorer_sha256": file_hash(__file__),
        "responses": len(records),
        "tokens": sum(len(r["response_token_ids"]) for r in records),
        "questions": len({(r["task"], r["question_uid"]) for r in records}),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy")
        },
    }
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if file_hash(source) != source_hash:
        raise ValueError("Source trajectory file changed during preflight")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "trajectories.json", records)
    config["trajectory_snapshot_sha256"] = file_hash(root / "trajectories.json")
    write_json(root / "config.json", config)
    try:
        for name in (teacher["name"] for teacher in teachers):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "_worker",
                    str(root / "config.json"),
                    name,
                ],
                check=True,
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            teachers[0]["path"], local_files_only=True
        )
        summarize(root, tokenizer)
    except BaseException as error:
        write_json(
            root / "FAILED.json",
            {"error": str(error), "partial_scores_preserved": True},
        )
        raise
    print(f"Completed: {root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
