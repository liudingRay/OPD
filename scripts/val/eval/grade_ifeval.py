#!/usr/bin/env python3
"""Score IFEval generations with the pinned strict and loose rule checkers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = SCRIPT_DIR.parent / "data" / "IFEval" / "ifeval_input_data.jsonl"
SCORER_REVISION = "EleutherAI/lm-evaluation-harness@d6de81643928d653435c431bae19945d41d32520"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object in {path} at line {line_number}.")
            records.append(record)
    if not records:
        raise ValueError(f"No JSON records found in {path}.")
    return records


def load_dataset(path: Path) -> list[dict[str, Any]]:
    required_fields = {"key", "prompt", "instruction_id_list", "kwargs"}
    rows = read_jsonl(path)
    keys = set()
    prompts = set()
    for index, row in enumerate(rows):
        missing = required_fields.difference(row)
        if missing:
            raise ValueError(f"Dataset row {index} is missing fields: {sorted(missing)}")
        key = row["key"]
        prompt = row["prompt"]
        instruction_ids = row["instruction_id_list"]
        kwargs = row["kwargs"]
        if not isinstance(key, int):
            raise TypeError(f"Dataset row {index} has a non-integer key: {key!r}")
        if not isinstance(prompt, str) or not prompt:
            raise TypeError(f"Dataset row {index} has an empty or non-string prompt.")
        if not isinstance(instruction_ids, list) or not instruction_ids:
            raise TypeError(f"Dataset row {index} has no instruction IDs.")
        if not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
            raise ValueError(f"Dataset row {index} has mismatched instruction_id_list and kwargs lengths.")
        if key in keys:
            raise ValueError(f"Duplicate dataset key: {key}")
        if prompt in prompts:
            raise ValueError(f"Duplicate dataset prompt at row {index}.")
        keys.add(key)
        prompts.add(prompt)
    return rows


def _load_scorer() -> tuple[type, Callable, Callable]:
    try:
        from langdetect import DetectorFactory

        DetectorFactory.seed = 0
        from ifeval_scorer import (
            InputExample,
            test_instruction_following_loose,
            test_instruction_following_strict,
        )
        from ifeval_scorer.instructions_util import require_nltk_resources
    except (ImportError, AssertionError) as exc:
        raise RuntimeError(
            "IFEval scorer dependencies are unavailable. Install scripts/val/eval/requirements-ifeval.txt "
            "in the evaluation environment."
        ) from exc

    require_nltk_resources()
    return InputExample, test_instruction_following_strict, test_instruction_following_loose


def validate_scorer_environment() -> None:
    """Fail before GPU inference if the CPU scorer cannot run locally."""
    _load_scorer()


def _normalize_generation_records(
    records: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    *,
    allow_partial: bool,
) -> list[dict[str, Any]]:
    by_key = {row["key"]: row for row in dataset_rows}
    by_prompt = {row["prompt"]: row for row in dataset_rows}
    normalized = []
    seen_pairs = set()

    for index, record in enumerate(records):
        if "prompt" not in record or "response" not in record:
            raise ValueError(f"Generation row {index} must contain prompt and response.")
        prompt = record["prompt"]
        response = record["response"]
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError(f"Generation row {index} has a non-string prompt or response.")

        if "key" in record:
            key = record["key"]
            if key not in by_key:
                raise ValueError(f"Generation row {index} references unknown key {key!r}.")
            dataset_row = by_key[key]
            if prompt != dataset_row["prompt"]:
                raise ValueError(f"Generation row {index} prompt does not match dataset key {key} exactly.")
        else:
            if prompt not in by_prompt:
                raise ValueError(f"Generation row {index} prompt does not occur in the pinned dataset.")
            dataset_row = by_prompt[prompt]
            key = dataset_row["key"]

        sample_id = record.get("sample_id", 0)
        if not isinstance(sample_id, int) or sample_id < 0:
            raise ValueError(f"Generation row {index} has invalid sample_id {sample_id!r}.")
        pair = (sample_id, key)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate generation for sample_id={sample_id}, key={key}.")
        seen_pairs.add(pair)

        normalized.append(
            {
                **record,
                "key": key,
                "sample_id": sample_id,
                "dataset_row": dataset_row,
            }
        )

    keys_by_sample: dict[int, set[int]] = defaultdict(set)
    for record in normalized:
        keys_by_sample[record["sample_id"]].add(record["key"])
    reference_keys = next(iter(keys_by_sample.values()))
    for sample_id, keys in keys_by_sample.items():
        if keys != reference_keys:
            raise ValueError(
                f"sample_id={sample_id} covers {len(keys)} keys, but every sample must cover the same key set."
            )
    if not allow_partial and reference_keys != set(by_key):
        missing = sorted(set(by_key).difference(reference_keys))
        raise ValueError(
            f"Generation file covers {len(reference_keys)}/{len(by_key)} prompts. "
            f"Use --allow-partial only for smoke tests. First missing keys: {missing[:5]}"
        )
    return normalized


def _metric_block(details: list[dict[str, Any]]) -> dict[str, float]:
    instruction_total = sum(len(item["instruction_id_list"]) for item in details)
    return {
        "prompt_level_strict_acc": sum(item["strict_follow_all"] for item in details) / len(details),
        "prompt_level_loose_acc": sum(item["loose_follow_all"] for item in details) / len(details),
        "instruction_level_strict_acc": (
            sum(sum(item["strict_follow_instruction_list"]) for item in details) / instruction_total
        ),
        "instruction_level_loose_acc": (
            sum(sum(item["loose_follow_instruction_list"]) for item in details) / instruction_total
        ),
    }


def score_records(
    dataset_rows: list[dict[str, Any]],
    generation_records: list[dict[str, Any]],
    *,
    allow_partial: bool = False,
    scorer: tuple[type, Callable, Callable] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = _normalize_generation_records(generation_records, dataset_rows, allow_partial=allow_partial)
    InputExample, test_strict, test_loose = scorer or _load_scorer()
    details = []

    for record in normalized:
        row = record["dataset_row"]
        example = InputExample(
            key=row["key"],
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=row["kwargs"],
        )
        strict = test_strict(example, record["response"])
        loose = test_loose(example, record["response"])
        if len(strict.follow_instruction_list) != len(loose.follow_instruction_list):
            raise AssertionError(f"Strict/loose instruction count mismatch for key {row['key']}.")
        if any(
            strict_value and not loose_value
            for strict_value, loose_value in zip(
                strict.follow_instruction_list,
                loose.follow_instruction_list,
            )
        ):
            raise AssertionError(f"Loose scoring regressed below strict scoring for key {row['key']}.")

        details.append(
            {
                "example_id": record.get("example_id"),
                "key": row["key"],
                "sample_id": record["sample_id"],
                "prompt": row["prompt"],
                "response": record["response"],
                "instruction_id_list": row["instruction_id_list"],
                "strict_follow_all": strict.follow_all_instructions,
                "strict_follow_instruction_list": strict.follow_instruction_list,
                "loose_follow_all": loose.follow_all_instructions,
                "loose_follow_instruction_list": loose.follow_instruction_list,
                "finish_reason": record.get("finish_reason"),
                "generated_tokens": record.get("generated_tokens"),
            }
        )

    metrics = _metric_block(details)
    sample_ids = sorted({item["sample_id"] for item in details})
    per_sample = []
    for sample_id in sample_ids:
        sample_details = [item for item in details if item["sample_id"] == sample_id]
        per_sample.append({"sample_id": sample_id, **_metric_block(sample_details)})

    group_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "strict_correct": 0, "loose_correct": 0}
    )
    for item in details:
        for instruction_id, strict_value, loose_value in zip(
            item["instruction_id_list"],
            item["strict_follow_instruction_list"],
            item["loose_follow_instruction_list"],
        ):
            group = instruction_id.split(":", 1)[0]
            group_counts[group]["total"] += 1
            group_counts[group]["strict_correct"] += int(strict_value)
            group_counts[group]["loose_correct"] += int(loose_value)

    by_instruction_group = {
        group: {
            **counts,
            "strict_acc": counts["strict_correct"] / counts["total"],
            "loose_acc": counts["loose_correct"] / counts["total"],
        }
        for group, counts in sorted(group_counts.items())
    }
    covered_keys = {item["key"] for item in details}
    summary = {
        "benchmark": "IFEval",
        "main_metrics": {
            "prompt_level_strict_acc": metrics["prompt_level_strict_acc"],
            "prompt_level_loose_acc": metrics["prompt_level_loose_acc"],
        },
        "diagnostic_metrics": {
            "instruction_level_strict_acc": metrics["instruction_level_strict_acc"],
            "instruction_level_loose_acc": metrics["instruction_level_loose_acc"],
        },
        "counts": {
            "dataset_prompts": len(dataset_rows),
            "evaluated_prompts_per_sample": len(covered_keys),
            "samples_per_prompt": len(sample_ids),
            "evaluated_responses": len(details),
            "evaluated_instruction_instances": sum(len(item["instruction_id_list"]) for item in details),
            "truncated_responses": sum(item["finish_reason"] == "length" for item in details),
        },
        "per_sample": per_sample,
        "by_instruction_group": by_instruction_group,
        "scorer_revision": SCORER_REVISION,
        "langdetect_seed": 0,
        "partial_evaluation": len(covered_keys) != len(dataset_rows),
    }
    return summary, details


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def score_file(
    response_file: Path,
    *,
    data_file: Path = DEFAULT_DATA_FILE,
    summary_file: Path | None = None,
    details_file: Path | None = None,
    allow_partial: bool = False,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    response_file = response_file.resolve()
    summary_file = summary_file or response_file.with_suffix(".summary.json")
    details_file = details_file or response_file.with_suffix(".results.jsonl")
    for output_path in (summary_file, details_file):
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace existing score output: {output_path}")

    dataset_rows = load_dataset(data_file)
    generation_records = read_jsonl(response_file)
    summary, details = score_records(dataset_rows, generation_records, allow_partial=allow_partial)
    _atomic_write_json(summary_file, summary)
    _atomic_write_jsonl(details_file, details)
    return summary_file, details_file, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score IFEval generations with strict and loose prompt metrics.")
    parser.add_argument("--response-file", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--details-file", type=Path, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a smoke-test subset instead of all prompts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_file, details_file, summary = score_file(
        args.response_file,
        data_file=args.data_file,
        summary_file=args.summary_file,
        details_file=args.details_file,
        allow_partial=args.allow_partial,
        overwrite=args.overwrite,
    )
    main_metrics = summary["main_metrics"]
    print(f"prompt_level_strict_acc={main_metrics['prompt_level_strict_acc']:.6f}")
    print(f"prompt_level_loose_acc={main_metrics['prompt_level_loose_acc']:.6f}")
    print(f"Summary: {summary_file}")
    print(f"Per-response details: {details_file}")


if __name__ == "__main__":
    main()
