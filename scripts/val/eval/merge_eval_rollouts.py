#!/usr/bin/env python3
"""Merge independently generated rollout shards into one evaluation JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_shard(path: Path) -> tuple[list[dict[str, object]], int]:
    records = []
    counts: Counter[int] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if "example_id" not in record:
                raise ValueError(f"{path}:{line_number} is missing example_id.")
            example_id = int(record["example_id"])
            counts[example_id] += 1
            records.append(record)

    if not records:
        raise ValueError(f"{path} is empty.")
    rollout_counts = set(counts.values())
    if len(rollout_counts) != 1:
        raise ValueError(f"{path} has unequal rollout counts across examples: {sorted(rollout_counts)}.")
    return records, rollout_counts.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge rollout shards while preserving all responses.")
    parser.add_argument("--input", action="append", type=Path, required=True, help="One source JSONL per worker.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-total-samples", type=int, required=True)
    args = parser.parse_args()

    if args.expected_total_samples <= 0:
        raise ValueError("--expected-total-samples must be positive.")
    if args.output in args.input:
        raise ValueError("The output path must differ from every input path.")

    merged_records = []
    expected_example_ids: set[int] | None = None
    seed_offset = 0
    for input_path in args.input:
        records, samples_per_example = load_shard(input_path)
        example_ids = {int(record["example_id"]) for record in records}
        if expected_example_ids is None:
            expected_example_ids = example_ids
        elif example_ids != expected_example_ids:
            raise ValueError(f"{input_path} has different example IDs from the preceding shard.")

        for record in records:
            merged_record = dict(record)
            merged_record["seed"] = int(merged_record.get("seed", 0)) + seed_offset
            merged_records.append(merged_record)
        seed_offset += samples_per_example

    if seed_offset != args.expected_total_samples:
        raise ValueError(
            f"Merged {seed_offset} samples per example, expected {args.expected_total_samples}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in merged_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(args.output)
    print(
        f"Merged {len(args.input)} shards into {args.output}: "
        f"{len(expected_example_ids or set())} examples x {seed_offset} samples.",
    )


if __name__ == "__main__":
    main()
