#!/usr/bin/env python3
"""Create a comparison CSV from per-model ``grading_results.json`` files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = (
    "mean_score",
    "best_score",
    "distinct_4gram",
    "avg_output_length",
    "format_error_rollouts",
    "solve_none",
    "solve_all",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline grading outputs into CSV.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    output_csv = args.output_csv or args.output_root / "baseline_summary.csv"
    rows = []
    grouped = defaultdict(list)
    for results_path in sorted(args.output_root.glob("*/grading_results.json")):
        model_name = results_path.parent.name
        with results_path.open(encoding="utf-8") as handle:
            for result in json.load(handle):
                row = {"model": model_name, "benchmark": result["hyperparameters"]["task_name"]}
                parameter_keys = ("n", "temperature", "top_p", "max_tokens")
                row.update({key: result["hyperparameters"].get(key, "") for key in parameter_keys})
                row.update({metric: result.get(metric, "") for metric in METRICS})
                rows.append(row)
                grouped[model_name].append(row)

    for model_name, model_rows in grouped.items():
        macro = {"model": model_name, "benchmark": "ALL_MACRO"}
        macro.update({key: model_rows[0][key] for key in ("n", "temperature", "top_p", "max_tokens")})
        for metric in METRICS:
            values = [row[metric] for row in model_rows if isinstance(row[metric], (int, float))]
            macro[metric] = sum(values) / len(values) if values else ""
        rows.append(macro)

    if not rows:
        raise FileNotFoundError(f"No grading_results.json files found below {args.output_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("model", "benchmark", "n", "temperature", "top_p", "max_tokens", *METRICS)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
