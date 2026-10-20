#!/usr/bin/env python3
"""Grade GPQA-D and MMLU-Pro generations by their boxed option letter.

The math grader intentionally requires the complete contents of ``\\boxed{}``
to match the gold answer. Multiple-choice models often include both the option
letter and its text, for example ``\\boxed{\\text{C. answer text}}``. This
scorer accepts that form while still requiring an unambiguous leading A-D in
the final boxed expression.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Callable


SUPPORTED_TASKS = frozenset({"gpqa", "mmlu-pro"})
FILENAME_PATTERN = re.compile(
    r"^(?P<task>gpqa|mmlu-pro)_t(?P<temperature>[\d.]+)_p(?P<top_p>[\d.]+)"
    r"_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)$"
)
LEADING_FORMAT_COMMAND = re.compile(r"^\\(?:text|textrm|textbf|mathrm|mathbf|operatorname)\s*")
OPTION_PATTERN = re.compile(r"^(?:option\s*)?([A-D])(?=$|[\s.\):=,;\\}\]\-])", re.IGNORECASE)


def extract_last_boxed_content(response: str) -> str | None:
    """Return the balanced contents of the final ``\\boxed{...}``."""
    command_start = response.rfind("\\boxed")
    if command_start < 0:
        return None

    cursor = command_start + len("\\boxed")
    while cursor < len(response) and response[cursor].isspace():
        cursor += 1
    if cursor >= len(response) or response[cursor] != "{":
        return None

    depth = 0
    for index in range(cursor, len(response)):
        character = response[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return response[cursor + 1 : index]
    return None


def extract_boxed_option(response: str) -> str | None:
    """Extract a leading A-D option from the final boxed expression."""
    content = extract_last_boxed_content(response)
    if content is None:
        return None

    content = content.strip().lstrip("$").lstrip()
    while True:
        previous = content
        content = content.lstrip("{ ~$")
        content = LEADING_FORMAT_COMMAND.sub("", content, count=1)
        if content == previous:
            break

    match = OPTION_PATTERN.match(content)
    return match.group(1).upper() if match else None


def get_diverse_score(sequences: list[str], n: int = 4) -> float:
    distinct_ngrams: set[tuple[str, ...]] = set()
    total_ngrams = 0
    for sequence in sequences:
        tokens = sequence.split()
        for index in range(len(tokens) - n + 1):
            distinct_ngrams.add(tuple(tokens[index : index + n]))
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams else 0.0


def parse_hyperparameters(file_path: Path) -> dict[str, str]:
    match = FILENAME_PATTERN.fullmatch(file_path.stem)
    if not match:
        raise ValueError(f"Unrecognized multiple-choice result filename: {file_path.name}")
    return match.groupdict()


def load_grouped_responses(file_path: Path) -> list[dict[str, object]]:
    grouped: OrderedDict[object, dict[str, object]] = OrderedDict()
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            try:
                example_id = record["example_id"]
                ground_truth = str(record["answer"]).strip().upper()
                response = str(record["response"])
            except KeyError as exc:
                raise ValueError(f"{file_path}:{line_number} is missing {exc.args[0]!r}") from exc
            if ground_truth not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"{file_path}:{line_number} has unsupported ground truth {ground_truth!r}; expected A-D"
                )

            item = grouped.setdefault(example_id, {"ground_truth": ground_truth, "responses": []})
            if item["ground_truth"] != ground_truth:
                raise ValueError(f"{file_path}: inconsistent ground truth for example_id={example_id!r}")
            item["responses"].append(response)

    if not grouped:
        raise ValueError(f"No generations found in {file_path}")
    return list(grouped.values())


def grade_file(file_path: Path, get_length: Callable[[str], int] = len) -> dict[str, object]:
    hyperparameters = parse_hyperparameters(file_path)
    task_name = hyperparameters.pop("task")
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported multiple-choice task: {task_name}")
    hyperparameters["task_name"] = task_name

    expected_samples = int(hyperparameters["n"])
    grouped = load_grouped_responses(file_path)
    question_scores: list[float] = []
    question_best: list[bool] = []
    diversity_scores: list[float] = []
    response_lengths: list[int] = []
    format_errors = 0
    parsed_rollouts = 0

    for item in grouped:
        responses = item["responses"]
        if len(responses) != expected_samples:
            raise ValueError(
                f"{file_path}: expected {expected_samples} responses per question, got {len(responses)}"
            )
        ground_truth = item["ground_truth"]
        scores = []
        for response in responses:
            option = extract_boxed_option(response)
            if option is None:
                format_errors += 1
            else:
                parsed_rollouts += 1
            scores.append(option == ground_truth)
            response_lengths.append(get_length(response))

        question_scores.append(sum(scores) / expected_samples)
        question_best.append(any(scores))
        diversity_scores.append(get_diverse_score(responses))

    result: dict[str, object] = {
        "hyperparameters": hyperparameters,
        "mean_score": sum(question_scores) / len(question_scores),
        "distinct_4gram": sum(diversity_scores) / len(diversity_scores),
        "best_score": sum(question_best) / len(question_best),
        "solve_none": sum(score == 0 for score in question_scores),
        "solve_all": sum(score == 1 for score in question_scores),
        "avg_output_length": sum(response_lengths) / len(response_lengths),
        "format_error_rollouts": format_errors,
        "parsed_rollouts": parsed_rollouts,
        "scorer": "boxed_multiple_choice_v1",
    }
    return result


def build_length_function(tokenizer_path: str | None) -> Callable[[str], int]:
    if not tokenizer_path:
        return len
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception as exc:
        print(f"Warning: could not load length tokenizer {tokenizer_path}: {exc}")
        print("Falling back to character counts for avg_output_length.")
        return len
    return lambda response: len(tokenizer.encode(response))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade GPQA-D and MMLU-Pro boxed multiple-choice answers.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--length-tokenizer", default=None)
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory does not exist: {args.eval_dir}")
    output_file = args.output_file or args.eval_dir / "grading_results.json"
    get_length = build_length_function(args.length_tokenizer)

    results = []
    for file_path in sorted(args.eval_dir.glob("*.jsonl")):
        match = FILENAME_PATTERN.fullmatch(file_path.stem)
        if match and match.group("task") in SUPPORTED_TASKS:
            print(f"Processing multiple-choice file: {file_path}")
            results.append(grade_file(file_path, get_length=get_length))
    if not results:
        raise FileNotFoundError(f"No GPQA-D or MMLU-Pro JSONL files found in {args.eval_dir}")

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(f"Grading results saved to {output_file}")


if __name__ == "__main__":
    main()
