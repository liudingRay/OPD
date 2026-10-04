#!/usr/bin/env python3
"""Compute three-teacher token EU on text-only student evaluation rollouts.

The standard baseline evaluator saves prompt/response text but not vLLM token
IDs. This wrapper reconstructs the visible token sequence with the evaluated
student's tokenizer, verifies a lossless text round trip, delegates raw-logit
EU scoring to ``eval_token_eu.py``, and creates self-contained HTML/SVG reports.
It never generates a new response and never overwrites an output directory.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import eval_token_eu as token_eu
from eval_fixed_overlap import file_hash, read_json, tokenizer_signature, write_json


COLORS = ("#2563eb", "#dc2626", "#059669")
SEMANTIC_LATEX_SCAFFOLDING = {
    "begin",
    "big",
    "bigg",
    "boxed",
    "displaystyle",
    "end",
    "frac",
    "left",
    "mathbf",
    "mathrm",
    "right",
    "text",
}
SEMANTIC_MATH_SYMBOLS = frozenset("+-−×÷*/=<>≤≥±∑√^")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            row["_source_line"] = line_number
            records.append(row)
    if not records:
        raise ValueError(f"No rollout records in {path}")
    return records


def resolve_source(evaluation_root: Path, task: str, model_label: str | None, explicit: Path | None):
    evaluation_root = evaluation_root.resolve()
    config_path = evaluation_root / "evaluation_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing evaluation_config.json: {config_path}")
    configs = read_json(config_path)
    if not isinstance(configs, list) or not configs:
        raise ValueError("evaluation_config.json must contain a nonempty list")
    if model_label:
        matches = [item for item in configs if item.get("label") == model_label]
    elif len(configs) == 1:
        matches = configs
    else:
        raise ValueError("Multiple evaluated models found; set --model-label")
    if len(matches) != 1:
        raise ValueError(f"Expected one evaluation config for model label {model_label!r}")
    config = matches[0]
    label = config.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("Evaluation config has no model label")
    if task not in config.get("tasks", []):
        raise ValueError(f"Task {task} is not listed in the evaluation config")
    if explicit:
        rollout_path = explicit.resolve()
    else:
        candidates = sorted((evaluation_root / label).glob(f"{task.lower()}_*.jsonl"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one {task} JSONL under {evaluation_root / label}, found {candidates}"
            )
        rollout_path = candidates[0]
    if not rollout_path.is_file():
        raise ValueError(f"Rollout JSONL does not exist: {rollout_path}")
    return config_path, config, rollout_path


def _decode(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_trajectories(rows, tokenizer, config, task: str, max_trajectories: int = 0):
    if max_trajectories < 0:
        raise ValueError("max_trajectories must be nonnegative")
    selected = rows[:max_trajectories] if max_trajectories else rows
    thinking = bool(config.get("enable_thinking", False))
    records, seen = [], set()
    for source_index, row in enumerate(selected):
        required = ("example_id", "question", "prompt", "answer", "seed", "response")
        if any(key not in row for key in required):
            raise ValueError(f"Source row {source_index} is missing one of {required}")
        if type(row["example_id"]) is not int or type(row["seed"]) is not int:
            raise ValueError("example_id and seed must be integers")
        if any(not isinstance(row[key], str) for key in ("question", "prompt", "answer", "response")):
            raise ValueError("question, prompt, answer and response must be strings")
        if not row["response"]:
            raise ValueError(f"Empty response at source row {source_index}")
        key = (row["example_id"], row["seed"])
        if key in seen:
            raise ValueError(f"Duplicate example/rollout pair: {key}")
        seen.add(key)
        messages = [{"role": "user", "content": row["prompt"]}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        rendered_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if list(prompt_ids) != list(rendered_ids):
            raise ValueError(f"Prompt template tokenization is not reproducible for source row {source_index}")
        response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        if not response_ids or _decode(tokenizer, response_ids) != row["response"]:
            raise ValueError(
                f"Response text is not a lossless tokenizer round trip at source row {source_index}"
            )
        uid = f"{task}:{row['example_id']}:response-{row['seed']:02d}"
        records.append(
            {
                "uid": uid,
                "question_uid": f"{task}:{row['example_id']}",
                "response_index": row["seed"],
                "task": task,
                "question": row["question"],
                "prompt": row["prompt"],
                "answer": row["answer"],
                "seed": row["seed"],
                "response": row["response"],
                "finish_reason": "not_recorded_by_eval_baselines",
                "prompt_token_ids": list(prompt_ids),
                "response_token_ids": list(response_ids),
                "response_mask": [1] * len(response_ids),
                "source_line": row.get("_source_line", source_index + 1),
                "token_reconstruction": "student_tokenizer_from_saved_visible_text",
            }
        )
    return records


def validate_models(records, student_path: Path, teachers: list[dict]):
    from transformers import AutoConfig, AutoTokenizer

    tokenizers = [AutoTokenizer.from_pretrained(student_path, local_files_only=True)]
    student_files = token_eu.model_files(student_path)
    student_config = AutoConfig.from_pretrained(student_path, local_files_only=True)
    if student_config.model_type != "qwen3":
        raise ValueError("The evaluated student must be a Qwen3 model")
    signature = tokenizer_signature(tokenizers[0])
    max_id = max(max(row[key]) for row in records for key in ("prompt_token_ids", "response_token_ids"))
    max_context = max(len(row["prompt_token_ids"]) + len(row["response_token_ids"]) - 1 for row in records)
    checked_teachers = []
    for teacher in teachers:
        path = Path(teacher["path"])
        files = token_eu.model_files(path)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if tokenizer_signature(tokenizer) != signature:
            raise ValueError(f"{teacher['name']}: tokenizer/token-ID mapping differs from the student")
        if config.model_type != "qwen3" or max_id >= config.vocab_size:
            raise ValueError(f"{teacher['name']}: incompatible model type or vocabulary")
        if max_context > config.max_position_embeddings:
            raise ValueError(f"{teacher['name']}: a reconstructed trajectory exceeds model context")
        checked_teachers.append(
            {
                **teacher,
                "files": files,
                "max_position_embeddings": config.max_position_embeddings,
                "tokenizer_sha256": signature,
            }
        )
    return {
        "student_files": student_files,
        "student_tokenizer_sha256": signature,
        "max_token_id": max_id,
        "max_context_tokens": max_context,
        "teachers": checked_teachers,
    }


def extended_statistics(values):
    import numpy as np

    raw = np.asarray(values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    result = {
        "tokens": int(raw.size),
        "finite_tokens": int(finite.size),
        "nonfinite_count": int(raw.size - finite.size),
        "nonpositive_count": int((finite <= 0).sum()),
    }
    for key in ("mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "p95", "max"):
        result[key] = None
    if finite.size:
        quantiles = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        result.update(
            mean=float(finite.mean()),
            std=float(finite.std()),
            min=float(finite.min()),
            max=float(finite.max()),
            **dict(zip(("p10", "p25", "p50", "p75", "p90", "p95"), quantiles.tolist(), strict=True)),
        )
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.6g}"


def _readable_token_piece(token_piece: str) -> str:
    return token_piece.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def _is_semantic_token(decoded_piece: str) -> bool:
    """Keep content-bearing pieces while dropping Markdown/LaTeX scaffolding."""
    piece = _readable_token_piece(decoded_piece).strip()
    if not piece:
        return False
    normalized = piece.strip("*_`#")
    if not normalized or normalized in {"$", "$$", "\\", "{", "}", "}{", "[", "]", "(", ")"}:
        return False
    command = normalized.lstrip("\\")
    if command in SEMANTIC_LATEX_SCAFFOLDING:
        return False
    if len(normalized) > 1 and set(normalized) <= set("-_*`#"):
        return False
    return any(character.isalnum() for character in normalized) or any(
        character in SEMANTIC_MATH_SYMBOLS for character in normalized
    )


def _semantic_unit(tokenizer, token_ids, token_pieces, position):
    """Expand one high-EU piece to its surrounding word or bounded math expression."""
    left_math = next((i for i in range(position - 1, max(-1, position - 40), -1) if "$" in token_pieces[i]), None)
    right_math = next(
        (i for i in range(position + 1, min(len(token_pieces), position + 40)) if "$" in token_pieces[i]), None
    )
    if left_math is not None and right_math is not None:
        expression = _decode(tokenizer, token_ids[left_math : right_math + 1].tolist()).strip()
        if len(expression) <= 180:
            return expression

    left = position
    while left > 0 and position - left < 8 and token_pieces[left] and not token_pieces[left][0].isspace():
        previous = token_pieces[left - 1]
        if not previous or (not previous[-1].isalnum() and previous[-1] not in "_-'"):
            break
        left -= 1
        if previous[0].isspace():
            break
    right = position + 1
    while right < len(token_pieces) and right - position < 8:
        piece = token_pieces[right]
        if not piece or piece[0].isspace() or (not piece[0].isalnum() and piece[0] not in "_-'"):
            break
        right += 1
    return _decode(tokenizer, token_ids[left:right].tolist()).strip()


def _teacher_value_cell(row, name):
    return f"{_number(row[f'eu_{name}'])} ({_number(row[f'percentile_{name}'])}%)"


def _map(value, low, high, start, width):
    if high <= low:
        return start + width / 2
    return start + width * (min(max(value, low), high) - low) / (high - low)


def distribution_svg(values_by_teacher, names):
    import numpy as np

    finite = [np.asarray(values)[np.isfinite(values)] for values in values_by_teacher]
    combined = np.concatenate([values for values in finite if values.size])
    low, high = np.quantile(combined, [0.01, 0.99]).tolist()
    if high <= low:
        high = low + 1.0
    bins = np.linspace(low, high, 61)
    width, height, left, top, plot_w, plot_h = 1100, 430, 80, 35, 970, 320
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="22" font-family="sans-serif" font-size="16">'
        "Teacher EU distribution (1st–99th percentile display range)</text>",
    ]
    histograms = []
    peak = 1.0
    for values in finite:
        clipped = np.clip(values, low, high)
        hist, _ = np.histogram(clipped, bins=bins)
        hist = hist / max(hist.sum(), 1)
        histograms.append(hist)
        peak = max(peak, float(hist.max()))
    for index, (name, hist) in enumerate(zip(names, histograms, strict=True)):
        points = []
        for bin_index, value in enumerate(hist):
            x = left + plot_w * (bin_index + 0.5) / len(hist)
            y = top + plot_h * (1 - value / peak)
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index]}" '
            'stroke-width="2" opacity="0.9"/>'
        )
        parts.append(
            f'<text x="{left + 210 * index}" y="{height - 20}" fill="{COLORS[index]}" '
            f'font-family="sans-serif" font-size="14">{html.escape(name)}</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111"/>',
            f'<text x="{left}" y="{top + plot_h + 24}" font-family="sans-serif" font-size="12">{low:.6g}</text>',
            f'<text x="{left + plot_w - 70}" y="{top + plot_h + 24}" '
            f'font-family="sans-serif" font-size="12">{high:.6g}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def trajectory_svg(eu, valid, names, highlighted):
    import numpy as np

    finite_values = eu[valid & np.isfinite(eu).all(axis=1)]
    if not finite_values.size:
        return '<svg xmlns="http://www.w3.org/2000/svg"><text>No finite EU values</text></svg>'
    low, high = np.quantile(finite_values, [0.01, 0.99]).tolist()
    if high <= low:
        high = low + 1.0
    n = eu.shape[0]
    width, height, left, top, plot_w, plot_h = 1250, 470, 85, 40, 1110, 340
    sample = np.unique(np.linspace(0, max(n - 1, 0), min(n, 1800), dtype=int))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-family="sans-serif" font-size="16">'
        f"Per-token teacher EU (display clipped to p01–p99: {low:.6g}–{high:.6g})</text>",
    ]
    for teacher_index, name in enumerate(names):
        points = []
        for position in sample:
            value = eu[position, teacher_index]
            if valid[position] and np.isfinite(value):
                x = _map(position, 0, max(n - 1, 1), left, plot_w)
                y = top + plot_h - _map(float(value), low, high, 0, plot_h)
                points.append(f"{x:.2f},{y:.2f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[teacher_index]}" '
            'stroke-width="1.4" opacity="0.82"/>'
        )
        for position in highlighted:
            value = eu[position, teacher_index]
            if valid[position] and np.isfinite(value):
                x = _map(position, 0, max(n - 1, 1), left, plot_w)
                y = top + plot_h - _map(float(value), low, high, 0, plot_h)
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{COLORS[teacher_index]}"/>')
        parts.append(
            f'<text x="{left + 250 * teacher_index}" y="{height - 22}" fill="{COLORS[teacher_index]}" '
            f'font-family="sans-serif" font-size="14">{html.escape(name)}</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111"/>',
            f'<text x="{left}" y="{top + plot_h + 24}" font-family="sans-serif" font-size="12">token 0</text>',
            f'<text x="{left + plot_w - 80}" y="{top + plot_h + 24}" '
            f'font-family="sans-serif" font-size="12">token {n - 1}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def _select_plot_indices(records, specs):
    if "all" in specs:
        return set(range(len(records)))
    selected = set()
    for spec in specs:
        if spec.startswith("index:"):
            index = int(spec.split(":", 1)[1])
            if not 0 <= index < len(records):
                raise ValueError(f"Plot trajectory index out of range: {index}")
            selected.add(index)
            continue
        try:
            example_id, response_index = (int(value) for value in spec.split(":", 1))
        except Exception as error:
            raise ValueError(f"Plot spec must be all, index:N, or EXAMPLE_ID:RESPONSE_INDEX: {spec}") from error
        matches = [
            index
            for index, row in enumerate(records)
            if row["question_uid"].endswith(f":{example_id}") and row["response_index"] == response_index
        ]
        if len(matches) != 1:
            raise ValueError(f"Plot spec {spec} matched {len(matches)} trajectories")
        selected.add(matches[0])
    return selected


def analyze_output(root: Path, tokenizer, plot_specs: list[str], top_n: int):
    import numpy as np

    if top_n < 1:
        raise ValueError("top_n must be positive")
    root = root.resolve()
    config = read_json(root / "config.json")
    records = read_json(root / "trajectories.json")
    names = [teacher["name"] for teacher in config["teachers"]]
    if len(names) != 3:
        raise ValueError("Visualization expects exactly three teachers")
    selected = _select_plot_indices(records, plot_specs)
    analysis = root / "analysis"
    trajectory_reports = analysis / "trajectories"
    analysis.mkdir(exist_ok=False)
    trajectory_reports.mkdir()
    global_values = [[] for _ in names]
    stats_rows, high_rows, semantic_rows, pages = [], [], [], []
    for index, record in enumerate(records):
        path = root / "tokens" / f"trajectory-{index:06d}.npz"
        with np.load(path, allow_pickle=False) as content:
            eu = content["eu"].astype(np.float64)
            valid = content["response_mask"].astype(bool)
            token_ids = content["response_token_ids"].astype(np.int64)
            stored_names = content["teacher_names"].tolist()
        if stored_names != names or eu.shape != (len(token_ids), len(names)) or valid.shape != (len(token_ids),):
            raise ValueError(f"Misaligned joined token file: {path}")
        token_strings = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        readable_pieces = [_readable_token_piece(piece) for piece in token_strings]
        thresholds, sorted_values, top_positions, highlighted = [], [], [], set()
        for teacher_index, name in enumerate(names):
            values = eu[:, teacher_index][valid]
            global_values[teacher_index].append(values)
            stats = extended_statistics(values)
            stats_rows.append(
                {
                    "trajectory_index": index,
                    "trajectory_uid": record["uid"],
                    "question_uid": record["question_uid"],
                    "response_index": record["response_index"],
                    "teacher": name,
                    **stats,
                }
            )
            finite_positions = np.where(valid & np.isfinite(eu[:, teacher_index]))[0]
            ordered = finite_positions[np.argsort(eu[finite_positions, teacher_index])[::-1]]
            selected_positions = set(ordered[:top_n].tolist())
            top_positions.append(selected_positions)
            highlighted.update(selected_positions)
            finite = np.sort(eu[finite_positions, teacher_index])
            sorted_values.append(finite)
            thresholds.append(float(np.quantile(finite, 0.9)) if finite.size else math.nan)
        percentile_rows = [
            [
                (
                    100.0 * np.searchsorted(sorted_values[j], eu[position, j], side="right") / len(sorted_values[j])
                    if len(sorted_values[j]) and valid[position] and np.isfinite(eu[position, j])
                    else None
                )
                for j in range(len(names))
            ]
            for position in range(len(token_ids))
        ]
        for position in sorted(highlighted):
            selected_by = [
                names[j]
                for j in range(len(names))
                if position in top_positions[j]
            ]
            percentiles = percentile_rows[position]
            high_flags = [
                bool(np.isfinite(eu[position, j]) and eu[position, j] >= thresholds[j])
                for j in range(len(names))
            ]
            context_ids = token_ids[max(0, position - 6) : min(len(token_ids), position + 7)].tolist()
            high_rows.append(
                {
                    "trajectory_index": index,
                    "trajectory_uid": record["uid"],
                    "question_uid": record["question_uid"],
                    "response_index": record["response_index"],
                    "token_position": position,
                    "token_id": int(token_ids[position]),
                    "token_string": token_strings[position],
                    "context": _decode(tokenizer, context_ids).replace("\n", "\\n"),
                    "selected_by_top_n": ",".join(selected_by),
                    "teachers_above_trajectory_p90": sum(high_flags),
                    **{f"eu_{name}": float(eu[position, j]) for j, name in enumerate(names)},
                    **{f"percentile_{name}": percentiles[j] for j, name in enumerate(names)},
                }
            )
        for position, token_piece in enumerate(readable_pieces):
            percentiles = percentile_rows[position]
            above_p90 = [names[j] for j, value in enumerate(percentiles) if value is not None and value >= 90.0]
            above_p95 = [names[j] for j, value in enumerate(percentiles) if value is not None and value >= 95.0]
            if not _is_semantic_token(token_piece) or (not above_p95 and len(above_p90) < 2):
                continue
            decoded_piece = _decode(tokenizer, [int(token_ids[position])])
            context_ids = token_ids[max(0, position - 10) : min(len(token_ids), position + 11)].tolist()
            semantic_rows.append(
                {
                    "trajectory_index": index,
                    "trajectory_uid": record["uid"],
                    "question_uid": record["question_uid"],
                    "response_index": record["response_index"],
                    "token_position": position,
                    "token_id": int(token_ids[position]),
                    "token_string": token_strings[position],
                    "decoded_token": decoded_piece.replace("\n", "\\n"),
                    "semantic_unit": _semantic_unit(tokenizer, token_ids, readable_pieces, position).replace(
                        "\n", "\\n"
                    ),
                    "context": _decode(tokenizer, context_ids).replace("\n", "\\n"),
                    "teachers_above_trajectory_p95": ",".join(above_p95),
                    "teachers_above_trajectory_p90": ",".join(above_p90),
                    "teacher_count_above_p95": len(above_p95),
                    "teacher_count_above_p90": len(above_p90),
                    **{f"eu_{name}": float(eu[position, j]) for j, name in enumerate(names)},
                    **{f"percentile_{name}": percentiles[j] for j, name in enumerate(names)},
                }
            )
        if index in selected:
            svg_name = f"trajectory-{index:06d}.svg"
            html_name = f"trajectory-{index:06d}.html"
            (trajectory_reports / svg_name).write_text(
                trajectory_svg(eu, valid, names, highlighted), encoding="utf-8"
            )
            trajectory_stats = [row for row in stats_rows if row["trajectory_index"] == index]
            trajectory_high = [row for row in high_rows if row["trajectory_index"] == index]
            trajectory_semantic = [row for row in semantic_rows if row["trajectory_index"] == index]
            semantic_table_rows = "".join(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(row[key]))}</td>"
                    for key in (
                        "token_position",
                        "decoded_token",
                        "semantic_unit",
                        "context",
                        "teachers_above_trajectory_p95",
                        "teachers_above_trajectory_p90",
                    )
                )
                + "".join(f"<td>{_teacher_value_cell(row, name)}</td>" for name in names)
                + "</tr>"
                for row in trajectory_semantic
            )
            table_rows = "".join(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(row[key]))}</td>"
                    for key in (
                        "token_position",
                        "token_string",
                        "context",
                        "selected_by_top_n",
                        "teachers_above_trajectory_p90",
                    )
                )
                + "".join(f"<td>{_number(row[f'eu_{name}'])}</td>" for name in names)
                + "</tr>"
                for row in trajectory_high
            )
            stats_table = "".join(
                f"<tr><td>{html.escape(row['teacher'])}</td><td>{_number(row['min'])}</td>"
                f"<td>{_number(row['mean'])}</td><td>{_number(row['p50'])}</td>"
                f"<td>{_number(row['p90'])}</td><td>{_number(row['p95'])}</td><td>{_number(row['max'])}</td></tr>"
                for row in trajectory_stats
            )
            page = f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(record['uid'])} EU</title>
<style>
body{{font-family:system-ui;max-width:1280px;margin:24px auto}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:5px}}
pre{{white-space:pre-wrap;background:#f6f8fa;padding:12px}}
</style>
<h1>{html.escape(record['uid'])}</h1><p><b>Question:</b> {html.escape(record['question'])}</p>
<img src="{svg_name}" alt="Per-token EU plot" style="max-width:100%">
<h2>Teacher interval and mean</h2>
<table><tr><th>Teacher</th><th>Min</th><th>Mean</th><th>Median</th>
<th>p90</th><th>p95</th><th>Max</th></tr>{stats_table}</table>
<h2>Semantic high-EU tokens</h2>
<p>Formatting-only Markdown/LaTeX pieces are excluded. A content token appears when any teacher is at or above
its trajectory p95, or at least two teachers are at or above their own trajectory p90. Values are raw EU followed
by the within-teacher trajectory percentile in parentheses.</p>
<table><tr><th>Position</th><th>High-EU token</th><th>Semantic unit</th><th>Context</th><th>Teachers &ge; p95</th>
<th>Teachers &ge; p90</th>{''.join(f'<th>{html.escape(name)} EU (percentile)</th>' for name in names)}</tr>
{semantic_table_rows}</table>
<h2>Raw top-{top_n} EU tokens</h2>
<p>Rows are the union of each teacher's top {top_n} tokens; the p90 count is trajectory-relative.</p>
<table><tr><th>Position</th><th>Token</th><th>Context</th><th>Selected by top-N</th>
<th>Teachers ≥ p90</th>{''.join(f'<th>{html.escape(name)}</th>' for name in names)}</tr>{table_rows}</table>
<h2>Student response</h2><pre>{html.escape(record['response'])}</pre>"""
            (trajectory_reports / html_name).write_text(page, encoding="utf-8")
            pages.append((record["uid"], f"trajectories/{html_name}"))
    flattened = [np.concatenate(parts) for parts in global_values]
    global_stats = [
        {"teacher": name, **extended_statistics(values)}
        for name, values in zip(names, flattened, strict=True)
    ]
    write_json(analysis / "teacher_summary.json", global_stats)
    write_json(analysis / "per_trajectory_teacher_stats.json", stats_rows)
    stats_fields = list(stats_rows[0])
    write_csv(analysis / "per_trajectory_teacher_stats.csv", stats_rows, stats_fields)
    high_fields = list(high_rows[0]) if high_rows else []
    if high_fields:
        write_csv(analysis / "high_eu_tokens.csv", high_rows, high_fields)
    semantic_fields = list(semantic_rows[0]) if semantic_rows else []
    if semantic_fields:
        write_csv(analysis / "semantic_high_eu_tokens.csv", semantic_rows, semantic_fields)
    (analysis / "teacher_distribution.svg").write_text(
        distribution_svg(flattened, names), encoding="utf-8"
    )
    summary_table = "".join(
        f"<tr><td>{html.escape(row['teacher'])}</td><td>{row['finite_tokens']}</td>"
        f"<td>{_number(row['min'])}</td><td>{_number(row['mean'])}</td><td>{_number(row['p50'])}</td>"
        f"<td>{_number(row['p90'])}</td><td>{_number(row['p95'])}</td><td>{_number(row['max'])}</td></tr>"
        for row in global_stats
    )
    links = "".join(f'<li><a href="{href}">{html.escape(uid)}</a></li>' for uid, href in pages)
    report = f"""<!doctype html><meta charset="utf-8"><title>Teacher token EU report</title>
<style>
body{{font-family:system-ui;max-width:1200px;margin:24px auto}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:6px}}
</style>
<h1>Three-teacher token-level EU</h1>
<p>EU = 16 / sum(Top16(raw teacher logits) + 1), before temperature or softmax.
Raw-logit scales are not calibrated across model sizes; compare within-teacher percentiles
as well as absolute values.</p>
<p>The saved baseline JSONL omitted original vLLM token IDs and terminal stop-token metadata.
This report covers the visible response text reconstructed losslessly with the original student tokenizer.</p>
<img src="teacher_distribution.svg" alt="Teacher EU distributions" style="max-width:100%">
<h2>Corpus interval and mean</h2>
<table><tr><th>Teacher</th><th>Finite tokens</th><th>Min</th><th>Mean</th><th>Median</th>
<th>p90</th><th>p95</th><th>Max</th></tr>{summary_table}</table>
<h2>Selected trajectory reports</h2><ul>{links}</ul>
<p>Machine-readable files: teacher_summary.json, per_trajectory_teacher_stats.csv/json,
semantic_high_eu_tokens.csv, and the legacy raw top-N high_eu_tokens.csv.</p>"""
    (analysis / "report.html").write_text(report, encoding="utf-8")
    write_json(
        analysis / "ANALYSIS_COMPLETED.json",
        {
            "trajectories": len(records),
            "plotted_trajectories": sorted(selected),
            "high_token_top_n_per_teacher": top_n,
            "semantic_high_eu_rule": "content token with any teacher >= trajectory p95 or at least two >= p90",
            "semantic_high_eu_tokens": len(semantic_rows),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--rollouts-jsonl", type=Path)
    parser.add_argument("--model-label")
    parser.add_argument("--task", default="AMC23", choices=("AIME24", "AIME25", "AMC23"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-4b", type=Path, required=True)
    parser.add_argument("--teacher-8b", type=Path, required=True)
    parser.add_argument("--teacher-14b", type=Path, required=True)
    parser.add_argument("--score-chunk-size", type=int, default=128)
    parser.add_argument("--max-trajectories", type=int, default=0, help="0 means all source rollouts")
    parser.add_argument(
        "--plot-trajectory",
        action="append",
        default=[],
        help="all, index:N, or EXAMPLE_ID:RESPONSE_INDEX; defaults to 0:0",
    )
    parser.add_argument("--high-token-top-n", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ValueError(f"Output root already exists: {output_root}")
    config_path, eval_config, rollout_path = resolve_source(
        args.evaluation_root, args.task, args.model_label, args.rollouts_jsonl
    )
    student_path = Path(eval_config["model_path"]).resolve()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    source_rows = read_jsonl(rollout_path)
    records = build_trajectories(source_rows, tokenizer, eval_config, args.task, args.max_trajectories)
    teachers = [
        {"name": name, "path": str(path.resolve())}
        for name, path in zip(
            token_eu.TEACHER_NAMES,
            (args.teacher_4b, args.teacher_8b, args.teacher_14b),
            strict=True,
        )
    ]
    validation = validate_models(records, student_path, teachers)
    plot_specs = args.plot_trajectory or ["0:0"]
    manifest = {
        "evaluation_root": str(args.evaluation_root.resolve()),
        "evaluation_config": str(config_path),
        "evaluation_config_sha256": file_hash(config_path),
        "source_rollouts": str(rollout_path),
        "source_rollouts_sha256": file_hash(rollout_path),
        "task": args.task,
        "student_model": str(student_path),
        "source_rows": len(source_rows),
        "selected_trajectories": len(records),
        "visible_response_tokens": sum(len(row["response_token_ids"]) for row in records),
        "token_reconstruction": {
            "method": "student tokenizer encode(saved visible text)",
            "roundtrip_required": True,
            "original_vllm_token_ids_available": False,
            "terminal_stop_token_recoverable": False,
        },
        "plot_trajectories": plot_specs,
        "validation": validation,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    temp_root = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(prefix="student-rollout-eu-", dir=temp_root) as directory:
        trajectories = Path(directory) / "trajectories.json"
        write_json(trajectories, records)
        command = [
            sys.executable,
            str(Path(token_eu.__file__).resolve()),
            "--trajectories",
            str(trajectories),
            "--output-root",
            str(output_root),
            "--teacher-4b",
            teachers[0]["path"],
            "--teacher-8b",
            teachers[1]["path"],
            "--teacher-14b",
            teachers[2]["path"],
            "--tokenizer-path",
            str(student_path),
            "--score-chunk-size",
            str(args.score_chunk_size),
        ]
        subprocess.run(command, check=True)
    write_json(output_root / "source_manifest.json", manifest)
    analyze_output(output_root, tokenizer, plot_specs, args.high_token_top_n)
    print(f"Completed EU report: {output_root / 'analysis/report.html'}", flush=True)


if __name__ == "__main__":
    main()
