#!/usr/bin/env python3
"""Fixed-question, on-policy overlap diagnostics (not an accuracy evaluation).

Each pair normally generates its own trajectories. Pairs that share an exact
student can explicitly reuse a prior pair's trajectories and student scores.
Both models then score exactly those token prefixes with untempered,
full-vocabulary-normalized probabilities.
Generation and each scoring model run in separate processes to release VRAM.
Only response predictions are measured, including generated EOS if present.
Outputs never overwrite an existing run directory. No training is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys


TASKS = ("AIME24", "AIME25", "AMC23")
PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_overlap(student_ids, student_probs, teacher_ids, teacher_probs, k):
    """Return count ratio and ORIGINAL probability masses on the intersection."""
    if k < 1 or min(len(student_ids), len(teacher_ids)) < k:
        raise ValueError("Not enough top-k entries")
    student = dict(zip(student_ids[:k], student_probs[:k], strict=True))
    teacher = dict(zip(teacher_ids[:k], teacher_probs[:k], strict=True))
    if len(student) != k or len(teacher) != k:
        raise ValueError("Duplicate top-k token IDs")
    shared = student.keys() & teacher.keys()
    return (
        len(shared) / k,
        math.fsum(student[token] for token in shared),
        math.fsum(teacher[token] for token in shared),
    )


def topk_mechanism_metrics(
    student_ids,
    student_probs,
    teacher_ids,
    teacher_probs,
    k,
    teacher_on_student_log_probs=None,
):
    """Compute per-position metrics available from saved top-k scores.

    ``sparse_union_cosine`` treats probabilities outside each model's own top-k
    as zero. This makes it exactly recoverable from historical result files.
    ``shared_weighted_abs_logprob_gap`` is conditional on a nonempty top-k
    intersection. The full student-top-k gap is returned only when the scorer
    saved teacher log probabilities on every student top-k token.
    """
    import numpy as np

    arrays = (student_ids, student_probs, teacher_ids, teacher_probs)
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Top-k score arrays must be two-dimensional")
    if len({array.shape[0] for array in arrays}) != 1 or min(array.shape[1] for array in arrays) < k:
        raise ValueError("Top-k score arrays do not have compatible shapes")

    student_ids = student_ids[:, :k]
    student_probs = student_probs[:, :k].astype(np.float64, copy=False)
    teacher_ids = teacher_ids[:, :k]
    teacher_probs = teacher_probs[:, :k].astype(np.float64, copy=False)
    if ((np.diff(np.sort(student_ids, axis=1), axis=1) == 0).any()
            or (np.diff(np.sort(teacher_ids, axis=1), axis=1) == 0).any()):
        raise ValueError("Top-k token IDs must be unique within each response position")
    if (not np.isfinite(student_probs).all() or not np.isfinite(teacher_probs).all()
            or (student_probs <= 0).any() or (teacher_probs <= 0).any()):
        raise ValueError("Top-k probabilities must be finite and strictly positive")

    shared = student_ids[:, :, None] == teacher_ids[:, None, :]
    student_shared = shared.any(axis=2)
    teacher_on_shared_student = (shared * teacher_probs[:, None, :]).sum(axis=2)

    numerator = (shared * student_probs[:, :, None] * teacher_probs[:, None, :]).sum(axis=(1, 2))
    denominator = np.sqrt((student_probs ** 2).sum(axis=1) * (teacher_probs ** 2).sum(axis=1))
    sparse_union_cosine = numerator / denominator

    shared_student_mass = (student_probs * student_shared).sum(axis=1)
    safe_teacher_probs = np.where(student_shared, teacher_on_shared_student, 1.0)
    shared_gap_numerator = (
        student_probs
        * student_shared
        * np.abs(np.log(student_probs) - np.log(safe_teacher_probs))
    ).sum(axis=1)
    shared_gap = np.full(len(student_ids), np.nan, dtype=np.float64)
    np.divide(shared_gap_numerator, shared_student_mass, out=shared_gap, where=shared_student_mass > 0)

    exact_gap = None
    if teacher_on_student_log_probs is not None:
        if teacher_on_student_log_probs.ndim != 2 or teacher_on_student_log_probs.shape[0] != len(student_ids) \
                or teacher_on_student_log_probs.shape[1] < k:
            raise ValueError("Teacher-on-student log probabilities do not match student top-k scores")
        teacher_on_student_log_probs = teacher_on_student_log_probs[:, :k].astype(np.float64, copy=False)
        if not np.isfinite(teacher_on_student_log_probs).all():
            raise ValueError("Teacher-on-student log probabilities must be finite")
        student_weights = student_probs / student_probs.sum(axis=1, keepdims=True)
        exact_gap = (
            student_weights
            * np.abs(np.log(student_probs) - teacher_on_student_log_probs)
        ).sum(axis=1)

    return {
        "sparse_union_cosine": sparse_union_cosine,
        "shared_weighted_abs_logprob_gap": shared_gap,
        "student_topk_weighted_abs_logprob_gap": exact_gap,
    }


def select_samples(data_dir, count, seed):
    import pandas as pd

    selected, sources = [], {}
    for task in TASKS:
        source = Path(data_dir) / task / "test.parquet"
        frame = pd.read_parquet(source)
        if not 0 < count <= len(frame):
            raise ValueError(f"{task}: sample count {count} must be within 1..{len(frame)}")
        sources[task] = {"path": str(source.resolve()), "sha256": file_hash(source), "rows": len(frame)}
        # A task-specific RNG prevents changes in other tasks from changing this subset.
        indices = sorted(random.Random(f"{seed}:{task}").sample(range(len(frame)), count))
        for index in indices:
            row = frame.iloc[index]
            selected.append({
                "uid": f"{task}:{index}", "task": task, "row_index": index,
                "question": str(row["prompt"][0]["content"]).strip(),
                "answer": str(row["reward_model"]["ground_truth"]).strip(),
            })
    return {"seed": seed, "questions_per_task": count, "sources": sources, "samples": selected}


def expand_response_samples(samples, samples_per_question, seed):
    """Assign stable unique IDs and seeds to repeated responses for each question."""
    if samples_per_question < 1:
        raise ValueError("Samples per question must be positive")
    expanded = []
    for sample_index, sample in enumerate(samples):
        for response_index in range(samples_per_question):
            expanded.append({
                **sample,
                "question_uid": sample["uid"],
                "uid": f"{sample['uid']}:response-{response_index + 1:02d}",
                "response_index": response_index,
                "seed": seed + sample_index * samples_per_question + response_index,
            })
    return expanded


def tokenizer_signature(tokenizer):
    payload = {"vocab": tokenizer.get_vocab(), "special_tokens": tokenizer.special_tokens_map}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def preflight(config):
    from transformers import AutoConfig, AutoTokenizer

    for pair in config["pairs"]:
        signatures = []
        prompt_lengths = []
        context_limits = []
        pair["model_files"] = {}
        for role in ("student", "teacher"):
            path = Path(pair[role])
            if not path.is_absolute() or not (path / "config.json").is_file():
                raise ValueError(f"{role} requires an absolute, merged HF model directory: {path}")
            if not list(path.glob("*.safetensors")):
                raise ValueError(f"No safetensors weights in {path}; FSDP actors must be merged first")
            model_config = AutoConfig.from_pretrained(path, local_files_only=True)
            context_limits.append(model_config.max_position_embeddings)
            pair["model_files"][role] = {
                "config_sha256": file_hash(path / "config.json"),
                "weights": [{"name": weight.name, "size": weight.stat().st_size,
                             "mtime_ns": weight.stat().st_mtime_ns}
                            for weight in sorted(path.glob("*.safetensors"))],
            }
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            signatures.append(tokenizer_signature(tokenizer))
            if model_config.vocab_size < max(config["ks"]):
                raise ValueError("Diagnostic k exceeds model vocabulary")
            if model_config.model_type != "qwen3":
                raise ValueError("This scoring path is validated for Qwen3 causal language models only")
            for sample in config["samples"]:
                prompt_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=sample["question"])}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=True,
                )
                prompt_lengths.append(len(prompt_ids))
                if len(prompt_ids) + config["max_tokens"] > model_config.max_position_embeddings:
                    raise ValueError(f"{pair['label']} {role}: prompt + response exceeds model context")
        if signatures[0] != signatures[1]:
            raise ValueError(f"{pair['label']}: teacher/student token-ID mappings or special tokens differ")
        pair["tokenizer_sha256"] = signatures[0]
        pair["max_prompt_tokens"] = max(prompt_lengths)
        if pair["max_prompt_tokens"] + config["max_tokens"] > min(context_limits):
            raise ValueError(f"{pair['label']}: shared prefix budget exceeds student/teacher context")


def generate(config, pair, directory):
    from vllm import LLM, SamplingParams

    llm = LLM(model=pair["student"], tensor_parallel_size=1,
              gpu_memory_utilization=0.85, max_num_seqs=1,
              max_model_len=config["max_tokens"] + pair["max_prompt_tokens"],
              seed=config["seed"], enable_prefix_caching=False)
    tokenizer = llm.get_tokenizer()
    if tokenizer_signature(tokenizer) != pair["tokenizer_sha256"]:
        raise ValueError("Generation tokenizer differs from preflight tokenizer")
    records = []
    response_samples = expand_response_samples(
        config["samples"], config.get("samples_per_question", 1), config["seed"]
    )
    for index, sample in enumerate(response_samples):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=sample["question"])}],
            tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
        params = SamplingParams(n=1, seed=sample["seed"],
                                temperature=config["temperature"], top_p=config["top_p"],
                                top_k=config["generation_top_k"], min_p=0.0,
                                max_tokens=config["max_tokens"])
        result = llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)[0].outputs[0]
        response_ids = list(result.token_ids)
        if not response_ids:
            raise ValueError(f"Empty response for {sample['uid']}")
        record = {**sample, "prompt_token_ids": prompt_ids,
                  "response_token_ids": response_ids, "response": result.text,
                  "finish_reason": result.finish_reason, "stop_reason": result.stop_reason}
        records.append(record)
        write_json(directory / f"trajectory-{index:03d}.json", record)
        print(f"{pair['label']} generated {sample['uid']}: {len(response_ids)} tokens", flush=True)
    write_json(directory / "trajectories.json", records)


def score(config, pair, directory, role):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        pair[role], local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0").eval().requires_grad_(False)
    k = max(config["ks"])
    chunk_size = config["score_chunk_size"]
    for index, record in enumerate(read_json(directory / "trajectories.json")):
        # Position P-1 predicts response token 0. Last response token is a target,
        # never an extra input whose next-token distribution would be unobserved.
        prompt_size = len(record["prompt_token_ids"])
        inputs = record["prompt_token_ids"] + record["response_token_ids"][:-1]
        student_top_k_ids = None
        if role == "teacher":
            with np.load(directory / f"student-topk-{index:03d}.npz", allow_pickle=False) as student:
                student_top_k_ids = student["ids"]
            if student_top_k_ids.shape != (len(record["response_token_ids"]), k):
                raise ValueError("Student top-k IDs are unavailable or misaligned for teacher cross-scoring")
        cache, id_chunks, probability_chunks, teacher_on_student_chunks = None, [], [], []
        with torch.inference_mode():
            for start in range(0, len(inputs), chunk_size):
                end = min(start + chunk_size, len(inputs))
                batch = torch.tensor([inputs[start:end]], dtype=torch.long, device="cuda:0")
                output = model(input_ids=batch, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                first = max(prompt_size - 1 - start, 0)
                if first < end - start:
                    logits = output.logits[0, first:].float()
                    values, ids = torch.topk(logits, k=k, dim=-1)
                    log_normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
                    probabilities = (values - log_normalizer).exp()
                    id_chunks.append(ids.cpu().numpy())
                    probability_chunks.append(probabilities.cpu().numpy())
                    if student_top_k_ids is not None:
                        response_start = start + first - (prompt_size - 1)
                        response_end = response_start + len(logits)
                        target_ids = torch.as_tensor(
                            student_top_k_ids[response_start:response_end], dtype=torch.long, device="cuda:0"
                        )
                        teacher_on_student = torch.gather(logits, dim=-1, index=target_ids) - log_normalizer
                        teacher_on_student_chunks.append(teacher_on_student.cpu().numpy())
                        del target_ids, teacher_on_student
                    del logits, values, ids, probabilities, log_normalizer
                del output, batch
        ids = np.concatenate(id_chunks)
        probabilities = np.concatenate(probability_chunks)
        if ids.shape != (len(record["response_token_ids"]), k):
            raise ValueError("Response/logit alignment error")
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite top-k probabilities")
        np.savez_compressed(directory / f"{role}-topk-{index:03d}.npz", ids=ids, probabilities=probabilities)
        if student_top_k_ids is not None:
            teacher_on_student = np.concatenate(teacher_on_student_chunks)
            if teacher_on_student.shape != student_top_k_ids.shape or not np.isfinite(teacher_on_student).all():
                raise ValueError("Teacher-on-student score alignment error")
            np.savez_compressed(
                directory / f"teacher-on-student-topk-{index:03d}.npz",
                ids=student_top_k_ids,
                log_probabilities=teacher_on_student,
            )
        del cache
        print(f"{pair['label']} {role} scored {record['uid']}: {len(ids)} tokens", flush=True)


def _question_uid(row):
    return row.get("question_uid", row["uid"])


def _question_mean(rows, value_key):
    from collections import defaultdict

    grouped = defaultdict(list)
    for row in rows:
        if row[value_key] is not None:
            grouped[_question_uid(row)].append(row[value_key])
    if not grouped:
        return None
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def summarize(config, root):
    import numpy as np

    rows, summaries, chunk_rows = [], [], []
    # Match ray_trainer.py's response-position bins, independently of the
    # smaller forward-pass chunks used solely to bound GPU memory.
    position_chunk_size = 1024
    for pair in config["pairs"]:
        directory = root / pair["label"]
        records = read_json(directory / "trajectories.json")
        max_response_length = max(len(record["response_token_ids"]) for record in records)
        for index, record in enumerate(records):
            with np.load(directory / f"student-topk-{index:03d}.npz", allow_pickle=False) as student:
                student_ids, student_probs = student["ids"], student["probabilities"]
            with np.load(directory / f"teacher-topk-{index:03d}.npz", allow_pickle=False) as teacher:
                teacher_ids, teacher_probs = teacher["ids"], teacher["probabilities"]
            n = len(record["response_token_ids"])
            if student_ids.shape != teacher_ids.shape or len(student_ids) != n:
                raise ValueError("Student and teacher score arrays are misaligned")
            for k in config["ks"]:
                # Compute small k x k intersections per position, never a vocab x vocab matrix.
                shared = student_ids[:, :k, None] == teacher_ids[:, None, :k]
                student_mask, teacher_mask = shared.any(axis=2), shared.any(axis=1)
                values = np.stack((student_mask.sum(axis=1) / k,
                                   (student_probs[:, :k] * student_mask).sum(axis=1),
                                   (teacher_probs[:, :k] * teacher_mask).sum(axis=1)), axis=1)
                np.savez_compressed(directory / f"overlap-k{k}-{index:03d}.npz",
                                    ratio=values[:, 0], student_mass=values[:, 1], teacher_mass=values[:, 2])
                rows.append({"label": pair["label"], "uid": record["uid"],
                             "question_uid": record.get("question_uid", record["uid"]),
                             "response_index": record.get("response_index", 0), "task": record["task"],
                             "k": k, "tokens": n, "finish_reason": record["finish_reason"],
                             "ratio": float(values[:, 0].mean()),
                             "student_mass": float(values[:, 1].mean()),
                             "teacher_mass": float(values[:, 2].mean())})
                for start in range(0, n, position_chunk_size):
                    end = min(start + position_chunk_size, max_response_length)
                    segment = values[start:min(end, n)]
                    nonempty = segment[:, 0] > 0
                    chunk_rows.append({
                        "label": pair["label"], "uid": record["uid"],
                        "question_uid": record.get("question_uid", record["uid"]),
                        "response_index": record.get("response_index", 0), "task": record["task"], "k": k,
                        "start": start, "end": end, "chunk_key": f"{start}_{end}",
                        "tokens": len(segment), "nonempty_tokens": int(nonempty.sum()),
                        "ratio": float(segment[:, 0].mean()),
                        "student_mass": float(segment[:, 1].mean()),
                        "teacher_mass": float(segment[:, 2].mean()),
                        "student_mass_nonempty": float(segment[nonempty, 1].mean()) if nonempty.any() else None,
                        "teacher_mass_nonempty": float(segment[nonempty, 2].mean()) if nonempty.any() else None,
                    })
        for task in (*TASKS, "ALL"):
            for k in config["ks"]:
                subset = [r for r in rows if r["label"] == pair["label"] and r["k"] == k
                          and (task == "ALL" or r["task"] == task)]
                total = sum(r["tokens"] for r in subset)
                result = {
                    "label": pair["label"], "task": task, "k": k,
                    "questions": len({_question_uid(row) for row in subset}),
                    "responses": len(subset), "tokens": total,
                    "length_limited": sum(r["finish_reason"] == "length" for r in subset),
                }
                for name in ("ratio", "student_mass", "teacher_mass"):
                    result[f"token_mean_{name}"] = sum(r[name] * r["tokens"] for r in subset) / total
                    result[f"question_mean_{name}"] = _question_mean(subset, name)
                summaries.append(result)
    write_json(root / "per_question.json", rows)
    write_json(root / "per_question_chunks.json", chunk_rows)
    summarize_chunks(chunk_rows, root)
    write_json(root / "summary.json", summaries)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    has_reuse = any(pair.get("trajectory_source") for pair in config["pairs"])
    trajectory_note = (
        "Pairs with trajectory_source reused the exact source trajectories and student scores; all teachers scored "
        "those identical prefixes." if has_reuse else
        "Each student generated its own trajectory; teacher and student scored identical prefixes within each pair."
    )
    lines = ["# Fixed-question overlap diagnostics", "", trajectory_note,
             "These are different teacher/student pairs, not a within-stage training trend or an accuracy benchmark.",
             "Mass uses raw full-vocabulary probabilities at temperature 1, without sampling filters or top-k renormalization.",
             "Token means weight long answers more; question means first average repeated responses for each question. "
             "EOS is included if generated.",
             "Response-position statistics (1024 tokens per bin): [chunks](summary_chunks.md), "
             "with CSV/JSON and per_question_chunks.json. These bins are independent of score_chunk_size.",
             "", "| Pair | Task | k | Questions | Responses | Tokens | Length-limited responses | "
             "Ratio (token mean) | Student mass | Teacher mass |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summaries:
        lines.append(f"| {row['label']} | {row['task']} | {row['k']} | {row['questions']} | "
                     f"{row['responses']} | {row['tokens']} | {row['length_limited']} | "
                     f"{row['token_mean_ratio']:.6f} | "
                     f"{row['token_mean_student_mass']:.6f} | {row['token_mean_teacher_mass']:.6f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_chunks(rows, root):
    """Aggregate occupied response bins; missing suffixes contribute no padding."""
    from collections import defaultdict

    groups = defaultdict(list)
    for row in rows:
        for task in (row["task"], "ALL"):
            groups[(row["label"], task, row["k"], row["start"], row["end"])].append(row)
    results = []
    for (label, task, k, start, end), subset in sorted(groups.items()):
        total = sum(row["tokens"] for row in subset)
        nonempty = sum(row["nonempty_tokens"] for row in subset)
        key = f"{start}_{end}"
        result = {
            "label": label, "task": task, "k": k, "start": start, "end": end, "chunk_key": key,
            "questions": len({_question_uid(row) for row in subset}),
            "responses": len(subset), "tokens": total, "nonempty_tokens": nonempty,
        }
        for name in ("ratio", "student_mass", "teacher_mass"):
            result[f"token_mean_{name}"] = sum(row[name] * row["tokens"] for row in subset) / total
            result[f"question_mean_{name}"] = _question_mean(subset, name)
        for name in ("student_mass", "teacher_mass"):
            result[f"token_mean_{name}_nonempty"] = (
                sum(row[f"{name}_nonempty"] * row["nonempty_tokens"]
                    for row in subset if row["nonempty_tokens"]) / nonempty if nonempty else None
            )
        # Legacy metric names identify the original aggregation semantics.
        result["legacy_ratio_metric"] = f"val-topk/overlap_ratio_chunk_{key}"
        result["legacy_student_mass_metric"] = f"val-topk/student_p_sum_intersection_chunk_{key}"
        result["legacy_teacher_mass_metric"] = f"val-topk/teacher_p_sum_intersection_chunk_{key}"
        results.append(result)
    write_json(root / "summary_chunks.json", results)
    with (root / "summary_chunks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Overlap by response position", "",
             "Zero-based half-open bins [start, end), excluding the prompt. No padding is counted.",
             "Only questions reaching a bin contribute; later bins can contain a different subset of questions.",
             "Ratio and default mass means include every valid response position (zero mass for empty intersections).",
             "The *_mass_nonempty columns reproduce the original trainer's intersection-nonempty averaging rule; "
             "null means no positions with a nonempty intersection, not zero probability.",
             "The legacy metric names in CSV/JSON map ratio to token_mean_ratio and mass to token_mean_*_mass_nonempty.",
             "", "| Pair | Task | k | Response positions | Questions | Responses | Tokens | "
             "Ratio | Student mass | Teacher mass |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results:
        lines.append(f"| {row['label']} | {row['task']} | {row['k']} | {row['chunk_key']} | "
                     f"{row['questions']} | {row['responses']} | {row['tokens']} | "
                     f"{row['token_mean_ratio']:.6f} | "
                     f"{row['token_mean_student_mass']:.6f} | {row['token_mean_teacher_mass']:.6f} |")
    (root / "summary_chunks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optional_weighted_mean(rows, value_key, weight_key):
    usable = [row for row in rows if row[value_key] is not None and row[weight_key] > 0]
    weight = sum(row[weight_key] for row in usable)
    if not weight:
        return None
    return sum(row[value_key] * row[weight_key] for row in usable) / weight


def _optional_question_mean(rows, value_key):
    return _question_mean(rows, value_key)


def _mechanism_aggregate(subset):
    total = sum(row["tokens"] for row in subset)
    shared_total = sum(row["shared_gap_tokens"] for row in subset)
    exact_total = sum(row["exact_gap_tokens"] for row in subset)
    return {
        "questions": len({_question_uid(row) for row in subset}),
        "responses": len(subset),
        "tokens": total,
        "shared_gap_tokens": shared_total,
        "exact_gap_tokens": exact_total,
        "exact_gap_available": bool(total) and exact_total == total,
        "token_mean_sparse_union_cosine": _optional_weighted_mean(
            subset, "sparse_union_cosine", "tokens"
        ),
        "question_mean_sparse_union_cosine": _optional_question_mean(subset, "sparse_union_cosine"),
        "token_mean_shared_weighted_abs_logprob_gap": _optional_weighted_mean(
            subset, "shared_weighted_abs_logprob_gap", "shared_gap_tokens"
        ),
        "question_mean_shared_weighted_abs_logprob_gap": _optional_question_mean(
            subset, "shared_weighted_abs_logprob_gap"
        ),
        "token_mean_student_topk_weighted_abs_logprob_gap": _optional_weighted_mean(
            subset, "student_topk_weighted_abs_logprob_gap", "exact_gap_tokens"
        ),
        "question_mean_student_topk_weighted_abs_logprob_gap": _optional_question_mean(
            subset, "student_topk_weighted_abs_logprob_gap"
        ),
    }


def _format_optional(value):
    return "N/A" if value is None else f"{value:.6f}"


def analyze_existing(config, source_root, output_root, require_completed=True):
    """Derive mechanism metrics without changing an existing evaluation tree."""
    import numpy as np

    source_root, output_root = Path(source_root), Path(output_root)
    if require_completed and not (source_root / "COMPLETED.json").is_file():
        raise ValueError(f"Existing evaluation is not marked complete: {source_root}")
    if output_root.exists():
        raise ValueError(f"Analysis output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    question_rows, chunk_rows = [], []
    position_chunk_size = 1024
    try:
        for pair in config["pairs"]:
            source_directory = source_root / pair["label"]
            output_directory = output_root / pair["label"]
            output_directory.mkdir()
            records = read_json(source_directory / "trajectories.json")
            max_response_length = max(len(record["response_token_ids"]) for record in records)
            for index, record in enumerate(records):
                with np.load(source_directory / f"student-topk-{index:03d}.npz", allow_pickle=False) as student:
                    student_ids, student_probs = student["ids"], student["probabilities"]
                with np.load(source_directory / f"teacher-topk-{index:03d}.npz", allow_pickle=False) as teacher:
                    teacher_ids, teacher_probs = teacher["ids"], teacher["probabilities"]
                n = len(record["response_token_ids"])
                if student_ids.shape != teacher_ids.shape or len(student_ids) != n:
                    raise ValueError(f"{pair['label']} {record['uid']}: top-k score arrays are misaligned")

                cross_path = source_directory / f"teacher-on-student-topk-{index:03d}.npz"
                cross_ids = cross_log_probs = None
                if cross_path.is_file():
                    with np.load(cross_path, allow_pickle=False) as cross:
                        cross_ids, cross_log_probs = cross["ids"], cross["log_probabilities"]
                    if cross_ids.shape != student_ids.shape or not np.array_equal(cross_ids, student_ids):
                        raise ValueError(f"{pair['label']} {record['uid']}: teacher-on-student IDs are misaligned")

                for k in config["ks"]:
                    metrics = topk_mechanism_metrics(
                        student_ids, student_probs, teacher_ids, teacher_probs, k, cross_log_probs
                    )
                    shared_gap = metrics["shared_weighted_abs_logprob_gap"]
                    exact_gap = metrics["student_topk_weighted_abs_logprob_gap"]
                    saved_metrics = {name: value for name, value in metrics.items() if value is not None}
                    np.savez_compressed(output_directory / f"mechanism-k{k}-{index:03d}.npz", **saved_metrics)
                    shared_valid = np.isfinite(shared_gap)
                    row = {
                        "label": pair["label"], "uid": record["uid"],
                        "question_uid": record.get("question_uid", record["uid"]),
                        "response_index": record.get("response_index", 0), "task": record["task"], "k": k,
                        "tokens": n,
                        "shared_gap_tokens": int(shared_valid.sum()),
                        "exact_gap_tokens": n if exact_gap is not None else 0,
                        "sparse_union_cosine": float(metrics["sparse_union_cosine"].mean()),
                        "shared_weighted_abs_logprob_gap": (
                            float(shared_gap[shared_valid].mean()) if shared_valid.any() else None
                        ),
                        "student_topk_weighted_abs_logprob_gap": (
                            float(exact_gap.mean()) if exact_gap is not None else None
                        ),
                    }
                    question_rows.append(row)
                    for start in range(0, n, position_chunk_size):
                        end = min(start + position_chunk_size, max_response_length)
                        limit = min(end, n)
                        shared_segment = shared_gap[start:limit]
                        shared_segment_valid = np.isfinite(shared_segment)
                        exact_segment = exact_gap[start:limit] if exact_gap is not None else None
                        chunk_rows.append({
                            "label": pair["label"], "uid": record["uid"],
                            "question_uid": record.get("question_uid", record["uid"]),
                            "response_index": record.get("response_index", 0),
                            "task": record["task"], "k": k,
                            "start": start, "end": end, "chunk_key": f"{start}_{end}",
                            "tokens": limit - start,
                            "shared_gap_tokens": int(shared_segment_valid.sum()),
                            "exact_gap_tokens": limit - start if exact_segment is not None else 0,
                            "sparse_union_cosine": float(metrics["sparse_union_cosine"][start:limit].mean()),
                            "shared_weighted_abs_logprob_gap": (
                                float(shared_segment[shared_segment_valid].mean())
                                if shared_segment_valid.any() else None
                            ),
                            "student_topk_weighted_abs_logprob_gap": (
                                float(exact_segment.mean()) if exact_segment is not None else None
                            ),
                        })

        summaries = []
        for pair in config["pairs"]:
            for task in (*TASKS, "ALL"):
                for k in config["ks"]:
                    subset = [row for row in question_rows if row["label"] == pair["label"] and row["k"] == k
                              and (task == "ALL" or row["task"] == task)]
                    summaries.append({"label": pair["label"], "task": task, "k": k,
                                      **_mechanism_aggregate(subset)})

        from collections import defaultdict
        chunk_groups = defaultdict(list)
        for row in chunk_rows:
            for task in (row["task"], "ALL"):
                chunk_groups[(row["label"], task, row["k"], row["start"], row["end"])].append(row)
        chunk_summaries = []
        for (label, task, k, start, end), subset in sorted(chunk_groups.items()):
            chunk_summaries.append({
                "label": label, "task": task, "k": k, "start": start, "end": end,
                "chunk_key": f"{start}_{end}", **_mechanism_aggregate(subset),
            })

        write_json(output_root / "per_question.json", question_rows)
        write_json(output_root / "per_question_chunks.json", chunk_rows)
        write_json(output_root / "summary.json", summaries)
        write_json(output_root / "summary_chunks.json", chunk_summaries)
        for name, rows in (("summary.csv", summaries), ("summary_chunks.csv", chunk_summaries)):
            with (output_root / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        lines = [
            "# Top-k mechanism diagnostics", "",
            "Sparse-union cosine aligns token IDs and treats probabilities outside each model's own top-k as zero.",
            "It is a hard-truncated top-k diagnostic, not full-vocabulary cosine.",
            "Shared gap is the student-probability-weighted absolute log-probability gap conditional on a nonempty "
            "student/teacher top-k intersection. Exact student-top-k gap also includes student-only top-k tokens;",
            "it is N/A for historical runs that did not save teacher-on-student scores.",
            "All probabilities come from the untempered full-vocabulary softmax. Metrics are computed per response "
            "position before aggregation; token means weight long responses more than question means.", "",
            "| Pair | Task | k | Questions | Responses | Tokens | Sparse union cosine | "
            "Shared weighted abs log-prob gap | Exact student-top-k gap |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summaries:
            lines.append(
                f"| {row['label']} | {row['task']} | {row['k']} | {row['questions']} | "
                f"{row['responses']} | {row['tokens']} | "
                f"{_format_optional(row['token_mean_sparse_union_cosine'])} | "
                f"{_format_optional(row['token_mean_shared_weighted_abs_logprob_gap'])} | "
                f"{_format_optional(row['token_mean_student_topk_weighted_abs_logprob_gap'])} |"
            )
        (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        chunk_lines = [
            "# Top-k mechanism diagnostics by response position", "",
            "Zero-based half-open 1024-token bins; missing response suffixes do not contribute padding.", "",
            "| Pair | Task | k | Positions | Questions | Responses | Tokens | "
            "Sparse union cosine | Shared gap | Exact gap |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in chunk_summaries:
            chunk_lines.append(
                f"| {row['label']} | {row['task']} | {row['k']} | {row['chunk_key']} | "
                f"{row['questions']} | {row['responses']} | {row['tokens']} | "
                f"{_format_optional(row['token_mean_sparse_union_cosine'])} | "
                f"{_format_optional(row['token_mean_shared_weighted_abs_logprob_gap'])} | "
                f"{_format_optional(row['token_mean_student_topk_weighted_abs_logprob_gap'])} |"
            )
        (output_root / "summary_chunks.md").write_text("\n".join(chunk_lines) + "\n", encoding="utf-8")
        write_json(output_root / "analysis.json", {
            "source_root": str(source_root.resolve()),
            "source_config_sha256": file_hash(source_root / "config.json"),
            "source_completed": (source_root / "COMPLETED.json").is_file(),
            "pairs": [pair["label"] for pair in config["pairs"]],
            "ks": config["ks"],
            "historical_exact_gap_available": all(row["exact_gap_available"] for row in summaries),
        })
        write_json(output_root / "COMPLETED.json", {"complete": True, "pairs": len(config["pairs"])})
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def parse_pair(value):
    try:
        label, paths = value.split("=", 1)
        student, teacher = paths.split(",", 1)
        if not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label):
            raise ValueError("unsafe label")
        return {"label": label, "student": student, "teacher": teacher}
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use LABEL=/absolute/student,/absolute/teacher") from error


def parse_reuse_spec(value):
    try:
        label, source = value.split("=", 1)
        safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not label or not source or any(c not in safe for c in label + source):
            raise ValueError("unsafe label")
        return label, source
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use LABEL=EARLIER_LABEL") from error


def apply_trajectory_reuse(pairs, specs):
    """Validate and record exact trajectory/student-score reuse dependencies."""
    by_label = {pair["label"]: pair for pair in pairs}
    positions = {pair["label"]: index for index, pair in enumerate(pairs)}
    for label, source in specs or []:
        if label not in by_label or source not in by_label:
            raise ValueError(f"Unknown trajectory reuse label: {label}={source}")
        if label == source or positions[source] >= positions[label]:
            raise ValueError(f"Trajectory source must precede its target: {label}={source}")
        if "trajectory_source" in by_label[label]:
            raise ValueError(f"Duplicate trajectory reuse target: {label}")
        if by_label[label]["student"] != by_label[source]["student"]:
            raise ValueError(f"Trajectory reuse requires the exact same student path: {label}={source}")
        by_label[label]["trajectory_source"] = source


def copy_reused_student_outputs(root, pair):
    """Copy immutable trajectories and student logits from an earlier pair."""
    source = root / pair["trajectory_source"]
    target = root / pair["label"]
    required = [source / "trajectories.json", *sorted(source.glob("trajectory-*.json")),
                *sorted(source.glob("student-topk-*.npz"))]
    if not required or any(not path.is_file() for path in required):
        raise ValueError(f"Incomplete trajectory source: {source}")
    for path in required:
        shutil.copy2(path, target / path.name)
    print(f"{pair['label']} reused trajectories and student scores from {pair['trajectory_source']}", flush=True)


def main():
    # Internal workers consume the immutable run config written by the parent.
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        config = read_json(sys.argv[2])
        pair = next(p for p in config["pairs"] if p["label"] == sys.argv[3])
        directory = Path(sys.argv[2]).parent / pair["label"]
        if sys.argv[4] == "generate":
            generate(config, pair, directory)
        else:
            score(config, pair, directory, sys.argv[4])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "analyze-existing":
        analysis_parser = argparse.ArgumentParser(
            description="Compute CPU-only mechanism diagnostics from one completed fixed-overlap evaluation."
        )
        analysis_parser.add_argument("--input-root", type=Path, required=True)
        analysis_parser.add_argument("--output-root", type=Path)
        analysis_args = analysis_parser.parse_args(sys.argv[2:])
        input_root = analysis_args.input_root.resolve()
        output_root = analysis_args.output_root
        if output_root is None:
            output_root = input_root.parent / f"{input_root.name}-mechanism-analysis"
        config_path = input_root / "config.json"
        if not config_path.is_file():
            analysis_parser.error(f"Missing evaluation config: {config_path}")
        config = read_json(config_path)
        analyze_existing(config, input_root, output_root, require_completed=True)
        print(f"Completed: {Path(output_root).resolve() / 'summary.md'}")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=parse_pair, action="append", required=True)
    parser.add_argument("--reuse-trajectories", type=parse_reuse_spec, action="append", default=[],
                        metavar="LABEL=EARLIER_LABEL",
                        help="Reuse exact trajectories and student top-k scores from an earlier pair with the same student.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--questions-per-task", type=int, required=True)
    parser.add_argument("--samples-per-question", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--generation-top-k", type=int, required=True)
    parser.add_argument("--ks", type=int, nargs="+", required=True)
    parser.add_argument("--score-chunk-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and selection without loading weights")
    args = parser.parse_args()
    if len({p["label"] for p in args.pair}) != len(args.pair):
        parser.error("Pair labels must be unique")
    try:
        apply_trajectory_reuse(args.pair, args.reuse_trajectories)
    except ValueError as error:
        parser.error(str(error))
    if min(args.questions_per_task, args.samples_per_question, args.max_tokens, args.score_chunk_size, *args.ks) < 1:
        parser.error("Counts and diagnostic k must be positive")
    if (not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1
            or (args.generation_top_k != -1 and args.generation_top_k < 1)):
        parser.error("Invalid sampling configuration")
    total_responses = len(TASKS) * args.questions_per_task * args.samples_per_question
    if not 0 <= args.seed <= 2**32 - total_responses:
        parser.error("Seed must allow nonnegative 32-bit per-response seeds")
    if args.output_root.exists():
        parser.error("Output root already exists; choose a new directory. Existing results are never overwritten.")
    manifest = select_samples(args.data_dir, args.questions_per_task, args.seed)
    config = {"pairs": args.pair, "samples": manifest["samples"], "seed": args.seed,
              "samples_per_question": args.samples_per_question,
              "max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
              "generation_top_k": args.generation_top_k, "ks": sorted(set(args.ks)),
              "score_chunk_size": args.score_chunk_size, "thinking": True,
              "scoring_temperature": 1.0, "source_sha256": file_hash(__file__)}
    config["versions"] = {name: importlib.metadata.version(name)
                          for name in ("torch", "transformers", "vllm", "numpy", "pandas", "pyarrow")}
    preflight(config)
    print(json.dumps({"config": config, "data_sources": manifest["sources"]}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        parser.error("Set CUDA_VISIBLE_DEVICES to exactly one explicitly selected idle physical GPU")
    args.output_root.mkdir(parents=True, exist_ok=False)
    write_json(args.output_root / "samples.json", manifest)
    write_json(args.output_root / "config.json", config)
    for pair in config["pairs"]:
        (args.output_root / pair["label"]).mkdir()
        if pair.get("trajectory_source"):
            source_pair = next(p for p in config["pairs"] if p["label"] == pair["trajectory_source"])
            if pair["tokenizer_sha256"] != source_pair["tokenizer_sha256"]:
                raise ValueError("Trajectory reuse tokenizer mismatch")
            copy_reused_student_outputs(args.output_root, pair)
            phases = ("teacher",)
        else:
            phases = ("generate", "student", "teacher")
        for phase in phases:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker",
                            str((args.output_root / "config.json").resolve()), pair["label"], phase], check=True)
    summarize(config, args.output_root)
    analyze_existing(config, args.output_root, args.output_root / "mechanism-analysis", require_completed=False)
    write_json(args.output_root / "COMPLETED.json", {"complete": True, "pairs": len(config["pairs"])})
    print(f"Completed: {args.output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
