#!/usr/bin/env python3
"""Generate and score IFEval responses with a local Hugging Face/vLLM model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from grade_ifeval import DEFAULT_DATA_FILE, load_dataset, score_file, validate_scorer_environment


@dataclass(frozen=True)
class IFEvalConfig:
    model_name: str
    model_path: str
    data_file: str
    output_root: str
    gpu_ids: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str
    trust_remote_code: bool
    enable_thinking: bool
    num_samples: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    max_tokens: int
    seed: int
    max_examples: int | None
    overwrite: bool
    score_after_generation: bool


def validate_config(config: IFEvalConfig) -> None:
    model_path = Path(config.model_path)
    data_file = Path(config.data_file)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", config.model_name):
        raise ValueError("--model-name must use only letters, digits, dots, underscores, and hyphens.")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model directory is missing config.json: {model_path}")
    if not data_file.is_file():
        raise FileNotFoundError(f"IFEval data file does not exist: {data_file}")
    if config.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if config.max_examples is not None and config.max_examples <= 0:
        raise ValueError("--max-examples must be positive when provided.")
    if config.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if not 0 < config.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if not 0 <= config.min_p <= 1:
        raise ValueError("--min-p must be in [0, 1].")
    if config.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if config.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if config.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive.")
    if not 0 < config.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1].")
    visible_gpu_ids = [item.strip() for item in config.gpu_ids.split(",") if item.strip()]
    if len(visible_gpu_ids) < config.tensor_parallel_size:
        raise ValueError(
            f"Received {len(visible_gpu_ids)} GPU IDs for tensor_parallel_size={config.tensor_parallel_size}."
        )


def _format_number(value: float) -> str:
    return format(value, "g")


def output_path(config: IFEvalConfig) -> Path:
    thinking_label = "thinking" if config.enable_thinking else "nonthinking"
    limit_label = f"-limit{config.max_examples}" if config.max_examples is not None else ""
    filename = (
        f"ifeval-{thinking_label}-t{_format_number(config.temperature)}-p{_format_number(config.top_p)}"
        f"-topk{config.top_k}-minp{_format_number(config.min_p)}-n{config.num_samples}"
        f"-mnt{config.max_tokens}-seed{config.seed}{limit_label}.jsonl"
    )
    return Path(config.output_root) / config.model_name / filename


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stop_token_ids(tokenizer: Any) -> list[int]:
    token_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, int):
            token_ids.append(tokenizer.eos_token_id)
        else:
            token_ids.extend(tokenizer.eos_token_id)
    for stop_token in ("<|im_end|>", "<|endoftext|>"):
        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
        if len(encoded) == 1:
            token_ids.append(encoded[0])
    return list(dict.fromkeys(token_ids))


def build_chat_messages(prompt: str) -> list[dict[str, str]]:
    """Return the exact zero-shot user message without benchmark-specific suffixes."""
    return [{"role": "user", "content": prompt}]


def generate(config: IFEvalConfig, response_path: Path) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids
    from vllm import LLM, SamplingParams

    dataset_rows = load_dataset(Path(config.data_file))
    if config.max_examples is not None:
        if config.max_examples > len(dataset_rows):
            raise ValueError(
                f"Requested {config.max_examples} examples, but the dataset contains only {len(dataset_rows)}."
            )
        dataset_rows = dataset_rows[: config.max_examples]

    llm = LLM(
        model=config.model_path,
        trust_remote_code=config.trust_remote_code,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
        dtype=config.dtype,
        seed=config.seed,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            build_chat_messages(row["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )
        for row in dataset_rows
    ]
    stop_token_ids = _stop_token_ids(tokenizer)
    records = []
    try:
        for sample_id in range(config.num_samples):
            sample_seed = config.seed + sample_id
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                max_tokens=config.max_tokens,
                stop_token_ids=stop_token_ids or None,
                seed=sample_seed,
            )
            print(
                f"Generating IFEval sample {sample_id + 1}/{config.num_samples} "
                f"for {len(dataset_rows)} prompts with seed={sample_seed}",
                flush=True,
            )
            outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
            if len(outputs) != len(dataset_rows):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(dataset_rows)} prompts.")
            for example_id, (row, output) in enumerate(zip(dataset_rows, outputs)):
                completion = output.outputs[0]
                records.append(
                    {
                        "example_id": example_id,
                        "key": row["key"],
                        "sample_id": sample_id,
                        "prompt": row["prompt"],
                        "instruction_id_list": row["instruction_id_list"],
                        "kwargs": row["kwargs"],
                        "seed": sample_seed,
                        "response": completion.text,
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "generated_tokens": len(completion.token_ids),
                    }
                )
    finally:
        del llm

    expected = len(dataset_rows) * config.num_samples
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} generations, got {len(records)}.")
    _atomic_write_jsonl(response_path, records)
    print(f"Saved {len(records)} generations to {response_path}", flush=True)
    return len(dataset_rows)


def _print_main_metrics(summary: dict[str, Any]) -> None:
    metrics = summary["main_metrics"]
    print(f"prompt_level_strict_acc={metrics['prompt_level_strict_acc']:.6f}")
    print(f"prompt_level_loose_acc={metrics['prompt_level_loose_acc']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent OPD IFEval generation and scoring pipeline.")
    parser.add_argument("--model-name", required=True, help="Short label used in the output directory.")
    parser.add_argument("--model-path", type=Path, required=True, help="Absolute local Hugging Face model path.")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0", help="CUDA-visible IDs, comma-separated for tensor parallelism.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action="store_true", help="Disabled by default.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Smoke-test prefix; never use for final results.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true", help="Generate only; run grade_ifeval.py later.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = IFEvalConfig(
        model_name=args.model_name,
        model_path=str(args.model_path.resolve()),
        data_file=str(args.data_file.resolve()),
        output_root=str(args.output_root.resolve()),
        gpu_ids=args.gpu_ids,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        enable_thinking=args.enable_thinking,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_examples=args.max_examples,
        overwrite=args.overwrite,
        score_after_generation=not args.skip_scoring,
    )
    validate_config(config)
    load_dataset(Path(config.data_file))
    if config.score_after_generation:
        validate_scorer_environment()

    response_path = output_path(config)
    summary_path = response_path.with_suffix(".summary.json")
    details_path = response_path.with_suffix(".results.jsonl")
    config_path = response_path.with_suffix(".config.json")

    generated = False
    if response_path.exists() and not config.overwrite:
        print(f"Reusing existing generations: {response_path}", flush=True)
    else:
        generate(config, response_path)
        generated = True
        config_payload = {
            **asdict(config),
            "data_sha256": _sha256(Path(config.data_file)),
            "response_file": str(response_path),
        }
        _atomic_write_json(config_path, config_payload)

    if not config.score_after_generation:
        return
    if summary_path.exists() and details_path.exists() and not config.overwrite and not generated:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        print(f"Reusing existing scores: {summary_path}", flush=True)
        _print_main_metrics(summary)
        return

    _, _, summary = score_file(
        response_path,
        data_file=Path(config.data_file),
        summary_file=summary_path,
        details_file=details_path,
        allow_partial=config.max_examples is not None,
        overwrite=config.overwrite or generated,
    )
    _print_main_metrics(summary)
    print(f"Summary: {summary_path}")
    print(f"Per-response details: {details_path}")


if __name__ == "__main__":
    main()
