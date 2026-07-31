#!/usr/bin/env python3
"""Run the OPD/JustRL math evaluation protocol for several local baselines.

Each model is assigned to one GPU.  The default registry evaluates the four
baselines in ``<project>/model`` used by the Leonardo batch submission script.
It can either run the models concurrently on separate GPUs or serially on one
GPU, while keeping every model's outputs in a separate directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
DEFAULT_MODELS = (
    "DeepSeek-R1-Distill-Qwen-1.5B",
    "JustRL-DeepSeek-1.5B",
    "Qwen3-4B-Base",
    "Qwen3-8B-Base",
)
DEFAULT_TASKS = ("AIME24", "AIME25", "AMC23")


@dataclass(frozen=True)
class EvalConfig:
    label: str
    model_path: str
    gpu_id: str
    data_dir: str
    output_root: str
    tasks: tuple[str, ...]
    num_samples: int
    temperature: float
    top_p: float
    max_tokens: int
    enable_thinking: bool
    overwrite: bool


def parse_model_spec(value: str) -> tuple[str, str]:
    """Parse LABEL=PATH passed through --model."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model specs must use LABEL=PATH.")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Both LABEL and PATH must be non-empty.")
    return label, path


def load_samples(filepath: Path) -> list[dict[str, object]]:
    """Load one benchmark in the same format as the authors' generator."""
    df = pd.read_parquet(filepath)
    samples = [
        {
            "example_id": index,
            "question": df.at[index, "prompt"][0]["content"].strip(),
            "answer": df.at[index, "reward_model"]["ground_truth"].strip(),
        }
        for index in range(len(df))
    ]
    print(f"Loaded {len(samples)} unique samples from {filepath}", flush=True)
    return samples


def validate_config(config: EvalConfig) -> None:
    model_path = Path(config.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"{config.label}: model directory does not exist: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"{config.label}: {model_path} is not a Hugging Face/vLLM model directory (missing config.json)."
        )
    for task_name in config.tasks:
        task_path = Path(config.data_dir) / task_name / "test.parquet"
        if not task_path.is_file():
            raise FileNotFoundError(f"Missing benchmark file: {task_path}")


def output_path(config: EvalConfig, task_name: str) -> Path:
    filename = (
        f"{task_name.lower()}_t{config.temperature}_p{config.top_p}_n{config.num_samples}"
        f"-MNT{config.max_tokens}.jsonl"
    )
    return Path(config.output_root) / config.label / filename


def run_model(config: EvalConfig) -> dict[str, object]:
    """Evaluate one model on one GPU and return a compact completion record."""
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    validate_config(config)

    print(
        f"[GPU {config.gpu_id}] Starting {config.label}: model={config.model_path}, "
        f"thinking={config.enable_thinking}",
        flush=True,
    )
    llm = LLM(
        model=config.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    tokenizer = llm.get_tokenizer()
    stop_token_ids = []
    for stop_token in ("<|im_end|>", "<|endoftext|>"):
        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
        if encoded:
            stop_token_ids.append(encoded[0])

    completed_tasks = []
    try:
        for task_name in config.tasks:
            out_path = output_path(config, task_name)
            if out_path.exists() and not config.overwrite:
                print(f"[GPU {config.gpu_id}] {config.label}: reusing {out_path}", flush=True)
                completed_tasks.append(task_name)
                continue

            samples = load_samples(Path(config.data_dir) / task_name / "test.parquet")
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=sample["question"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=config.enable_thinking,
                )
                for sample in samples
            ]
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                stop_token_ids=stop_token_ids or None,
            )
            results = []
            for rollout_id in tqdm(
                range(config.num_samples),
                desc=f"{config.label} {task_name}",
                position=int(config.gpu_id) if config.gpu_id.isdigit() else 0,
                leave=False,
            ):
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                for sample, output in zip(samples, outputs, strict=True):
                    results.append(
                        {
                            "example_id": sample["example_id"],
                            "question": sample["question"],
                            "prompt": PROMPT_TEMPLATE.format(problem=sample["question"]),
                            "answer": sample["answer"],
                            "seed": rollout_id,
                            "response": output.outputs[0].text,
                        }
                    )

            expected = len(samples) * config.num_samples
            if len(results) != expected:
                raise RuntimeError(
                    f"{config.label} {task_name}: expected {expected} completions, got {len(results)}."
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = out_path.with_suffix(".jsonl.tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                for item in results:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            temporary_path.replace(out_path)
            print(f"[GPU {config.gpu_id}] Saved {expected} completions to {out_path}", flush=True)
            completed_tasks.append(task_name)
    finally:
        del llm

    return {"label": config.label, "gpu_id": config.gpu_id, "tasks": completed_tasks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local baseline models with one vLLM worker per GPU.")
    parser.add_argument("--model-root", type=Path, default=Path("../../model"))
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        metavar="LABEL=PATH",
        help="Override the default model registry. May be supplied multiple times.",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3", help="Comma-separated GPU IDs, one per model.")
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Evaluate all models sequentially on the single GPU specified by --gpu-ids.",
    )
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, default=list(DEFAULT_TASKS))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=31744)
    parser.add_argument("--enable-thinking", action="store_true", help="Disabled by default for this baseline run.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate result files that already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    gpu_ids = [gpu_id.strip() for gpu_id in args.gpu_ids.split(",") if gpu_id.strip()]
    model_specs = args.model or [(label, str(args.model_root / label)) for label in DEFAULT_MODELS]
    if args.serial:
        if len(gpu_ids) != 1:
            raise ValueError("--serial requires exactly one GPU ID.")
        assigned_gpu_ids = gpu_ids * len(model_specs)
    elif len(model_specs) != len(gpu_ids):
        raise ValueError(f"Received {len(model_specs)} model(s) but {len(gpu_ids)} GPU ID(s); they must match.")
    else:
        assigned_gpu_ids = gpu_ids
    labels = [label for label, _ in model_specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Model labels must be unique because they define output directories.")

    configs = [
        EvalConfig(
            label=label,
            model_path=str(Path(model_path).resolve()),
            gpu_id=gpu_id,
            data_dir=str(args.data_dir.resolve()),
            output_root=str(args.output_root.resolve()),
            tasks=tuple(args.tasks),
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            overwrite=args.overwrite,
        )
        for (label, model_path), gpu_id in zip(model_specs, assigned_gpu_ids, strict=True)
    ]
    for config in configs:
        validate_config(config)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "evaluation_config.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(config) for config in configs], handle, indent=2)

    context = multiprocessing.get_context("spawn")
    if args.serial:
        for config in configs:
            # Use a fresh process for each model so vLLM/CUDA state from the
            # previous model cannot remain attached to the allocated GPU.
            with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
                future = executor.submit(run_model, config)
                try:
                    print(f"Completed: {future.result()}", flush=True)
                except Exception as exc:
                    print(f"FAILED: {config.label} on GPU {config.gpu_id}: {exc}", flush=True)
                    traceback.print_exc()
                    raise
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(configs), mp_context=context) as executor:
            futures = {executor.submit(run_model, config): config for config in configs}
            for future in concurrent.futures.as_completed(futures):
                config = futures[future]
                try:
                    print(f"Completed: {future.result()}", flush=True)
                except Exception as exc:
                    print(f"FAILED: {config.label} on GPU {config.gpu_id}: {exc}", flush=True)
                    traceback.print_exc()
                    raise


if __name__ == "__main__":
    main()
