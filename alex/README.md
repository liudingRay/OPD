# Alex curriculum OPD

This directory contains the NHR@FAU Alex launcher for the three-stage
curriculum OPD experiment. The training behavior and hyperparameters match
`puhui/curriculum-opd`; only storage, CUDA/cache setup, and job allocation are
adapted for Alex.

## SSH login troubleshooting

If `ssh alex` returns the following error, DNS has selected an unavailable
frontend (`10.28.52.21`, `alex1`):

```text
Not allowed at this time
kex_exchange_identification: Connection closed by remote host
```

Connect through the healthy `alex2` frontend instead:

```bash
ssh -o HostName=10.28.52.22 -o HostKeyAlias=alex.nhr.fau.de alex
```

## Required layout

The launcher resolves the workspace with `ws_find opd` and expects this layout:

```text
/anvme/workspace/b270dd10-opd/
├── OPD/                         # Git checkout
├── envs/opd-py312/             # verified uv Python 3.12 environment
├── models/
│   ├── Qwen3-1.7B/
│   ├── Qwen3-4B/
│   ├── Qwen3-8B/
│   ├── Qwen3-14B/
│   ├── Qwen3-1.7B-OPD-4B/      # merged output consumed by stage2
│   └── Qwen3-1.7B-OPD-4B-8B/   # merged output consumed by stage3
├── experiments/OPD/             # checkpoints, validation data, and logs
└── .cache/opd/                  # HF, FlashInfer, vLLM, Triton, and W&B caches
```

Every model directory must be a complete Hugging Face model containing
`config.json` and `model*.safetensors`. Symlinks to complete models elsewhere
on `/anvme` are accepted.

## Validate without submitting

The preflight checks all stage-specific model and dataset paths before a job is
submitted:

```bash
cd "$(ws_find opd)/OPD"
DRY_RUN=1 bash alex/curriculum-opd stage1
```

Until `models/Qwen3-4B` exists, the stage1 preflight intentionally fails before
requesting GPUs.

## Submit training

Run the same file with `bash`. On a login node it submits itself through
`sbatch`; inside the allocation it starts the training driver.

```bash
cd "$(ws_find opd)/OPD"
bash alex/curriculum-opd stage1
```

The job requests one Alex node with four A100-80GB GPUs and 32 CPUs for 24
hours. Repeat with `stage2` and `stage3` after merging the preceding stage.

## One-step smoke test

Before the full 24-hour submission, request a shorter backfill-friendly job:

```bash
SMOKE=1 bash alex/curriculum-opd stage1
```

Smoke mode requests two A100-80GB GPUs and 16 CPUs while keeping the production
batch, rollout, sequence-length, sampling, and OPD settings. It changes verl's
`trainer.n_gpus_per_node` to two, limits the dataset to one 32-prompt batch,
runs exactly one complete optimizer step, requests two hours, disables
checkpoint/resume, and writes to an independent experiment and W&B run whose
name ends in `-smoke`. It cannot overwrite or resume the formal stage1 run.

Online W&B logging is enabled by default. Alex currently has no saved W&B
credential for this account, so export the API key in the same login shell
before submission. Do not put it in this repository or in a Slurm script:

```bash
read -rs WANDB_API_KEY
export WANDB_API_KEY
echo
bash alex/curriculum-opd stage1
unset WANDB_API_KEY
```

The launcher refuses to request GPUs when `WANDB_MODE=online` and neither an
environment key nor a non-empty `~/.netrc` is available. To run without live
upload, opt into offline logging explicitly:

```bash
WANDB_MODE=offline bash alex/curriculum-opd stage1
```

Monitor the submitted job with:

```bash
squeue -u "$USER"
tail -f "$(ws_find opd)/experiments/OPD/logs/curriculum-opd-stage1-<job-id>.out"
```

## Merge one FSDP actor checkpoint

The merge launcher defaults to the quick-80 Qwen3-8B-teacher run at step 300.
It verifies the checkpoint tracker and actor files, requests one A100 because
Alex does not accept GPU-partition jobs without a GPU GRES, writes into a
job-specific staging directory, and publishes the Hugging Face model only
after the merge has completed successfully:

```bash
cd "$(ws_find opd)/OPD"
bash alex/merge-fsdp-actor
```

For another run, override the checkpoint root, step, and output model name
together:

```bash
RUN_DIR="$(ws_find opd)/experiments/OPD/checkpoints/<experiment-name>" \
EXPECTED_STEP=400 \
MERGED_MODEL_NAME=Qwen3-1.7B-OPD-example-step400 \
bash alex/merge-fsdp-actor
```

The launcher never replaces an existing model directory. Use `DRY_RUN=1` to
validate the selected checkpoint and print the configuration without submitting
a job.

## Evaluate one merged model

The reusable evaluation launcher runs AIME24, AIME25, and AMC23 with thinking
enabled and 16 samples per question by default. It validates the model and
benchmark files on the login node before requesting one A100-80GB GPU:

```bash
cd "$(ws_find opd)/OPD"
bash alex/eval-qwen3-model-1xa100
```

Select another merged model with a unique label and its absolute path:

```bash
MODEL_NAME=Qwen3-1.7B-OPD-example-step400 \
MODEL_PATH="$(ws_find opd)/models/Qwen3-1.7B-OPD-example-step400" \
bash alex/eval-qwen3-model-1xa100
```

The default output is
`$(ws_find opd)/experiments/OPD/evaluation/<MODEL_NAME>-thinking-n16`. Existing
generation files are reused; set `EVAL_OVERWRITE=1` only when they should be
regenerated deliberately. Use `DRY_RUN=1` to print and validate the complete
configuration without submitting a job.

## Fixed-question curriculum overlap evaluation

Use the same shared evaluator as `puhui/eval-curriculum-overlap`:

```bash
cd "$(ws_find opd)/OPD"
bash alex/eval-curriculum-overlap
```

The launcher validates inputs on the login node before submitting one A100-80GB,
8 CPUs, and 24 hours. `DRY_RUN=1 bash alex/eval-curriculum-overlap` performs only
the CPU preflight and does not submit a job. The default final students are
`models/Qwen3-1.7B-OPD-4B`, `models/Qwen3-1.7B-OPD-4B-8B`, and
`models/Qwen3-1.7B-OPD-4B-8B-14B`, paired with `models/Qwen3-4B`,
`models/Qwen3-8B`, and `models/Qwen3-14B`, respectively, under `$(ws_find opd)`.
Override `STAGE1_MODEL_PATH`, `STAGE2_MODEL_PATH`, `STAGE3_MODEL_PATH`,
`TEACHER4B_PATH`, `TEACHER8B_PATH`, or `TEACHER14B_PATH` for other verified paths.

The settings match Puhui: five fixed questions per AIME24/AIME25/AMC23 (`SEED=42`),
one thinking response per question, `MAX_TOKENS=38912`, temperature 0.6, top-p
0.95, generation top-k 20, and diagnostic k=4/8/16. This is forward-only;
there is no training or backward pass. Each student generates its own trajectory,
and its teacher scores those same prefixes; this is not a fixed-prefix comparison
across all students or a start/middle/end checkpoint sweep.

Results go to `$(ws_find opd)/experiments/OPD/evaluation/curriculum-overlap-fixed`.
An existing output directory is rejected; set a new `OVERLAP_OUTPUT_ROOT` for a
repeat run. Outputs include `samples.json`, generated token IDs, per-token
metrics, `summary.md/csv/json`, and `summary_chunks.md/csv/json` using the original
1024-response-token bins. Chunk reports include both all-valid-position overlap
mass and the legacy nonempty-intersection mass mean, plus contributing question
and token counts. `SCORE_CHUNK_SIZE=128` controls forward-pass memory only.
The shared file `scripts/val/eval/eval_fixed_overlap.py` must be present on Alex.
Logs are saved to `experiments/OPD/logs/eval-curriculum-overlap-<job-id>.out/.err`.

## Operational behavior

- All behavior-changing training controls are inherited from the puhui
  curriculum launcher, including thinking mode, 15,360-token responses,
  four rollouts, top-k/top-p sampling, OPD top-16 support, checkpoint cadence,
  retention, W&B identity, and auto-resume.
- FlashInfer, vLLM, Triton, TorchInductor, Hugging Face, NumPy/Numba, CUDA, and
  W&B caches are routed under the OPD workspace. Nothing intentionally writes
  runtime caches to the over-quota `$HOME`.
- Slurm captures the complete job from its first command in separate `.out`
  and `.err` files. The training section is also copied to a timestamped log
  under `experiments/OPD/logs`, and its final line records success or the exact
  nonzero exit code.
- Slurm sends `USR1` twenty minutes before the time limit. The launcher writes
  the existing verl timeout-save marker so the trainer requests a checkpoint at
  the next safe step boundary.
- Training and FSDP merge remain separate operations. This launcher never
  merges or overwrites a model directory.
