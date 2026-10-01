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

## Equal-weight multi-teacher baseline

The independent multi-teacher baseline starts from the official post-trained
`Qwen3-1.7B` student. Qwen3-4B, Qwen3-8B, and Qwen3-14B score the same student
trajectory, and their token-level OPD rewards are combined with normalized
weights `[1, 1, 1]`. This is a uniform teacher ensemble, not a curriculum stage.

Its formal launcher inherits the quick-80 settings, except that both training
and validation response limits are deliberately reduced to 7680 tokens:

```bash
cd "$(ws_find opd)/OPD"
bash alex/multiteacher-opd-quick80-4xa100
```

Run the bounded end-to-end smoke test before the formal submission:

```bash
SMOKE=1 bash alex/multiteacher-opd-quick80-4xa100
```

The launcher validates the official `Qwen3-1.7B`, `Qwen3-4B`, `Qwen3-8B`, and
`Qwen3-14B` directories under `$(ws_find opd)/models`. Override
`ACTOR_MODEL_PATH`, `TEACHER4B_PATH`, `TEACHER8B_PATH`, or `TEACHER14B_PATH`
only with verified complete Hugging Face model directories. Use `DRY_RUN=1` to
perform the login-node preflight without submitting a job.

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

The default settings match Puhui: five fixed questions per AIME24/AIME25/AMC23
(`SEED=42`), one thinking response per question, `MAX_TOKENS=38912`, temperature
0.6, top-p 0.95, generation top-k 20, and diagnostic k=4/8/16. Set
`QUESTIONS_PER_TASK` and `SAMPLES_PER_QUESTION` explicitly for larger runs. Each
repeated response receives a unique deterministic seed and UID; summaries report
both unique-question and response counts. This is forward-only;
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

New runs also write `mechanism-analysis/` with sparse-union cosine and two
student-weighted absolute log-probability-gap variants. Sparse-union cosine aligns
token IDs and treats probabilities outside each model's own top-k as zero. The
shared-token gap is conditional on a nonempty intersection. The exact
student-top-k gap uses teacher probabilities on every student top-k token, which
the scorer now saves during the existing teacher forward pass.

Completed historical runs can be analyzed CPU-only without loading models or
changing the original result directory:

```bash
cd "$(ws_find opd)/OPD"
source "$(ws_find opd)/envs/opd-py312/bin/activate"
python scripts/val/eval/eval_fixed_overlap.py analyze-existing \
  --input-root "$(ws_find opd)/experiments/OPD/evaluation/direct-opd-overlap-fixed"
```

The default sibling output is `<input-name>-mechanism-analysis`; an existing
target is rejected. Historical files support exact sparse-union cosine and the
shared-token gap. Their exact student-top-k gap is reported as `N/A`, because
they did not save teacher probabilities for student-only top-k tokens. Use
`--output-root` to select another new directory.

To measure only the two curriculum transition points, submit:

```bash
PAIR_SET=curriculum_starts bash alex/eval-curriculum-overlap
```

This mode evaluates `Qwen3-1.7B-OPD-4B-quick80-step280` against Qwen3-8B
(the Stage-2 start) and `Qwen3-1.7B-OPD-4B-8B-quick80-step280` against
Qwen3-14B (the Stage-3 start). It uses the same fixed questions and metrics,
writes to `evaluation/curriculum-overlap-fixed-stage-starts`, and validates only
the four models needed by these two pairs. Override `STAGE2_START_MODEL_PATH` or
`STAGE3_START_MODEL_PATH` when testing other transition checkpoints.

To compare the unchanged official Qwen3-1.7B student with all three teachers:

```bash
PAIR_SET=official_student_teachers bash alex/eval-curriculum-overlap
```

This mode generates the fixed trajectories once with `models/Qwen3-1.7B`,
computes its student logits once, and evaluates Qwen3-4B, Qwen3-8B, and
Qwen3-14B on those exact same prefixes. The output directory is
`evaluation/official-qwen3-1p7b-overlap-fixed-teachers`. This reuse makes the
teacher-scale comparison prefix controlled and avoids two redundant student
generation/scoring passes. Override `OFFICIAL_STUDENT_PATH` if necessary.

To evaluate the three independent teacher-scale controls, each starting from the
official Qwen3-1.7B student rather than a preceding curriculum stage:

```bash
PAIR_SET=direct_opd_students bash alex/eval-curriculum-overlap
```

This pairs `Qwen3-1.7B-OPD-4B-quick80-step280`,
`Qwen3-1.7B-OPD-8B-quick80-step300`, and
`Qwen3-1.7B-OPD-14B-quick80-step300` with their respective official teachers.
Override `DIRECT_OPD4B_PATH`, `DIRECT_OPD8B_PATH`, or `DIRECT_OPD14B_PATH` for
other independent runs. These pairs are controls, not curriculum stages.

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

## Dynamic multi-teacher OPD (LN-softmax)

The Alex-only launch integration below uses the shared algorithm under `verl/`.
It inherits the uniform-three-teacher baseline's data, 7680-token train/validation
response limits, two rollouts, two epochs, seed, checkpoint cadence, and allocation.
It starts from official post-trained Qwen3-1.7B with frozen 4B/8B/14B teachers;
this is one simultaneous multi-teacher run, not sequential curriculum stages.

```bash
cd "$(ws_find opd)/OPD"
DRY_RUN=1 bash alex/dynamic-multiteacher-opd-quick80-4xa100
# Submit a bounded smoke test explicitly after preflight:
SMOKE=1 bash alex/dynamic-multiteacher-opd-quick80-4xa100
# Formal run (explicit submission):
bash alex/dynamic-multiteacher-opd-quick80-4xa100
```

`TEACHER_EMA_BETA=0.9` and `TEACHER_SELECTION_TAU=0.1` are configurable.
The generated experiment family is `multiteacher-ln-softmax-b0.9-tau0.1`;
checkpoint and W&B identities are separate from uniform3. Formal runs auto-resume
only their own experiment directory. Smoke runs disable resume and checkpointing.
Do not override `EXPERIMENT_NAME` to an existing baseline name. There is no
cross-step EMA state to checkpoint: each newly sampled trajectory initializes
L from its first valid response token and resets independently.

At each valid thinking/answer token, overlap mass sums the student's original
full-vocabulary probabilities for shared top-K IDs. L is its trajectory EMA.
N is JS/ln(2) after conditioning both models on the same student candidate IDs.
Teacher weights are softmax(L*N/tau), without a fixed prior. Detached weights
mix the original per-teacher OPD rewards using full-vocabulary log probabilities;
the existing candidate weighting, advantages and clipped actor loss are unchanged.
Padding does not update EMA. Three zero utilities yield uniform weights.

Per-teacher logs contain mean/p10/p50/p90 of `overlap_mass`, `learnability`, `js`,
`utility`, and `weight`; `teacher/weight_entropy` measures selection concentration.
`teacher/entropy` uses dynamic weights in this mode, explicitly marked by
`teacher/entropy_uses_dynamic_weights`. Original single-teacher and fixed-weight
modes remain available. Leonardo/Puhui launchers are not changed for this experiment.
GPU smoke validation is required before claiming distributed training is verified;
CPU numerical tests alone do not establish GPU memory sufficiency or throughput.

Local validation (2026-09-30): 22 CPU tests passed (7 existing fixed-weight
regressions and 15 dynamic tests). The numerical suite also executes the actual
worker candidate-scoring and actor reward methods in isolation, checking batch
partition invariance. Optional distributed imports were bypassed for these CPU
tests; Ray/FSDP transport itself was not exercised. Shell syntax and temporary
fixture preflights passed for both uniform/dynamic formal and smoke launchers.
The temporary fixture only checks launcher configuration, not real model files.
No Alex GPU job was submitted, and no checkpoint was merged.

For numerical smoke auditing, set `OPD_DYNAMIC_AUDIT_DIR` to a new absolute
output directory when submitting the smoke. The trainer saves the first four
complete trajectories' real scoring tensors, masks, diagnostics and mixed rewards
as `step-<step>.pt` before the actor update. Existing files are rejected. Leave
this variable unset for normal training. Independently recompute the dump using:

```bash
python verl/tests/utils/audit_dynamic_teacher.py /absolute/audit/directory/step-1.pt
```

This audit uses CPU float64 and a sequential masked EMA, without importing the
production weight function. It writes a JSON report with error tolerances and
max/mean errors. It checks arithmetic on actual scoring outputs; it does not
prove model quality or reproduce full-vocabulary model forwards independently.

GPU follow-up (2026-09-30): job 4419300 completed one full step on four
A100-80GB GPUs (a0532), exit 0, wall time 9m01s. Float64 independent audit of
four real trajectories / 12,708 valid response tokens passed; maximum absolute
errors were 7.02e-7 for teacher weights and 2.69e-6 for mixed rewards. Audit
artifacts are under `experiments/OPD/dynamic-audit-20260930`. A W&B atexit
BrokenPipeError appeared after training completed; the job still exited 0.
This verifies the smoke path, not long-run stability or model-quality gains.
