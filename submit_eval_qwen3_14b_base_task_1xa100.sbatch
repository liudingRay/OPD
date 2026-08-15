#!/bin/bash
# Evaluate one Qwen3-14B GRPO benchmark with avg@16 on two A100 GPUs.
# Submit each benchmark as an independent job:
#   sbatch --export=ALL,TASK=AIME24 submit_eval_qwen3_14b_base_task_1xa100.sbatch
#   sbatch --export=ALL,TASK=AIME25 submit_eval_qwen3_14b_base_task_1xa100.sbatch
#   sbatch --export=ALL,TASK=AMC23  submit_eval_qwen3_14b_base_task_1xa100.sbatch
#
# Each GPU generates eight independent samples per problem, then the two shards
# are merged into the standard 16-sample JSONL. After all three jobs complete,
# run the grading commands printed below to score and summarize the full set.
#SBATCH --job-name=eval-qwen3-14b-grpo
#SBATCH --account=EUHPC_D33_157
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.err

set -eo pipefail

TASK=${TASK:-}
case "$TASK" in
    AIME24|AIME25|AMC23) ;;
    *)
        echo "Set TASK to exactly one of: AIME24, AIME25, AMC23."
        echo "Example: sbatch --export=ALL,TASK=AIME24 $0"
        exit 2
        ;;
esac

PROJECT_DIR=${PROJECT_DIR:-/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD}
CONDA_BASE=${CONDA_BASE:-/leonardo/home/userexternal/jbi00000/miniconda3}
OPD_CACHE_ROOT=${OPD_CACHE_ROOT:-/leonardo_work/EUHPC_D33_157/OPD-experiment/.cache/opd}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$PROJECT_DIR/scripts/val/eval/justrl_eval_outputs/qwen3-14b-grpo-no-thinking-n16}
CONDA_ENV=${CONDA_ENV:-$WORK/$USER/conda_envs/opd-cu118}
MODEL_NAME=${MODEL_NAME:-Qwen3-14B-GRPO}

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export OPD_CACHE_ROOT
export TRITON_CACHE_DIR="$OPD_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$OPD_CACHE_ROOT/torchinductor"
export HF_HOME="$OPD_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export OUTLINES_CACHE_DIR="$OPD_CACHE_ROOT/outlines/${SLURM_JOB_ID}"
unset ROCR_VISIBLE_DEVICES

mkdir -p "$PROJECT_DIR/logs" "$EVAL_OUTPUT_ROOT" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$OUTLINES_CACHE_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Benchmark: $TASK"
echo "Results: $EVAL_OUTPUT_ROOT"
nvidia-smi

cd "$PROJECT_DIR/scripts/val/eval"
TASK_LOWER=$(printf '%s' "$TASK" | tr '[:upper:]' '[:lower:]')
MODEL_OUTPUT_DIR="$EVAL_OUTPUT_ROOT/$MODEL_NAME"
FINAL_OUTPUT="$MODEL_OUTPUT_DIR/${TASK_LOWER}_t0.7_p0.95_n16-MNT31744.jsonl"
WORKER_ROOT="$EVAL_OUTPUT_ROOT/.workers/$TASK"
WORKER_ZERO_ROOT="$WORKER_ROOT/gpu0"
WORKER_ONE_ROOT="$WORKER_ROOT/gpu1"
WORKER_ZERO_OUTPUT="$WORKER_ZERO_ROOT/$MODEL_NAME/${TASK_LOWER}_t0.7_p0.95_n8-MNT31744.jsonl"
WORKER_ONE_OUTPUT="$WORKER_ONE_ROOT/$MODEL_NAME/${TASK_LOWER}_t0.7_p0.95_n8-MNT31744.jsonl"

if [[ -f "$FINAL_OUTPUT" && "${EVAL_OVERWRITE:-0}" != "1" ]]; then
    echo "Reusing completed result: $FINAL_OUTPUT"
else
    WORKER_ARGS=(
        --model-root "$PROJECT_DIR/model"
        --data-dir "$PROJECT_DIR/scripts/val/data"
        --serial
        --tasks "$TASK"
        --num-samples 8
        --model "$MODEL_NAME=$PROJECT_DIR/model/$MODEL_NAME"
    )
    if [[ "${EVAL_OVERWRITE:-0}" == "1" ]]; then
        WORKER_ARGS+=(--overwrite)
    fi

    python eval_baselines.py --output-root "$WORKER_ZERO_ROOT" --gpu-ids 0 "${WORKER_ARGS[@]}" &
    WORKER_ZERO_PID=$!
    python eval_baselines.py --output-root "$WORKER_ONE_ROOT" --gpu-ids 1 "${WORKER_ARGS[@]}" &
    WORKER_ONE_PID=$!
    wait "$WORKER_ZERO_PID"
    wait "$WORKER_ONE_PID"

    python merge_eval_rollouts.py \
        --input "$WORKER_ZERO_OUTPUT" \
        --input "$WORKER_ONE_OUTPUT" \
        --output "$FINAL_OUTPUT" \
        --expected-total-samples 16
fi

echo "Generation complete for $TASK. After all three task jobs finish, run:"
echo "  cd $PROJECT_DIR/scripts/val/eval"
printf '  python grade.py --eval-dir %s/%s \\\n' "$EVAL_OUTPUT_ROOT" "$MODEL_NAME"
printf '    --length-tokenizer %s/model/%s\n' "$PROJECT_DIR" "$MODEL_NAME"
echo "  python summarize_baseline_results.py --output-root $EVAL_OUTPUT_ROOT"
