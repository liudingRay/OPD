#!/bin/bash
# Evaluate one Qwen3-14B-Base benchmark with avg@16 on one A100 GPU.
# Submit each benchmark as an independent job:
#   sbatch --export=ALL,TASK=AIME24 submit_eval_qwen3_14b_base_task_1xa100.sbatch
#   sbatch --export=ALL,TASK=AIME25 submit_eval_qwen3_14b_base_task_1xa100.sbatch
#   sbatch --export=ALL,TASK=AMC23  submit_eval_qwen3_14b_base_task_1xa100.sbatch
#
# This script generates only its selected JSONL. After all three jobs complete,
# run the grading commands printed below to score and summarize the full set.
#SBATCH --job-name=eval-qwen3-14b-base
#SBATCH --account=EUHPC_D33_157
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
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
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$PROJECT_DIR/scripts/val/eval/justrl_eval_outputs/qwen3-14b-base-no-thinking-n16}
CONDA_ENV=${CONDA_ENV:-$WORK/$USER/conda_envs/opd-cu118}

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
GENERATOR_ARGS=(
    --model-root "$PROJECT_DIR/model"
    --data-dir "$PROJECT_DIR/scripts/val/data"
    --output-root "$EVAL_OUTPUT_ROOT"
    --gpu-ids 0
    --serial
    --tasks "$TASK"
    --model "Qwen3-14B-Base=$PROJECT_DIR/model/Qwen3-14B-Base"
)
if [[ "${EVAL_OVERWRITE:-0}" == "1" ]]; then
    GENERATOR_ARGS+=(--overwrite)
fi
python eval_baselines.py "${GENERATOR_ARGS[@]}"

echo "Generation complete for $TASK. After all three task jobs finish, run:"
echo "  cd $PROJECT_DIR/scripts/val/eval"
echo "  python grade.py --eval-dir $EVAL_OUTPUT_ROOT/Qwen3-14B-Base --length-tokenizer $PROJECT_DIR/model/Qwen3-14B-Base"
echo "  python summarize_baseline_results.py --output-root $EVAL_OUTPUT_ROOT"
