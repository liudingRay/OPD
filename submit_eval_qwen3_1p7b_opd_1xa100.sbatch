#!/bin/bash
# Evaluate a merged Qwen3-1.7B OPD model on AIME24, AIME25, and AMC23
# with avg@16. Override MODEL_NAME/MODEL_PATH for each teacher run.
# Submit from the inner OPD checkout with:
#   sbatch submit_eval_qwen3_1p7b_opd_1xa100.sbatch
#SBATCH --job-name=eval-qwen3-1p7b-opd
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

PROJECT_DIR=${PROJECT_DIR:-/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD}
CONDA_BASE=${CONDA_BASE:-/leonardo/home/userexternal/jbi00000/miniconda3}
OPD_CACHE_ROOT=${OPD_CACHE_ROOT:-/leonardo_work/EUHPC_D33_157/OPD-experiment/.cache/opd}
MODEL_NAME=${MODEL_NAME:-Qwen3-1.7B-opd-4B}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/model/$MODEL_NAME}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$PROJECT_DIR/scripts/val/eval/justrl_eval_outputs/$MODEL_NAME-thinking-n16}
ENABLE_THINKING=${ENABLE_THINKING:-1}
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
echo "Results: $EVAL_OUTPUT_ROOT"
nvidia-smi

cd "$PROJECT_DIR/scripts/val/eval"
GENERATOR_ARGS=(
    --model-root "$PROJECT_DIR/model"
    --data-dir "$PROJECT_DIR/scripts/val/data"
    --output-root "$EVAL_OUTPUT_ROOT"
    --gpu-ids 0
    --serial
    --model "$MODEL_NAME=$MODEL_PATH"
)
# Shell environment variables are strings. Accept common true/false spellings
# explicitly instead of silently treating ENABLE_THINKING=true as false.
case "$ENABLE_THINKING" in
    1|true|True|TRUE) GENERATOR_ARGS+=(--enable-thinking) ;;
    0|false|False|FALSE) ;;
    *)
        echo "ENABLE_THINKING must be one of: 1, true, 0, false; got: $ENABLE_THINKING" >&2
        exit 2
        ;;
esac
if [[ "${EVAL_OVERWRITE:-0}" == "1" ]]; then
    GENERATOR_ARGS+=(--overwrite)
fi
python eval_baselines.py "${GENERATOR_ARGS[@]}"

python grade.py \
    --eval-dir "$EVAL_OUTPUT_ROOT/$MODEL_NAME" \
    --length-tokenizer "$MODEL_PATH"

python summarize_baseline_results.py --output-root "$EVAL_OUTPUT_ROOT"
