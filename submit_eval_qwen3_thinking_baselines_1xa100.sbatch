#!/bin/bash
# Evaluate the official Qwen3-1.7B, 4B, and 8B thinking baselines serially:
# one vLLM process at a time on one A100 GPU.
# Submit from the inner OPD checkout with:
#   sbatch submit_eval_qwen3_thinking_baselines_1xa100.sbatch
# Override the result location or rerun existing outputs, for example:
#   EVAL_OUTPUT_ROOT=/path/to/results EVAL_OVERWRITE=1 sbatch submit_eval_qwen3_thinking_baselines_1xa100.sbatch
#SBATCH --job-name=eval-qwen3-thinking
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
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$PROJECT_DIR/scripts/val/eval/justrl_eval_outputs/qwen3-thinking-1p7b-4b-8b-n16}
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
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "Results: $EVAL_OUTPUT_ROOT"
nvidia-smi

cd "$PROJECT_DIR/scripts/val/eval"
GENERATOR_ARGS=(
    --model-root "$PROJECT_DIR/model"
    --data-dir "$PROJECT_DIR/scripts/val/data"
    --output-root "$EVAL_OUTPUT_ROOT"
    --gpu-ids 0
    --serial
    --enable-thinking
    --model "Qwen3-1.7B=$PROJECT_DIR/model/Qwen3-1.7B"
    --model "Qwen3-4B=$PROJECT_DIR/model/Qwen3-4B"
    --model "Qwen3-8B=$PROJECT_DIR/model/Qwen3-8B"
)
if [[ "${EVAL_OVERWRITE:-0}" == "1" ]]; then
    GENERATOR_ARGS+=(--overwrite)
fi
python eval_baselines.py "${GENERATOR_ARGS[@]}"

# Rule-based grading needs no additional GPU allocation. The models share the
# Qwen3 tokenizer; use the evaluated model for its output-length diagnostic.
for MODEL_NAME in Qwen3-1.7B Qwen3-4B Qwen3-8B; do
    python grade.py \
        --eval-dir "$EVAL_OUTPUT_ROOT/$MODEL_NAME" \
        --length-tokenizer "$PROJECT_DIR/model/$MODEL_NAME"
done

python summarize_baseline_results.py --output-root "$EVAL_OUTPUT_ROOT"
