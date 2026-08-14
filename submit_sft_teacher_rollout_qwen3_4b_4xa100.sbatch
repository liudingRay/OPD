#!/bin/bash
# Generate Qwen3-4B teacher responses for the 30K OpenThoughts3 SFT prompts.
# This is single-node only: vllm_rollout.py starts one independent vLLM worker
# per allocated GPU, so do not wrap the final python command in srun.
#
# Resume an interrupted job by submitting this exact script again. Accepted
# responses are flushed to output/Qwen3-4B/temp_rollout/worker_*.jsonl and
# will be reused automatically.
#SBATCH --job-name=sft-rollout-qwen3-4b
#SBATCH --account=EUHPC_D33_157
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD}
CONDA_BASE=${CONDA_BASE:-/leonardo/home/userexternal/jbi00000/miniconda3}
CONDA_ENV=${CONDA_ENV:-/leonardo_work/EUHPC_D33_157/jbi00000/conda_envs/opd-cu118}
OPD_CACHE_ROOT=${OPD_CACHE_ROOT:-/leonardo_work/EUHPC_D33_157/OPD-experiment/.cache/opd}

INPUT_PARQUET=${INPUT_PARQUET:-$PROJECT_DIR/datasets/OpenThoughts3_opd.parquet}
TEACHER_MODEL=${TEACHER_MODEL:-$PROJECT_DIR/model/Qwen3-4B}
GPU_IDS=${GPU_IDS:-0,1,2,3}

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

export TRITON_CACHE_DIR="$OPD_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$OPD_CACHE_ROOT/torchinductor"
export HF_HOME="$OPD_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

mkdir -p "$PROJECT_DIR/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

if [[ ! -f "$INPUT_PARQUET" ]]; then
    echo "Input parquet does not exist: $INPUT_PARQUET" >&2
    exit 2
fi

if [[ ! -d "$TEACHER_MODEL" ]]; then
    echo "Teacher model directory does not exist: $TEACHER_MODEL" >&2
    exit 2
fi

ROLLOUT_PID=""
save_and_exit() {
    echo "$(date): Slurm time-limit warning received; stopping rollout."
    echo "Accepted records are already flushed to temp_rollout and the next submission will resume."
    if [[ -n "$ROLLOUT_PID" ]] && kill -0 "$ROLLOUT_PID" 2>/dev/null; then
        kill -TERM "$ROLLOUT_PID"
        wait "$ROLLOUT_PID" || true
    fi
    exit 0
}
trap save_and_exit USR1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Teacher: $TEACHER_MODEL"
echo "Prompts: $INPUT_PARQUET"
echo "Output: $PROJECT_DIR/output/$(basename "$TEACHER_MODEL")/DAPO.jsonl"
nvidia-smi

python scripts/infer/vllm_rollout.py \
    --input-parquet "$INPUT_PARQUET" \
    --model-path "$TEACHER_MODEL" \
    --gpu-ids "$GPU_IDS" \
    --enable-thinking false \
    --enable-rejection-sampling true \
    --max-attempts-per-rollout 3 &
ROLLOUT_PID=$!
wait "$ROLLOUT_PID"
ROLLOUT_PID=""

echo "Rollout complete. Copy the finalized JSONL into LlamaFactory only after checking its completion rate:"
echo "  cp $PROJECT_DIR/output/$(basename "$TEACHER_MODEL")/DAPO.jsonl $PROJECT_DIR/LlamaFactory/data/OpenThought3-4B.jsonl"
