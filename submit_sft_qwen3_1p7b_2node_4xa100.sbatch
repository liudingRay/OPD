#!/bin/bash
# Full-parameter SFT cold start for Qwen3-1.7B-Base.
# Two nodes x four A100 GPUs matches the paper configuration's per-device
# batch size of 8 and yields a global batch size of 64.
#SBATCH --job-name=sft-qwen3-1p7b
#SBATCH --account=EUHPC_D33_157
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.err

set -eo pipefail

PROJECT_DIR=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD
CONDA_BASE=/leonardo/home/userexternal/jbi00000/miniconda3
CONDA_ENV=/leonardo_work/EUHPC_D33_157/jbi00000/conda_envs/opd-sft-py311
# DeepSpeed queries nvcc even when no custom op is compiled. Keep the Python
# runtime in CONDA_ENV, but use the existing CUDA 11.8 toolkit explicitly.
CUDA_TOOLKIT_ROOT=/leonardo_work/EUHPC_D33_157/jbi00000/conda_envs/opd-cu118
OPD_CACHE_ROOT=/leonardo_work/EUHPC_D33_157/OPD-experiment/.cache/opd
WANDB_DIR="$OPD_CACHE_ROOT/wandb"

SFT_DATA="$PROJECT_DIR/LlamaFactory/data/OpenThought3-4B.jsonl"
STUDENT_MODEL="$PROJECT_DIR/model/Qwen3-1.7B-Base"
SFT_OUTPUT="$PROJECT_DIR/model/Qwen3-1.7B-SFT-OpenThought3-4B"
SFT_CONFIG="examples/train_full/qwen3_base_full_sft_opd.yaml"
export PROJECT_DIR SFT_CONFIG

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export WANDB_MODE=offline
export WANDB_DIR
export FORCE_TORCHRUN=1
export NNODES="$SLURM_NNODES"
export NPROC_PER_NODE=4
export OPD_CACHE_ROOT
export TRITON_CACHE_DIR="$OPD_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$OPD_CACHE_ROOT/torchinductor"
export HF_HOME="$OPD_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
unset ROCR_VISIBLE_DEVICES

mkdir -p "$PROJECT_DIR/logs" "$WANDB_DIR" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

if [[ ! -f "$SFT_DATA" ]]; then
    echo "Missing SFT data: $SFT_DATA" >&2
    exit 2
fi

if [[ ! -d "$STUDENT_MODEL" ]]; then
    echo "Missing student model: $STUDENT_MODEL" >&2
    exit 2
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "Missing CUDA compiler: $CUDA_HOME/bin/nvcc" >&2
    exit 2
fi

echo "CUDA toolkit: $CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version
python - <<'PY'
import os
from torch.utils.cpp_extension import CUDA_HOME

if CUDA_HOME != os.environ["CUDA_HOME"]:
    raise RuntimeError(f"PyTorch resolved CUDA_HOME={CUDA_HOME!r}, expected {os.environ['CUDA_HOME']!r}")
print(f"PyTorch CUDA_HOME: {CUDA_HOME}")
PY

if [[ -e "$SFT_OUTPUT" ]]; then
    LATEST_CHECKPOINT=$(find "$SFT_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print \
        | sort -V | tail -n 1)
    if [[ -z "$LATEST_CHECKPOINT" ]]; then
        echo "SFT output exists but contains no resumable checkpoint: $SFT_OUTPUT" >&2
        echo "Refusing to overwrite it. Choose a new output path after inspecting this directory." >&2
        exit 2
    fi
    echo "Detected resumable checkpoint: $LATEST_CHECKPOINT"
    echo "LlamaFactory will automatically resume it because overwrite_output_dir is false."
else
    echo "No existing SFT output found; starting a fresh run."
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${NODES[0]}" hostname --ip-address)
if [[ "$MASTER_ADDR" == *" "* ]]; then
    read -r -a IP_PARTS <<<"$MASTER_ADDR"
    if [[ ${#IP_PARTS[0]} -gt 16 ]]; then
        MASTER_ADDR="${IP_PARTS[1]}"
    else
        MASTER_ADDR="${IP_PARTS[0]}"
    fi
fi
export MASTER_ADDR
export MASTER_PORT=29500

echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "SFT data: $SFT_DATA"
echo "Student model: $STUDENT_MODEL"
echo "Output: $SFT_OUTPUT"
echo "SFT records: $(wc -l < "$SFT_DATA")"

python -m pip check

# Launch one LlamaFactory process per node. The launcher uses NNODES,
# NODE_RANK, MASTER_ADDR, MASTER_PORT, and NPROC_PER_NODE to start four
# torchrun ranks on each node.
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    bash -c '
        set -eo pipefail
        export NODE_RANK="$SLURM_PROCID"
        cd "$PROJECT_DIR/LlamaFactory"
        echo "Node rank: $NODE_RANK | Host: $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
        llamafactory-cli train "$SFT_CONFIG"
    '
