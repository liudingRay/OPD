#!/bin/bash
# Convert the final Qwen3-14B GRPO FSDP actor checkpoint to a standalone
# Hugging Face/vLLM model directory. This is a CPU/RAM-only job: it does not
# start Ray, vLLM, or a training process.
#SBATCH --job-name=merge-qwen3-14b-grpo
#SBATCH --account=EUHPC_D33_157
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.err

# Conda's compiler deactivation hooks reference optional backup variables.
# Keep nounset disabled until `conda activate` has completed successfully.
set -eo pipefail

PROJECT_DIR=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD
CONDA_BASE=/leonardo/home/userexternal/jbi00000/miniconda3
RUN_DIR="$PROJECT_DIR/checkpoint/grpo_DAPO-Math-17k-qwen3-14b-grpo-2node-4xa100_Qwen3-14B_Qwen3-14B_7168-T_1.0-Tch_1.0-n_4-mbs_8-topk_0-topk_strategy_union-rw_student_p"
EXPECTED_STEP=1119
ACTOR_CKPT="$RUN_DIR/global_step_${EXPECTED_STEP}/actor"
MERGED_MODEL_NAME=Qwen3-14B-GRPO
MERGED_DIR="$PROJECT_DIR/model/$MERGED_MODEL_NAME"
STAGING_DIR="$PROJECT_DIR/model/.${MERGED_MODEL_NAME}.merge-${SLURM_JOB_ID}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$WORK/$USER/conda_envs/opd-cu118"
set -u
cd "$PROJECT_DIR/verl"

# Always use the checked-out verl merger rather than a possibly different
# installed package. The merge only reads local checkpoint/config files.
export PYTHONPATH="$PROJECT_DIR/verl${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Actor checkpoint: $ACTOR_CKPT"
echo "Final model directory: $MERGED_DIR"

if [[ ! -f "$RUN_DIR/latest_checkpointed_iteration.txt" ]]; then
    echo "Missing checkpoint tracker: $RUN_DIR/latest_checkpointed_iteration.txt" >&2
    exit 1
fi

RECORDED_STEP=$(tr -d '[:space:]' < "$RUN_DIR/latest_checkpointed_iteration.txt")
if [[ "$RECORDED_STEP" != "$EXPECTED_STEP" ]]; then
    echo "Refusing to merge: tracker records global_step_${RECORDED_STEP}, expected global_step_${EXPECTED_STEP}." >&2
    exit 1
fi

for REQUIRED_FILE in "$ACTOR_CKPT/fsdp_config.json" "$ACTOR_CKPT/huggingface/config.json"; do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "Missing required checkpoint file: $REQUIRED_FILE" >&2
        exit 1
    fi
done

# A successful FSDP merge requires every actor shard, not only rank zero.
# Check them before allocating RAM to deserialize the checkpoint.
python - "$ACTOR_CKPT" <<'PY'
import json
import sys
from pathlib import Path

actor_checkpoint = Path(sys.argv[1])
world_size = json.loads((actor_checkpoint / "fsdp_config.json").read_text())["world_size"]
missing = [
    actor_checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt"
    for rank in range(world_size)
    if not (actor_checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
]
if missing:
    raise SystemExit("Missing FSDP actor shard(s):\n" + "\n".join(map(str, missing)))
print(f"Verified {world_size} FSDP actor shard(s).")
PY

# Do not overwrite a completed model. Write to a job-specific staging directory
# first, then publish atomically only after the expected HF files exist.
if [[ -e "$MERGED_DIR" ]]; then
    echo "Target already exists: $MERGED_DIR" >&2
    echo "Refusing to overwrite it. Choose a new target only after inspecting the existing model." >&2
    exit 1
fi
if [[ -e "$STAGING_DIR" ]]; then
    echo "Unexpected existing staging directory: $STAGING_DIR" >&2
    exit 1
fi

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_CKPT" \
    --target_dir "$STAGING_DIR"

if [[ ! -f "$STAGING_DIR/config.json" ]]; then
    echo "Merge finished without $STAGING_DIR/config.json" >&2
    exit 1
fi
if ! compgen -G "$STAGING_DIR/model*.safetensors" > /dev/null; then
    echo "Merge finished without Hugging Face model safetensors in $STAGING_DIR" >&2
    exit 1
fi

mv "$STAGING_DIR" "$MERGED_DIR"
echo "Merged model is ready: $MERGED_DIR"
du -sh "$MERGED_DIR"
find "$MERGED_DIR" -maxdepth 1 -type f -printf '%f\n' | sort
