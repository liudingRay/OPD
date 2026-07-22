#!/bin/bash
# Submit with: sbatch submit_opd_pilot_2xa100.sbatch
# Two-A100, real-data OPD pilot: 64 DAPO examples, 1024 response tokens, and
# four responses per prompt.
#SBATCH --job-name=opd-pilot-2xa100
#SBATCH --account=EUHPC_D29_082
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.err

set -eo pipefail

PROJECT_DIR=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD
CONDA_BASE=/leonardo/home/userexternal/jbi00000/miniconda3
WANDB_DIR=/leonardo_work/EUHPC_D29_082/OPD-experiment/.cache/opd/wandb

mkdir -p "$PROJECT_DIR/logs" "$WANDB_DIR"
export WANDB_MODE=offline
export WANDB_DIR

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$WORK/$USER/conda_envs/opd-cu118"
cd "$PROJECT_DIR"

cleanup() {
    ray stop --force || true
}
trap cleanup EXIT

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
nvidia-smi

bash opd_pilot_2xa100.sh
