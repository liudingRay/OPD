#!/bin/bash
# One-A100 Slurm submission wrapper for the smoke-test configuration.
# Submit from the OPD repository with: sbatch submit_opd_smoke_1xa100.sbatch
# Increase --time only after this batch-mode smoke test succeeds.
#SBATCH --job-name=opd-smoke-1xa100
#SBATCH --account=EUHPC_D29_082
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.err

# Do not enable `set -u`: Conda's binutils_linux-64 activation hook references
# optional variables such as ADDR2LINE before defining them.
set -eo pipefail

PROJECT_DIR=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD
CONDA_BASE=/leonardo/home/userexternal/jbi00000/miniconda3
WANDB_DIR=/leonardo_work/EUHPC_D29_082/OPD-experiment/.cache/opd/wandb

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$WANDB_DIR"
# Compute nodes have no internet access.  Store W&B records locally, then run
# `wandb sync` from a networked login node after the job completes.
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

# This script's own #SBATCH lines are ignored because it is invoked through bash.
bash opd_smoke_1xa100.sh
