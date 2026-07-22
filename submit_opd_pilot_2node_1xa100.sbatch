#!/bin/bash
# Two-node OPD training pilot.  Each node receives one A100, eight CPUs, and
# 64G RAM.  A Ray head is started on the first node and a Ray worker on the
# second node before the OPD driver runs on the head.
#SBATCH --job-name=opd-pilot-2node
#SBATCH --account=EUHPC_D29_082
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.err

set -eo pipefail

PROJECT_DIR=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD
CONDA_BASE=/leonardo/home/userexternal/jbi00000/miniconda3
OPD_CACHE_ROOT=/leonardo_work/EUHPC_D29_082/OPD-experiment/.cache/opd
WANDB_DIR="$OPD_CACHE_ROOT/wandb"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$WORK/$USER/conda_envs/opd-cu118"
cd "$PROJECT_DIR"

export WANDB_MODE=offline
export WANDB_DIR
export OPD_CACHE_ROOT
export TRITON_CACHE_DIR="$OPD_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$OPD_CACHE_ROOT/torchinductor"
export HF_HOME="$OPD_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export OUTLINES_CACHE_DIR="$OPD_CACHE_ROOT/outlines/${SLURM_JOB_ID}"
unset ROCR_VISIBLE_DEVICES

mkdir -p "$PROJECT_DIR/logs" "$WANDB_DIR" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" \
    "$OUTLINES_CACHE_DIR" "$OPD_CACHE_ROOT/verl-rlhf"

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

if [[ "$HEAD_IP" == *" "* ]]; then
    read -r -a IP_PARTS <<<"$HEAD_IP"
    if [[ ${#IP_PARTS[0]} -gt 16 ]]; then
        HEAD_IP=${IP_PARTS[1]}
    else
        HEAD_IP=${IP_PARTS[0]}
    fi
fi

RAY_PORT=6379
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

cleanup() {
    srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NTASKS" \
        --ntasks-per-node=1 ray stop --force || true
}
trap cleanup EXIT

echo "Job ID: ${SLURM_JOB_ID}"
echo "Allocated nodes: ${SLURM_JOB_NODELIST}"
echo "Ray head: ${RAY_ADDRESS}"

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
    --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus=1 --block &
sleep 10

for ((NODE_INDEX = 1; NODE_INDEX < ${#NODES[@]}; NODE_INDEX++)); do
    WORKER_NODE=${NODES[$NODE_INDEX]}
    srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" \
        ray start --address="$RAY_ADDRESS" --num-cpus="$SLURM_CPUS_PER_TASK" \
        --num-gpus=1 --block &
    sleep 5
done

# Some Ray releases can report a transient cluster-metadata error in the
# status CLI even though a driver can connect normally.  Do not let this
# diagnostic command terminate the training job; main_ppo verifies the actual
# Ray connection immediately afterwards.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" ray status --address="$RAY_ADDRESS" \
    || echo "Warning: ray status failed; continuing to the driver connection test."
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash opd_pilot_2node_1xa100.sh
