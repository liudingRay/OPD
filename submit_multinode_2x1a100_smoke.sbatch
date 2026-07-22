#!/bin/bash
# Two-node allocation smoke test.  This intentionally does not start Ray or
# training; it verifies that Slurm can place one A100 task on each node.
#SBATCH --job-name=opd-multinode-smoke
#SBATCH --account=EUHPC_D29_082
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D29_082/OPD-experiment/OPD/logs/%x-%j.err

set -eo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Allocated nodes: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"

srun --ntasks=2 --ntasks-per-node=1 --label bash -c '
    echo "host=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi -L
'
