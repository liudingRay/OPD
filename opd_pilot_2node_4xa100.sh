#!/bin/bash
# Two-node OPD reproduction run: four A100 GPUs on each node.  The Slurm
# wrapper starts the shared Ray cluster; this launcher only supplies the OPD
# training settings.

set -eo pipefail

export RAY_CLUSTER_ALREADY_STARTED=1
export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-repro-2node-4xa100
export TRAIN_MAX_SAMPLES=-1

export MAX_RESP_LENGTH=7168
export MAX_VAL_RESP_LENGTH=7168
# Global train batch size: 64 prompts.
export MINI_BATCH_SIZE=64
export N_RESPONSES=4
export PARALLEL_SIZE=1

export ROLLOUT_GPU_MEMORY_UTILIZATION=0.8
export REWARD_MODEL_MICRO_BATCH_SIZE=24
export TRAINER_N_GPUS_PER_NODE=4
export TRAINER_NNODES=2
export TRAINER_SAVE_FREQ=40
export TRAINER_TEST_FREQ=-1
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_LOGGER='["console","wandb"]'

export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

exec bash on_policy_distillation.sh
