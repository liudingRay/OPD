#!/bin/bash
# Two-A100 OPD pilot.  This keeps four rollouts per prompt while testing the
# distributed worker path with a smaller, easier-to-schedule allocation.

set -eo pipefail

export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-pilot64-2xa100
export TRAIN_MAX_SAMPLES=64

export MAX_RESP_LENGTH=1024
export MAX_VAL_RESP_LENGTH=1024
# Global train batch size: two prompts, or one prompt per A100.
export MINI_BATCH_SIZE=2
export N_RESPONSES=4
export PARALLEL_SIZE=1

export ROLLOUT_GPU_MEMORY_UTILIZATION=0.4
export REWARD_MODEL_MICRO_BATCH_SIZE=1
export TRAINER_N_GPUS_PER_NODE=2
export TRAINER_NNODES=1
export TRAINER_SAVE_FREQ=10
export TRAINER_TEST_FREQ=-1
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_LOGGER='["console","wandb"]'

export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

exec bash on_policy_distillation.sh
