#!/bin/bash
# Single-node distributed OPD pilot.  It preserves the successful 1xA100
# settings while restoring the paper's four responses per prompt.

set -eo pipefail

export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-pilot128-4xa100
export TRAIN_MAX_SAMPLES=128

export MAX_RESP_LENGTH=1024
export MAX_VAL_RESP_LENGTH=1024
# Global train batch size: four prompts, or one prompt per A100.
export MINI_BATCH_SIZE=4
export N_RESPONSES=4
export PARALLEL_SIZE=1

export ROLLOUT_GPU_MEMORY_UTILIZATION=0.4
export REWARD_MODEL_MICRO_BATCH_SIZE=1
export TRAINER_N_GPUS_PER_NODE=4
export TRAINER_NNODES=1
export TRAINER_SAVE_FREQ=10
export TRAINER_TEST_FREQ=-1
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_LOGGER='["console","wandb"]'

export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

exec bash on_policy_distillation.sh
