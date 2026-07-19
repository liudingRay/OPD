#!/bin/bash
# First real-data OPD pilot.  This preserves the OPD objective while bounding
# data size, sequence length, and GPU usage so failures stay inexpensive.

set -eo pipefail

export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-pilot64
export TRAIN_MAX_SAMPLES=64

export MAX_RESP_LENGTH=1024
export MAX_VAL_RESP_LENGTH=1024
export MINI_BATCH_SIZE=1
export N_RESPONSES=1
export PARALLEL_SIZE=1

export ROLLOUT_GPU_MEMORY_UTILIZATION=0.4
export REWARD_MODEL_MICRO_BATCH_SIZE=1
export TRAINER_N_GPUS_PER_NODE=1
export TRAINER_NNODES=1
export TRAINER_SAVE_FREQ=20
export TRAINER_TEST_FREQ=-1
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_LOGGER='["console","wandb"]'

# The original launcher keeps these debug flags enabled by default.  They are
# unnecessary for this throughput-oriented pilot and slow GPU execution.
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

exec bash on_policy_distillation.sh
