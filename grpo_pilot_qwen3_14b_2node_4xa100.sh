#!/bin/bash
# Two-node Qwen3-14B GRPO pilot: four A100 GPUs on each node. The sbatch
# wrapper starts Ray; this file only sets GRPO and resource-sensitive options.

set -eo pipefail

export RAY_CLUSTER_ALREADY_STARTED=1
export TRAIN_DATASET=datasets/dapo-math-17k.parquet
export TRAIN_DATASET_NAME=DAPO-Math-17k-qwen3-14b-grpo-2node-4xa100
export TRAIN_MAX_SAMPLES=-1

export ACTOR_MODEL_PATH=model/Qwen3-14B-Base
export MODEL_DTYPE=bfloat16
export MAX_RESP_LENGTH=7168
export MAX_VAL_RESP_LENGTH=7168
# GRPO needs multiple rollouts per prompt. Four is a conservative A100-64GB
# starting point; retain the group-relative objective while limiting KV cache.
export MINI_BATCH_SIZE=8
export N_RESPONSES=4
# Two-way tensor/sequence parallelism shards the 14B rollout model. Keeping
# MINI_BATCH_SIZE at 8 preserves global train_batch_size = 8 x 2 = 16.
export PARALLEL_SIZE=2

# A 14B BF16 rollout consumes substantially more memory than the successful
# OPD pilot's 1.5B actor plus teacher.
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export PPO_MAX_TOKEN_LEN_PER_GPU=16384
export ACTOR_ENTROPY_FROM_LOGITS_WITH_CHUNKING=True
export TRAINER_N_GPUS_PER_NODE=4
export TRAINER_NNODES=2
export TRAINER_SAVE_FREQ=40
export TRAINER_TEST_FREQ=-1
export TRAINER_TOTAL_EPOCHS=1
export TRAINER_LOGGER='["console","wandb"]'

# Continue the interrupted 24-hour pilot from its complete GRPO checkpoint.
# Keeping the original run directory makes resume_mode=auto load global_step_440
# now and the latest checkpoint automatically on later resubmissions.
export CKPT_PATH=checkpoint/grpo_DAPO-Math-17k-qwen3-14b-grpo-2node-4xa100_Qwen3-14B-Base_Qwen3-14B-Base_7168-T_1.0-Tch_1.0-n_4-mbs_8-topk_0-topk_strategy_union-rw_student_p-2026-07-26_11-30-16
export TRAINER_RESUME_MODE=auto
export EXPERIMENT_NAME="${CKPT_PATH##*/}"

# Reuse the existing W&B run. The new offline folder will have this same ID
# and must be synced with `wandb sync --append` after the job finishes.
export WANDB_RUN_ID=7gahb82o
export WANDB_RESUME=allow

export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

exec bash grpo_qwen3_14b_base.sh
