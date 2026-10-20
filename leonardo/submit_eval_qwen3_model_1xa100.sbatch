#!/usr/bin/env bash
# Evaluate one local Qwen3 model on selected eval_baselines.py tasks.
#
# Prefer a benchmark-specific Leonardo wrapper when one exists. For the
# original AIME24/AIME25/AMC23 avg@16 protocol, submit from the inner checkout:
#   MODEL_NAME=... MODEL_PATH=/absolute/model/path \
#     sbatch --export=ALL leonardo/submit_eval_qwen3_model_1xa100.sbatch
#SBATCH --job-name=eval-qwen3-model
#SBATCH --account=EUHPC_D33_157
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.out
#SBATCH --error=/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/leonardo_work/EUHPC_D33_157/OPD-experiment/OPD}
CONDA_BASE=${CONDA_BASE:-/leonardo/home/userexternal/jbi00000/miniconda3}
OPD_CACHE_ROOT=${OPD_CACHE_ROOT:-/leonardo_work/EUHPC_D33_157/OPD-experiment/.cache/opd}
MODEL_ROOT=${MODEL_ROOT:-$PROJECT_DIR/model}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/scripts/val/data}
if [[ -z ${MODEL_NAME:-} || -z ${MODEL_PATH:-} ]]; then
    echo "Set both MODEL_NAME and the absolute MODEL_PATH for the model to evaluate." >&2
    exit 2
fi
if [[ "$MODEL_PATH" != /* ]]; then
    echo "MODEL_PATH must be absolute; got: $MODEL_PATH" >&2
    exit 2
fi

ENABLE_THINKING=${ENABLE_THINKING:-0}
case "$ENABLE_THINKING" in
    1) THINKING_LABEL=thinking ;;
    0) THINKING_LABEL=nonthinking ;;
    *)
        echo "ENABLE_THINKING must be 0 or 1; got: $ENABLE_THINKING" >&2
        exit 2
        ;;
esac
TASKS=${TASKS:-"AIME24 AIME25 AMC23"}
NUM_SAMPLES=${NUM_SAMPLES:-16}
MAX_EXAMPLES=${MAX_EXAMPLES:-}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.8}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0.0}
MAX_TOKENS=${MAX_TOKENS:-38912}
SEED=${SEED:-0}
EVAL_OVERWRITE=${EVAL_OVERWRITE:-0}
EVAL_PROTOCOL_LABEL=${EVAL_PROTOCOL_LABEL:-t${TEMPERATURE}-p${TOP_P}-topk${TOP_K}-minp${MIN_P}-mnt${MAX_TOKENS}}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-$PROJECT_DIR/scripts/val/eval/justrl_eval_outputs/$MODEL_NAME-$THINKING_LABEL-$EVAL_PROTOCOL_LABEL-n$NUM_SAMPLES}
CONDA_ENV=${CONDA_ENV:-/leonardo_work/EUHPC_D33_157/jbi00000/conda_envs/opd-cu118}

case "$EVAL_OVERWRITE" in
    0|1) ;;
    *)
        echo "EVAL_OVERWRITE must be 0 or 1; got: $EVAL_OVERWRITE" >&2
        exit 2
        ;;
esac
if [[ ! "$NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SAMPLES must be a positive integer; got: $NUM_SAMPLES" >&2
    exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer; got: $SEED" >&2
    exit 2
fi
if [[ -n "$MAX_EXAMPLES" && ! "$MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_EXAMPLES must be empty or a positive integer; got: $MAX_EXAMPLES" >&2
    exit 2
fi
for task in $TASKS; do
    case "$task" in
        AIME24|AIME25|AMC23|MATH-500|GPQA|MMLU-Pro) ;;
        *)
            echo "Unsupported task: $task (supported: AIME24 AIME25 AMC23 MATH-500 GPQA MMLU-Pro)" >&2
            exit 2
            ;;
    esac
    if [[ ! -f "$DATA_DIR/$task/test.parquet" ]]; then
        echo "Benchmark file is missing: $DATA_DIR/$task/test.parquet" >&2
        exit 1
    fi
done

set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
cd "$PROJECT_DIR"

export OPD_CACHE_ROOT
export TRITON_CACHE_DIR="$OPD_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$OPD_CACHE_ROOT/torchinductor"
export HF_HOME="$OPD_CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export OUTLINES_CACHE_DIR="$OPD_CACHE_ROOT/outlines/${SLURM_JOB_ID}"
unset ROCR_VISIBLE_DEVICES

mkdir -p "$PROJECT_DIR/logs" "$EVAL_OUTPUT_ROOT" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$OUTLINES_CACHE_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Model: $MODEL_NAME ($MODEL_PATH)"
echo "Tasks: $TASKS"
echo "Thinking: $ENABLE_THINKING"
echo "Samples: $NUM_SAMPLES"
echo "Max examples: ${MAX_EXAMPLES:-all}"
echo "Sampling: temperature=$TEMPERATURE, top_p=$TOP_P, top_k=$TOP_K, min_p=$MIN_P"
echo "Base seed: $SEED (rollout i uses seed+i)"
echo "Max tokens: $MAX_TOKENS"
echo "Overwrite: $EVAL_OVERWRITE"
echo "Results: $EVAL_OUTPUT_ROOT"
python --version
python -m pip check
nvidia-smi

cd "$PROJECT_DIR/scripts/val/eval"
read -r -a TASK_ARGS <<< "$TASKS"
GENERATOR_ARGS=(
    --model-root "$MODEL_ROOT"
    --data-dir "$DATA_DIR"
    --output-root "$EVAL_OUTPUT_ROOT"
    --gpu-ids 0
    --serial
    --tasks "${TASK_ARGS[@]}"
    --num-samples "$NUM_SAMPLES"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --min-p "$MIN_P"
    --max-tokens "$MAX_TOKENS"
    --seed "$SEED"
    --model "$MODEL_NAME=$MODEL_PATH"
)
if [[ -n "$MAX_EXAMPLES" ]]; then
    GENERATOR_ARGS+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "$THINKING_LABEL" == thinking ]]; then
    GENERATOR_ARGS+=(--enable-thinking)
fi
if [[ "$EVAL_OVERWRITE" == 1 ]]; then
    GENERATOR_ARGS+=(--overwrite)
fi
python eval_baselines.py "${GENERATOR_ARGS[@]}"

python grade.py \
    --eval-dir "$EVAL_OUTPUT_ROOT/$MODEL_NAME" \
    --length-tokenizer "$MODEL_PATH"

python summarize_baseline_results.py --output-root "$EVAL_OUTPUT_ROOT"
