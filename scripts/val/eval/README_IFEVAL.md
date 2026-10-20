# Independent IFEval evaluation

`eval_ifeval.py` is independent from the math generation and grading path. It
passes every benchmark prompt unchanged as one zero-shot user message, runs
local vLLM generation, and scores the responses with the pinned IFEval rule
checker.

## Default protocol

```text
thinking=false
num_samples=1
temperature=0.0
top_p=1.0
top_k=-1
min_p=0.0
max_tokens=4096
seed=0
```

All generation values have CLI overrides. The standard result should retain
the defaults above. Runs with multiple samples are supported for diagnostics;
their reported prompt metrics average all prompt-sample pairs and must not be
presented as the standard `n=1` result.

## Scorer dependencies

Install the small CPU-only scorer requirements in the selected evaluation
environment:

```bash
python -m pip install -r scripts/val/eval/requirements-ifeval.txt
python -m nltk.downloader -d /ABSOLUTE/PERSISTENT/NLTK_DATA punkt_tab
export NLTK_DATA=/ABSOLUTE/PERSISTENT/NLTK_DATA
```

The evaluation script never downloads NLTK data implicitly. It fails before
loading the GPU model if the scorer or `punkt_tab` resource is unavailable.

## Run

From the repository root:

```bash
python scripts/val/eval/eval_ifeval.py \
  --model-name MODEL_LABEL \
  --model-path /ABSOLUTE/PATH/TO/MODEL \
  --output-root /ABSOLUTE/PATH/TO/EVALUATION/OUTPUTS
```

Use `--max-examples 4` only for a smoke test. `--enable-thinking` creates a
separately named protocol output and must not be pooled with non-thinking
results.

## Outputs

For each protocol the script writes:

```text
*.jsonl          raw generations
*.config.json    exact generation configuration and dataset SHA-256
*.summary.json   aggregate metrics
*.results.jsonl  per-response strict and loose decisions
```

The two main reporting metrics are:

- `main_metrics.prompt_level_strict_acc`
- `main_metrics.prompt_level_loose_acc`

Instruction-level accuracy and instruction-group breakdowns are diagnostics.
The summary also reports response truncation counts.

Existing generations can be rescored without loading vLLM:

```bash
python scripts/val/eval/grade_ifeval.py \
  --response-file /ABSOLUTE/PATH/TO/ifeval-*.jsonl
```
