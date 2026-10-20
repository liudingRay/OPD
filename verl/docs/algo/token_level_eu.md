# Student trajectory EU-controlled teacher routing

`reward_model.teacher_weight_mode=eu_ln_softmax` enables the new routing mode.
The legacy `ln_softmax` mode remains unchanged for controlled comparisons.

## Student EU

`verl/utils/epistemic_uncertainty.py::compute_token_level_eu` implements LogTokU:

```text
EU = 16 / sum(Top16(raw_logits) + 1)
```

The actor computes `student_eu [B,T]` during its existing log-probability
forward, before rollout-temperature division. The computation is enabled only
for `eu_ln_softmax`; it does not add another model forward. Teacher workers do
not compute EU in the training path. The standalone evaluation tools may still
use the same numerical utility to analyze saved trajectories.

## Semantic high-EU rate

For each trajectory independently:

1. Compute p95 over all finite, valid response-token EU values, including
   formatting tokens.
2. Select token positions with `EU >= p95`.
3. Remove formatting-only positions using the shared predicate in
   `verl/utils/semantic_tokens.py`.
4. Divide the remaining count by all valid response tokens.

The same predicate is imported by `scripts/val/eval/eval_student_rollout_eu.py`.
Wordpiece/math span merging remains presentation-only; routing counts selected
token positions. For a trajectory with finite EU, normalize the rate as

```text
U = semantic_high_eu_rate / 0.05
```

If a trajectory has no finite EU value (including an empty trajectory), the
rate is zero and routing uses the neutral fallback described below.

## Routing

Existing overlap-mass EMA Learnability `L [B,T,M]` and JS Novelty `N [B,T,M]`
are unchanged. With the defaults `k=15` and `U0=0.25`, compute one pair of
trajectory-level coefficients and clip each coefficient independently:

```text
a_L = clip(1 - k*(U-U0), 0.1, 1.9)
a_N = clip(1 + k*(U-U0), 0.1, 1.9)
l[t,j] = log(clamp(L[t,j], 1e-8))
n[t,j] = log(clamp(N[t,j], 1e-8))
delta_l[t,j] = l[t,j] - mean_j(l[t,j])
delta_n[t,j] = n[t,j] - mean_j(n[t,j])
sigma_L = population_std_{valid t,j}(delta_l[t,j])
sigma_N = population_std_{valid t,j}(delta_n[t,j])
normalized_l[t,j] = delta_l[t,j] / max(sigma_L, 1e-8)
normalized_n[t,j] = delta_n[t,j] / max(sigma_N, 1e-8)
log_score = a_L*normalized_l + a_N*normalized_n
weight = softmax(log_score / teacher_selection_tau, dim=-1)
```

The mean is computed separately at every token over all teachers. There is
then exactly one Learnability scale and one Novelty scale for the complete
current rollout batch, pooling all valid token-teacher entries. Distributed
workers all-reduce the moment statistics before computing the two scales, so
every data-parallel rank routes with identical normalization constants.
Padding is excluded. The legacy `ln_softmax` path is not normalized.

`reward_model.teacher_eu_coefficient_gain` configures `k` and defaults to
`15.0`; `reward_model.teacher_eu_normalized_center` configures `U0` and defaults
to `0.25`. `reward_model.teacher_eu_coefficient_min` and
`reward_model.teacher_eu_coefficient_max` configure the clipping bounds and
default to `0.1` and `1.9`. Both coefficients and all controller parameters are
logged. If no finite EU exists, the effective `U` falls back to `U0` and both
coefficients are exactly `1`.
`teacher_selection_tau` is the existing routing-temperature configuration and
defaults to 0.1; no duplicate temperature parameter is introduced.

## Verification

```bash
python -m pytest tests/utils/test_epistemic_uncertainty_on_cpu.py \
  tests/utils/test_dynamic_teacher_on_cpu.py -q
```

These CPU tests cover the raw LogTokU formula, per-trajectory p95 and fallback,
coefficient adjustment/clipping, uniform teachers, temperature
sharpness/ranking, batch-shared log-metric scales, distributed sufficient-
statistic aggregation, reward broadcast, and the actual actor forward
method's pre-temperature student-EU calculation. They do not replace a real
FSDP/GPU smoke test.
