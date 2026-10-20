# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independently verify a real smoke scoring dump on CPU.

Usage: python audit_dynamic_teacher.py /absolute/path/step-1.pt
This script deliberately does not import the production weighting function.
"""

import json
import math
import sys
from pathlib import Path

import torch


def audit(path):
    dump = torch.load(path, map_location="cpu", weights_only=True)
    data = dump["tensors"]
    valid = data["response_mask"].bool()
    s = data["student_top_k_log_probs"].double()
    t = data["teacher_on_student_log_probs_by_teacher"].double()
    overlap = data["teacher_overlap_mask_by_teacher"].double()
    mass = (s.exp()[..., None] * overlap).sum(-2)
    q = s.softmax(-1)[..., None]
    p = t.softmax(-2)
    middle = (q + p) / 2
    js = 0.5 * ((q * (q.log() - middle.log())) + (p * (p.log() - middle.log()))).sum(-2) / math.log(2)
    learn = torch.zeros_like(mass)
    for b in range(s.shape[0]):
        previous = None
        for pos in range(s.shape[1]):
            if valid[b, pos]:
                previous = (
                    mass[b, pos] if previous is None else dump["beta"] * previous + (1 - dump["beta"]) * mass[b, pos]
                )
                learn[b, pos] = previous
    mode = dump.get("mode", "ln_softmax")
    if mode == "eu_ln_softmax":
        # Match the production code's float32 quantile boundary exactly. The
        # remaining algebra stays in float64, but changing dtype before this
        # discrete >= threshold decision can select a different boundary token.
        student_eu = data["student_eu"].float()
        semantic = data["semantic_token_mask"].bool()
        rate = torch.zeros(s.shape[0], dtype=torch.float32)
        normalized_u = torch.zeros_like(rate)
        a_l = torch.zeros_like(rate)
        a_n = torch.zeros_like(rate)
        coefficient_gain = dump.get("eu_coefficient_gain", 15.0)
        normalized_center = dump.get("eu_normalized_center", 0.25)
        coefficient_min = dump.get("eu_coefficient_min", 0.1)
        coefficient_max = dump.get("eu_coefficient_max", 1.9)
        for b in range(s.shape[0]):
            finite_valid = valid[b] & torch.isfinite(student_eu[b])
            if finite_valid.any():
                threshold = torch.quantile(student_eu[b, finite_valid], 0.95)
                rate[b] = (finite_valid & semantic[b] & (student_eu[b] >= threshold)).sum() / valid[b].sum()
                normalized_u[b] = rate[b] / 0.05
                adjustment = coefficient_gain * (normalized_u[b] - normalized_center)
                a_l[b] = (1 - adjustment).clamp(coefficient_min, coefficient_max)
                a_n[b] = (1 + adjustment).clamp(coefficient_min, coefficient_max)
            else:
                normalized_u[b] = normalized_center
                a_l[b] = 1
                a_n[b] = 1
        # The raw-logit checks below independently validate ``learn`` and
        # ``js``. Reuse those saved, validated float32 intermediates for the
        # normalization stage so log/centering and division by a small scale
        # follow the production dtype instead of amplifying harmless float64
        # reconstruction differences.
        routing_learn = data["dynamic_teacher_learnability"].float()
        routing_novelty = data["dynamic_teacher_js"].float()
        centered_log_l = routing_learn.clamp_min(1e-8).log()
        centered_log_l = centered_log_l - centered_log_l.mean(-1, keepdim=True)
        centered_log_n = routing_novelty.clamp_min(1e-8).log()
        centered_log_n = centered_log_n - centered_log_n.mean(-1, keepdim=True)
        l_scale = data["dynamic_teacher_learnability_scale"].float()
        n_scale = data["dynamic_teacher_novelty_scale"].float()
        if not torch.equal(l_scale, l_scale[:1].expand_as(l_scale)):
            raise AssertionError("Learnability scale is not shared across the dumped batch")
        if not torch.equal(n_scale, n_scale[:1].expand_as(n_scale)):
            raise AssertionError("Novelty scale is not shared across the dumped batch")
        normalized_log_l = centered_log_l / l_scale[:, None, None]
        normalized_log_n = centered_log_n / n_scale[:, None, None]
        centered_log_l = centered_log_l.masked_fill(~valid[..., None], 0)
        centered_log_n = centered_log_n.masked_fill(~valid[..., None], 0)
        normalized_log_l = normalized_log_l.masked_fill(~valid[..., None], 0)
        normalized_log_n = normalized_log_n.masked_fill(~valid[..., None], 0)
        log_score = a_l[:, None, None] * normalized_log_l + a_n[:, None, None] * normalized_log_n
        weights = (log_score / dump["tau"]).softmax(-1)
        refs = {
            "overlap_mass": mass,
            "learnability": learn,
            "js": js,
            "log_routing_score": log_score,
            "weight": weights,
            "semantic_high_eu_rate": rate,
            "eu_normalized_uncertainty": normalized_u,
            "learnability_coefficient": a_l,
            "novelty_coefficient": a_n,
            "centered_log_learnability": centered_log_l,
            "centered_log_novelty": centered_log_n,
            "normalized_log_learnability": normalized_log_l,
            "normalized_log_novelty": normalized_log_n,
            "learnability_scale": l_scale,
            "novelty_scale": n_scale,
        }
    else:
        utility = learn * js
        weights = (utility / dump["tau"]).softmax(-1)
        refs = {"overlap_mass": mass, "learnability": learn, "js": js, "utility": utility, "weight": weights}
    reward = (q * (t - s[..., None]) * weights[..., None, :]).sum(-1)
    report = {
        "step": dump["step"],
        "mode": mode,
        "trajectories": s.shape[0],
        "valid_tokens": valid.sum().item(),
        "beta": dump["beta"],
        "tau": dump["tau"],
        "checks": {},
    }
    if mode == "eu_ln_softmax":
        report["eu_coefficient_gain"] = coefficient_gain
        report["eu_normalized_center"] = normalized_center
        report["eu_coefficient_min"] = coefficient_min
        report["eu_coefficient_max"] = coefficient_max
        report["learnability_scale"] = l_scale[0].item()
        report["novelty_scale"] = n_scale[0].item()
    for name, ref in {**{f"dynamic_teacher_{k}": v for k, v in refs.items()}, "rm_scores": reward}.items():
        if ref.ndim == 1:
            actual = data[name].double()
            expected = ref.double()
        else:
            actual = data[name].double()[valid]
            expected = ref.double()[valid]
        absolute_error = (actual - expected).abs()
        # Reward retains the original scoring dtype/candidate normalization.
        # bfloat16 inputs imply larger rounding than the float32 weight path.
        tolerance = 0.02 if name == "rm_scores" and data["student_top_k_log_probs"].dtype == torch.bfloat16 else 2e-5
        try:
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        except AssertionError as assertion_error:
            raise AssertionError(
                f"Audit mismatch for {name}: max_abs_error={absolute_error.max().item():.8g}, "
                f"mean_abs_error={absolute_error.mean().item():.8g}, tolerance={tolerance}"
            ) from assertion_error
        report["checks"][name] = {
            "max_abs_error": absolute_error.max().item(),
            "mean_abs_error": absolute_error.mean().item(),
            "tolerance": tolerance,
        }
    report["weights_mean"] = weights[valid].mean(0).tolist()
    report["weights_min"] = weights[valid].amin(0).tolist()
    report["weights_max"] = weights[valid].amax(0).tolist()
    report["weight_sum_max_error"] = (data["dynamic_teacher_weight"][valid].sum(-1) - 1).abs().max().item()
    report["status"] = "PASS"
    return report


if __name__ == "__main__":
    result = audit(sys.argv[1])
    output = Path(sys.argv[1]).with_suffix(".audit.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
