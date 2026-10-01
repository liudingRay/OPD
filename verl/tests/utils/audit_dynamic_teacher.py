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

"""Independently verify a real smoke scoring dump in CPU float64.

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
    utility = learn * js
    weights = (utility / dump["tau"]).softmax(-1)
    reward = (q * (t - s[..., None]) * weights[..., None, :]).sum(-1)
    refs = {"overlap_mass": mass, "learnability": learn, "js": js, "utility": utility, "weight": weights}
    report = {
        "step": dump["step"],
        "trajectories": s.shape[0],
        "valid_tokens": valid.sum().item(),
        "beta": dump["beta"],
        "tau": dump["tau"],
        "checks": {},
    }
    for name, ref in {**{f"dynamic_teacher_{k}": v for k, v in refs.items()}, "rm_scores": reward}.items():
        actual = data[name].double()[valid]
        expected = ref[valid]
        error = (actual - expected).abs()
        # Reward retains the original scoring dtype/candidate normalization.
        # bfloat16 inputs imply larger rounding than the float32 weight path.
        tolerance = 0.02 if name == "rm_scores" and data["student_top_k_log_probs"].dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        report["checks"][name] = {
            "max_abs_error": error.max().item(),
            "mean_abs_error": error.mean().item(),
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
