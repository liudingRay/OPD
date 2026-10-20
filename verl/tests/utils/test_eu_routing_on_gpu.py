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

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("multi_teacher", ROOT / "verl/utils/multi_teacher.py")
multi_teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_teacher)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This check intentionally requires an allocated GPU")
def test_artificial_eu_learnability_novelty_produce_expected_dynamic_weights():
    device = torch.device("cuda")

    # With 80 valid positions, p95 selects positions 76-79. The semantic
    # filter keeps 0, 1, and 4 of them, producing U=0, 0.25, and 1.
    student_eu = torch.arange(80, device=device, dtype=torch.float32).repeat(3, 1)
    response_mask = torch.ones_like(student_eu, dtype=torch.bool)
    semantic_mask = torch.zeros_like(response_mask)
    semantic_mask[1, -1] = True
    semantic_mask[2, -4:] = True

    high_eu_rate, normalized_u, a_l, a_n = multi_teacher.trajectory_eu_routing_coefficients(
        student_eu,
        response_mask,
        semantic_mask,
    )
    torch.testing.assert_close(high_eu_rate, torch.tensor([0.0, 0.0125, 0.05], device=device))
    torch.testing.assert_close(normalized_u, torch.tensor([0.0, 0.25, 1.0], device=device))
    torch.testing.assert_close(a_l, torch.tensor([1.9, 1.0, 0.1], device=device))
    torch.testing.assert_close(a_n, torch.tensor([0.1, 1.0, 1.9], device=device))

    learnability = torch.tensor(
        [
            [[0.20, 0.40, 0.80], [0.70, 0.50, 0.10]],
            [[0.20, 0.40, 0.80], [0.70, 0.50, 0.10]],
            [[0.20, 0.40, 0.80], [0.70, 0.50, 0.10]],
        ],
        device=device,
    )
    novelty = torch.tensor(
        [
            [[0.90, 0.30, 0.10], [0.20, 0.50, 0.80]],
            [[0.90, 0.30, 0.10], [0.20, 0.50, 0.80]],
            [[0.90, 0.30, 0.10], [0.20, 0.50, 0.80]],
        ],
        device=device,
    )
    tau = 0.1
    weights, log_score = multi_teacher.eu_controlled_teacher_weights(
        learnability,
        novelty,
        a_l,
        a_n,
        tau=tau,
    )

    centered_l = learnability.log() - learnability.log().mean(dim=-1, keepdim=True)
    centered_n = novelty.log() - novelty.log().mean(dim=-1, keepdim=True)
    normalized_l = centered_l / centered_l.std(unbiased=False)
    normalized_n = centered_n / centered_n.std(unbiased=False)
    expected_log_score = a_l[:, None, None] * normalized_l + a_n[:, None, None] * normalized_n
    expected_weights = torch.softmax(expected_log_score / tau, dim=-1)
    torch.testing.assert_close(log_score, expected_log_score)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    torch.testing.assert_close(
        log_score[0], 1.9 * normalized_l[0] + 0.1 * normalized_n[0]
    )
    torch.testing.assert_close(log_score[1], normalized_l[1] + normalized_n[1])
    torch.testing.assert_close(
        log_score[2], 0.1 * normalized_l[2] + 1.9 * normalized_n[2]
    )

    # An all-non-finite trajectory uses the neutral U=U0 and coefficients 1,1.
    _, fallback_u, fallback_a_l, fallback_a_n = multi_teacher.trajectory_eu_routing_coefficients(
        torch.full((1, 80), float("nan"), device=device),
        torch.ones((1, 80), dtype=torch.bool, device=device),
        torch.ones((1, 80), dtype=torch.bool, device=device),
    )
    torch.testing.assert_close(fallback_u, torch.tensor([0.25], device=device))
    torch.testing.assert_close(fallback_a_l, torch.tensor([1.0], device=device))
    torch.testing.assert_close(fallback_a_n, torch.tensor([1.0], device=device))

    report = {
        "device": torch.cuda.get_device_name(device),
        "semantic_high_eu_rate": high_eu_rate.cpu().tolist(),
        "normalized_uncertainty": normalized_u.cpu().tolist(),
        "learnability_coefficient": a_l.cpu().tolist(),
        "novelty_coefficient": a_n.cpu().tolist(),
        "learnability": learnability.cpu().tolist(),
        "novelty": novelty.cpu().tolist(),
        "log_routing_score": log_score.cpu().tolist(),
        "teacher_weights": weights.cpu().tolist(),
        "fallback_normalized_uncertainty": fallback_u.cpu().tolist(),
        "fallback_learnability_coefficient": fallback_a_l.cpu().tolist(),
        "fallback_novelty_coefficient": fallback_a_n.cpu().tolist(),
    }
    print("EU_ROUTING_GPU_REPORT=" + json.dumps(report, sort_keys=True))
