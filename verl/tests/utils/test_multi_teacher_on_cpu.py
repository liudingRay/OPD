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

import pytest
import torch

from verl.utils.multi_teacher import (
    compute_weighted_teacher_rewards,
    normalize_teacher_weights,
    weighted_teacher_sum,
)


def test_uniform_teacher_sum_is_elementwise_mean():
    teacher_values = torch.tensor(
        [
            [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]],
            [[3.0, 6.0, 9.0], [4.0, 7.0, 10.0]],
        ]
    )

    combined = weighted_teacher_sum(teacher_values, [1.0, 1.0, 1.0])

    torch.testing.assert_close(combined, teacher_values.mean(dim=-1))


def test_teacher_weights_are_normalized_before_combining():
    teacher_values = torch.tensor([[2.0, 6.0, 10.0]])

    combined = weighted_teacher_sum(teacher_values, [1.0, 2.0, 1.0])

    torch.testing.assert_close(combined, torch.tensor([6.0]))


def test_opd_rewards_are_computed_per_teacher_before_uniform_average():
    student_log_probs = torch.tensor([[[-1.0, -2.0]]])
    teacher_log_probs = torch.tensor([[[[-0.5, -1.5, -2.5], [-2.5, -1.5, -0.5]]]])
    token_weights = torch.tensor([[[0.75, 0.25]]])

    combined, rewards_by_teacher = compute_weighted_teacher_rewards(
        student_log_probs,
        teacher_log_probs,
        token_weights,
        [1.0, 1.0, 1.0],
    )

    expected_by_teacher = -(
        student_log_probs.unsqueeze(-1) - teacher_log_probs
    ) * token_weights.unsqueeze(-1)
    torch.testing.assert_close(rewards_by_teacher, expected_by_teacher)
    torch.testing.assert_close(combined, expected_by_teacher.mean(dim=-1))


def test_dynamic_weights_can_vary_by_batch_item():
    teacher_values = torch.tensor([[[1.0, 3.0, 9.0]], [[1.0, 3.0, 9.0]]])
    weights = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]])

    combined = weighted_teacher_sum(teacher_values, weights)

    torch.testing.assert_close(combined, torch.tensor([[1.0], [9.0]]))


@pytest.mark.parametrize("weights", ([1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 2.0]))
def test_invalid_teacher_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        normalize_teacher_weights(weights, num_teachers=3)
