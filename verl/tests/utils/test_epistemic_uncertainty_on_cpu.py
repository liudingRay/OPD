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

"""CPU tests for the shared LogTokU numerical utility.

Training-time integration is covered by ``test_dynamic_teacher_on_cpu.py``:
the student actor computes EU from its raw logits only for ``eu_ln_softmax``.
Teacher workers deliberately do not compute EU.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "epistemic_uncertainty", ROOT / "verl/utils/epistemic_uncertainty.py"
)
EU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EU)


@pytest.mark.parametrize("shape", [(32,), (2, 5, 32), (2, 3, 4, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_raw_formula_shape_precision_detach_and_no_mutation(shape, dtype):
    logits = torch.arange(math.prod(shape)).reshape(shape).to(dtype).requires_grad_()
    before = logits.detach().clone()
    actual = EU.compute_token_level_eu(logits)
    expected = 16 / (torch.sort(before.float(), dim=-1).values[..., -16:] + 1).sum(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits, before)
    assert actual.shape == logits.shape[:-1]
    assert actual.dtype == torch.float32 and not actual.requires_grad


def test_invalid_inputs_and_unclipped_denominators():
    for logits, top_k in [(torch.zeros(15), 16), (torch.zeros(16), 0), (torch.ones(16).long(), 16)]:
        with pytest.raises(ValueError):
            EU.compute_token_level_eu(logits, top_k)
    assert torch.isinf(EU.compute_token_level_eu(torch.full((16,), -1.0)))
    assert EU.compute_token_level_eu(torch.full((16,), -2.0)).item() == -1.0


def test_statistics_mask_numerical_diagnostics_and_correlation():
    values = torch.tensor(
        [[[1.0, 2.0], [999.0, 999.0], [3.0, 6.0]], [[5.0, 10.0], [float("inf"), -1.0], [999.0, 999.0]]]
    )
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
    rewards = values * 2
    stats = EU.compute_eu_metrics(values, mask, ["4b", "8b"], rewards)
    assert stats["eu/4b/mean"] == 3
    assert stats["eu/4b/std"] == pytest.approx(math.sqrt(8 / 3))
    assert stats["eu/4b/min"] == 1 and stats["eu/4b/max"] == 5
    assert stats["eu/4b/nonfinite_count"] == 1
    assert stats["eu/8b/nonpositive_count"] == 1
    assert stats["eu/4b/valid_tokens"] == 4
    assert stats["eu_reward_corr/4b"] == pytest.approx(1)


def test_teacher_worker_does_not_compute_eu():
    source = (ROOT / "verl/workers/fsdp_workers.py").read_text()
    assert "compute_token_level_eu" not in source
    assert "teacher_eu" not in source
