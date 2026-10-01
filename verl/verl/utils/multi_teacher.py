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

from collections.abc import Sequence

import torch


def normalize_teacher_weights(
    weights: Sequence[float] | torch.Tensor,
    num_teachers: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Validate and normalize one non-negative weight per teacher."""
    normalized = torch.as_tensor(weights, device=device, dtype=dtype or torch.float32)
    if normalized.ndim < 1 or normalized.shape[-1] != num_teachers:
        raise ValueError(f"Expected a final dimension of {num_teachers}, got shape {tuple(normalized.shape)}")
    if not torch.isfinite(normalized).all():
        raise ValueError("Teacher weights must all be finite")
    if (normalized < 0).any():
        raise ValueError("Teacher weights must be non-negative")

    weight_sum = normalized.sum(dim=-1, keepdim=True)
    if (weight_sum <= 0).any():
        raise ValueError("Teacher weights must have a positive sum along the teacher dimension")
    return normalized / weight_sum


def weighted_teacher_sum(values: torch.Tensor, weights: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """Combine a tensor whose final dimension indexes teachers.

    Besides a fixed ``[num_teachers]`` vector, ``weights`` may contain leading
    batch/token dimensions as long as they broadcast to ``values``.
    """
    normalized = normalize_teacher_weights(
        weights,
        values.shape[-1],
        device=values.device,
        dtype=values.dtype,
    )
    return torch.sum(values * normalized, dim=-1)


def compute_weighted_teacher_rewards(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    token_weights: torch.Tensor,
    teacher_weights: Sequence[float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute each teacher's OPD reward, then aggregate on the teacher axis."""
    expected_teacher_shape = (*student_log_probs.shape, teacher_log_probs.shape[-1])
    if teacher_log_probs.shape != expected_teacher_shape:
        raise ValueError(
            f"Expected teacher log probabilities with shape {expected_teacher_shape}, "
            f"got {tuple(teacher_log_probs.shape)}"
        )
    if token_weights.shape != student_log_probs.shape:
        raise ValueError(
            f"Expected token weights with shape {tuple(student_log_probs.shape)}, got {tuple(token_weights.shape)}"
        )

    rewards_by_teacher = -(student_log_probs.unsqueeze(-1) - teacher_log_probs) * token_weights.unsqueeze(-1)
    return weighted_teacher_sum(rewards_by_teacher, teacher_weights), rewards_by_teacher


@torch.no_grad()
def trajectory_ln_weights(student_log_probs, teacher_log_probs, overlap_mask, response_mask, beta=0.9, tau=0.1):
    """Detached [B,T,M] LN-softmax weights; EMA resets for every trajectory.

    Inputs retain full-vocabulary log probabilities. Only JS conditions on the
    student candidate set. Invalid positions neither initialize nor update EMA.
    """
    import math

    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("EMA beta must be finite and in [0, 1)")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Teacher selection tau must be finite and positive")
    if student_log_probs.ndim != 3 or teacher_log_probs.ndim != 4:
        raise ValueError("Expected student [B,T,K] and teacher [B,T,K,M]")
    if teacher_log_probs.shape[:-1] != student_log_probs.shape or overlap_mask.shape != teacher_log_probs.shape:
        raise ValueError("Teacher candidates and overlap must align with student candidates")
    if response_mask.shape != student_log_probs.shape[:2] or min(teacher_log_probs.shape[-2:]) < 1:
        raise ValueError("Invalid response mask or empty candidate/teacher dimension")
    valid = response_mask.bool()
    if not torch.isfinite(overlap_mask[valid]).all():
        raise ValueError("Non-finite overlap mask at valid response positions")
    s = student_log_probs.detach().float()
    t = teacher_log_probs.detach().float()
    if not torch.isfinite(s[valid]).all() or not torch.isfinite(t[valid]).all():
        raise ValueError("Non-finite scoring log probabilities at valid response positions")
    s = torch.where(valid[..., None], s, 0.0)
    t = torch.where(valid[..., None, None], t, 0.0)
    mass = (s.exp()[..., None] * overlap_mask.bool()).sum(-2)
    mass = mass.masked_fill(~valid[..., None], 0.0)
    ls = torch.log_softmax(s, dim=-1)[..., None]
    lt = torch.log_softmax(t, dim=-2)
    lm = torch.logaddexp(ls, lt) - math.log(2)
    novelty = 0.5 * (ls.exp() * (ls - lm) + lt.exp() * (lt - lm)).sum(-2) / math.log(2)
    novelty = novelty.clamp(0, 1).masked_fill(~valid[..., None], 0.0)

    # Parallel affine prefix scan: composition of x -> a*x+b. This avoids
    # thousands of per-token GPU launches and unstable inverse beta powers.
    first = valid & (valid.long().cumsum(1) == 1)
    a = torch.where(valid, beta, 1.0).masked_fill(first, 0.0)[..., None]
    learnability = torch.where(first[..., None], mass, (1 - beta) * mass)
    offset = 1
    while offset < s.shape[1]:
        learnability = torch.cat(
            (learnability[:, :offset], learnability[:, offset:] + a[:, offset:] * learnability[:, :-offset]), dim=1
        )
        a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), dim=1)
        offset *= 2
    learnability = learnability.masked_fill(~valid[..., None], 0.0)
    # Reliable is implicitly one; any future reliability factor belongs here.
    utility = learnability * novelty
    weights = torch.softmax(utility / tau, dim=-1)
    if not torch.isfinite(weights).all() or not torch.isfinite(utility).all():
        raise ValueError("Non-finite LN utility or teacher weights")
    return weights, {
        "overlap_mass": mass,
        "learnability": learnability,
        "js": novelty,
        "utility": utility,
        "weight": weights,
    }
