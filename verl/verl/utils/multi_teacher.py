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

import math
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


def _validate_routing_temperature(tau: float) -> None:
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Teacher selection tau must be finite and positive")


def _trajectory_ln_components(student_log_probs, teacher_log_probs, overlap_mask, response_mask, beta=0.9):
    """Compute the existing overlap-mass Learnability and JS Novelty signals."""
    if not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("EMA beta must be finite and in [0, 1)")
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
    return mass, learnability, novelty


@torch.no_grad()
def trajectory_eu_routing_coefficients(
    student_eu: torch.Tensor,
    response_mask: torch.Tensor,
    semantic_mask: torch.Tensor,
    *,
    quantile: float = 0.95,
    reference_rate: float = 0.05,
    coefficient_gain: float = 15.0,
    normalized_center: float = 0.25,
    coefficient_min: float = 0.1,
    coefficient_max: float = 1.9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute trajectory EU and clipped Learnability/Novelty coefficients.

    The percentile includes every finite valid response token. Formatting-only
    positions are removed only after the percentile threshold is known. The
    numerator counts selected token positions; the denominator remains every
    valid response token. A trajectory without finite EU uses the neutral
    effective uncertainty ``normalized_center`` and coefficients ``(1, 1)``.
    A negative ``coefficient_gain`` reverses the Learnability/Novelty response
    to uncertainty while preserving the same clipping bounds.
    """
    if student_eu.ndim != 2 or response_mask.shape != student_eu.shape or semantic_mask.shape != student_eu.shape:
        raise ValueError("Student EU, response mask, and semantic mask must have identical [B,T] shapes")
    if not 0 <= quantile <= 1:
        raise ValueError("EU quantile must be in [0, 1]")
    if not math.isfinite(reference_rate) or reference_rate <= 0:
        raise ValueError("EU reference rate must be finite and positive")
    if not math.isfinite(coefficient_gain):
        raise ValueError("EU coefficient gain must be finite")
    if not math.isfinite(normalized_center) or normalized_center < 0:
        raise ValueError("EU normalized center must be finite and non-negative")
    if (
        not math.isfinite(coefficient_min)
        or not math.isfinite(coefficient_max)
        or coefficient_min <= 0
        or coefficient_min > 1
        or coefficient_max < 1
        or coefficient_min > coefficient_max
    ):
        raise ValueError("Coefficient bounds must be finite, positive, ordered, and contain 1")

    eu = student_eu.detach().float()
    valid = response_mask.bool()
    semantic = semantic_mask.bool()
    valid_counts = valid.sum(dim=-1)
    finite_valid = valid & torch.isfinite(eu)
    finite_counts = finite_valid.sum(dim=-1)

    if eu.shape[1] == 0:
        rate = eu.new_zeros(eu.shape[0])
        normalized_uncertainty = eu.new_full((eu.shape[0],), normalized_center)
        neutral_coefficient = eu.new_ones(eu.shape[0])
        return rate, normalized_uncertainty, neutral_coefficient, neutral_coefficient.clone()

    quantile_inputs = eu.masked_fill(~finite_valid, torch.nan)
    # Avoid asking nanquantile to reduce an all-NaN row; those trajectories are
    # replaced by the explicit fallback below.
    no_finite = finite_counts == 0
    if no_finite.any():
        quantile_inputs = quantile_inputs.clone()
        quantile_inputs[no_finite, 0] = 0.0
    thresholds = torch.nanquantile(quantile_inputs, quantile, dim=-1)
    semantic_high_eu = finite_valid & semantic & (eu >= thresholds.unsqueeze(-1))
    rate = semantic_high_eu.sum(dim=-1).float() / valid_counts.clamp_min(1).float()
    normalized_uncertainty = rate / reference_rate
    adjustment = coefficient_gain * (normalized_uncertainty - normalized_center)
    learnability_coefficient = (1.0 - adjustment).clamp(coefficient_min, coefficient_max)
    novelty_coefficient = (1.0 + adjustment).clamp(coefficient_min, coefficient_max)

    normalized_uncertainty = torch.where(
        no_finite,
        torch.full_like(normalized_uncertainty, normalized_center),
        normalized_uncertainty,
    )
    learnability_coefficient = torch.where(
        no_finite,
        torch.ones_like(learnability_coefficient),
        learnability_coefficient,
    )
    novelty_coefficient = torch.where(no_finite, torch.ones_like(novelty_coefficient), novelty_coefficient)
    return rate, normalized_uncertainty, learnability_coefficient, novelty_coefficient


@torch.no_grad()
def _standardize_teacher_log_metrics(
    learnability: torch.Tensor,
    novelty: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
    synchronize_stats: bool = False,
) -> dict[str, torch.Tensor]:
    """Teacher-center and batch-standardize log Learnability and Novelty.

    Each valid token is centered over its teacher dimension first. The two
    population standard deviations then pool every valid token-teacher entry
    in the current rollout batch. In distributed training, sufficient
    statistics are summed across the complete actor process group so every DP
    rank uses the same Learnability scale and the same Novelty scale.
    """
    if learnability.shape != novelty.shape or learnability.ndim != 3:
        raise ValueError("Learnability and Novelty must have identical [B,T,M] shapes")
    if response_mask.shape != learnability.shape[:2]:
        raise ValueError("Response mask must align with the [B,T] metric dimensions")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Routing epsilon must be finite and positive")

    valid = response_mask.bool()
    log_learnability = learnability.detach().float().clamp_min(eps).log()
    log_novelty = novelty.detach().float().clamp_min(eps).log()
    centered_log_learnability = log_learnability - log_learnability.mean(dim=-1, keepdim=True)
    centered_log_novelty = log_novelty - log_novelty.mean(dim=-1, keepdim=True)

    expanded_valid = valid.unsqueeze(-1).expand_as(centered_log_learnability)
    learnability_values = centered_log_learnability[expanded_valid].double()
    novelty_values = centered_log_novelty[expanded_valid].double()
    stats = torch.stack(
        (
            learnability_values.sum(),
            learnability_values.square().sum(),
            learnability_values.new_tensor(learnability_values.numel()),
            novelty_values.sum(),
            novelty_values.square().sum(),
            novelty_values.new_tensor(novelty_values.numel()),
        )
    )
    if synchronize_stats and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    def population_scale(offset: int, name: str) -> torch.Tensor:
        total, squared_total, count = stats[offset : offset + 3]
        if count.item() <= 0:
            raise ValueError(f"Cannot compute {name} scale without a valid response token")
        mean = total / count
        variance = (squared_total / count - mean.square()).clamp_min(0.0)
        return variance.sqrt().to(centered_log_learnability.dtype).clamp_min(eps)

    learnability_scale = population_scale(0, "Learnability")
    novelty_scale = population_scale(3, "Novelty")
    normalized_log_learnability = centered_log_learnability / learnability_scale
    normalized_log_novelty = centered_log_novelty / novelty_scale

    # Keep padding neutral in diagnostics and in any downstream computation.
    invalid = ~valid.unsqueeze(-1)
    centered_log_learnability = centered_log_learnability.masked_fill(invalid, 0.0)
    centered_log_novelty = centered_log_novelty.masked_fill(invalid, 0.0)
    normalized_log_learnability = normalized_log_learnability.masked_fill(invalid, 0.0)
    normalized_log_novelty = normalized_log_novelty.masked_fill(invalid, 0.0)
    batch_size = learnability.shape[0]
    return {
        "centered_log_learnability": centered_log_learnability,
        "centered_log_novelty": centered_log_novelty,
        "normalized_log_learnability": normalized_log_learnability,
        "normalized_log_novelty": normalized_log_novelty,
        "learnability_scale": learnability_scale.expand(batch_size).clone(),
        "novelty_scale": novelty_scale.expand(batch_size).clone(),
    }


@torch.no_grad()
def eu_controlled_teacher_weights(
    learnability: torch.Tensor,
    novelty: torch.Tensor,
    learnability_coefficient: torch.Tensor,
    novelty_coefficient: torch.Tensor,
    *,
    response_mask: torch.Tensor | None = None,
    tau: float = 0.1,
    eps: float = 1e-8,
    synchronize_stats: bool = False,
    return_normalization: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply trajectory coefficients to normalized log metrics."""
    _validate_routing_temperature(tau)
    if learnability.shape != novelty.shape or learnability.ndim != 3:
        raise ValueError("Learnability and Novelty must have identical [B,T,M] shapes")
    expected_coefficient_shape = learnability.shape[:1]
    if (
        learnability_coefficient.shape != expected_coefficient_shape
        or novelty_coefficient.shape != expected_coefficient_shape
    ):
        raise ValueError("Routing coefficients must contain one scalar per trajectory")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Routing epsilon must be finite and positive")
    if not torch.isfinite(learnability).all() or not torch.isfinite(novelty).all():
        raise ValueError("Non-finite Learnability or Novelty")
    if (learnability < 0).any() or (novelty < 0).any():
        raise ValueError("Learnability and Novelty must be non-negative")
    if (
        not torch.isfinite(learnability_coefficient).all()
        or not torch.isfinite(novelty_coefficient).all()
        or (learnability_coefficient <= 0).any()
        or (novelty_coefficient <= 0).any()
    ):
        raise ValueError("Routing coefficients must be finite and positive")

    if response_mask is None:
        response_mask = torch.ones(learnability.shape[:2], dtype=torch.bool, device=learnability.device)
    normalization = _standardize_teacher_log_metrics(
        learnability,
        novelty,
        response_mask,
        eps=eps,
        synchronize_stats=synchronize_stats,
    )
    a_l = learnability_coefficient.detach().float()[:, None, None]
    a_n = novelty_coefficient.detach().float()[:, None, None]
    log_routing_score = (
        a_l * normalization["normalized_log_learnability"]
        + a_n * normalization["normalized_log_novelty"]
    )
    weights = torch.softmax(log_routing_score / tau, dim=-1)
    if not torch.isfinite(weights).all() or not torch.isfinite(log_routing_score).all():
        raise ValueError("Non-finite EU-controlled routing score or teacher weights")
    if return_normalization:
        return weights, log_routing_score, normalization
    return weights, log_routing_score


@torch.no_grad()
def trajectory_ln_weights(student_log_probs, teacher_log_probs, overlap_mask, response_mask, beta=0.9, tau=0.1):
    """Detached [B,T,M] legacy LN-softmax weights."""
    _validate_routing_temperature(tau)
    mass, learnability, novelty = _trajectory_ln_components(
        student_log_probs, teacher_log_probs, overlap_mask, response_mask, beta
    )
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


@torch.no_grad()
def trajectory_eu_ln_weights(
    student_log_probs,
    teacher_log_probs,
    overlap_mask,
    response_mask,
    student_eu,
    semantic_mask,
    beta=0.9,
    tau=0.1,
    coefficient_gain=15.0,
    normalized_center=0.25,
    coefficient_min=0.1,
    coefficient_max=1.9,
    synchronize_stats=False,
):
    """Detached EU-adjusted and coefficient-clipped LN routing weights.

    Shapes are student ``[B,T,K]``, teacher/overlap ``[B,T,K,M]``, token
    masks/EU ``[B,T]``, trajectory coefficients ``[B]``, and weights
    ``[B,T,M]``.
    """
    mass, learnability, novelty = _trajectory_ln_components(
        student_log_probs, teacher_log_probs, overlap_mask, response_mask, beta
    )
    semantic_high_eu_rate, normalized_uncertainty, learnability_coefficient, novelty_coefficient = (
        trajectory_eu_routing_coefficients(
            student_eu,
            response_mask,
            semantic_mask,
            coefficient_gain=coefficient_gain,
            normalized_center=normalized_center,
            coefficient_min=coefficient_min,
            coefficient_max=coefficient_max,
        )
    )
    weights, log_routing_score, normalization = eu_controlled_teacher_weights(
        learnability,
        novelty,
        learnability_coefficient,
        novelty_coefficient,
        response_mask=response_mask,
        tau=tau,
        synchronize_stats=synchronize_stats,
        return_normalization=True,
    )
    return weights, {
        "overlap_mass": mass,
        "learnability": learnability,
        "js": novelty,
        "log_routing_score": log_routing_score,
        "weight": weights,
        "semantic_high_eu_rate": semantic_high_eu_rate,
        "eu_normalized_uncertainty": normalized_uncertainty,
        "learnability_coefficient": learnability_coefficient,
        "novelty_coefficient": novelty_coefficient,
        **normalization,
    }
