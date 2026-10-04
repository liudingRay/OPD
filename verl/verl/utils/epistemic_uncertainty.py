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

"""Raw LogTokU diagnostics. These values never enter OPD rewards or weights."""

from collections.abc import Sequence

import torch


@torch.no_grad()
def compute_token_level_eu(logits: torch.Tensor, top_k: int = 16) -> torch.Tensor:
    """Return K / sum(TopK(raw_logits) + 1), with shape logits.shape[:-1].

    Call BEFORE temperature scaling. Only the selected K values are promoted to
    float32, avoiding a full-vocabulary float32 copy. No softmax, clipping,
    epsilon, or cross-teacher normalization is applied. Zero/negative evidence
    denominators retain their raw inf/negative results for numerical diagnostics.
    """
    if logits.ndim < 1 or not logits.is_floating_point():
        raise ValueError("EU requires floating-point raw vocabulary logits")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("EU top_k must be a positive integer no larger than the vocabulary")
    evidence = torch.topk(logits, k=top_k, dim=-1).values.float()
    return top_k / (evidence + 1.0).sum(dim=-1)


@torch.no_grad()
def compute_eu_metrics(
    eu_by_teacher: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_names: Sequence[str],
    rewards_by_teacher: torch.Tensor | None = None,
) -> dict[str, float]:
    """Summarize [B,T,M] EU across the gathered batch, only at loss-valid tokens.

    Nonfinite raw EU is preserved in the tensor and explicitly counted. Moments
    use finite values; nonpositive finite EU is counted and NOT removed. Pearson
    correlation is NaN when undefined (empty, constant, or fewer than two pairs).
    """
    if eu_by_teacher.ndim != 3 or eu_by_teacher.shape[:2] != response_mask.shape:
        raise ValueError("EU must be [B,T,M] aligned with response_mask [B,T]")
    if len(teacher_names) != eu_by_teacher.shape[-1] or len(set(teacher_names)) != len(teacher_names):
        raise ValueError("EU requires one unique name per teacher")
    if rewards_by_teacher is not None and rewards_by_teacher.shape != eu_by_teacher.shape:
        raise ValueError("Teacher token rewards and EU must have identical shapes")
    valid = response_mask.bool()
    metrics = {}
    for index, name in enumerate(teacher_names):
        prefix = f"eu/{name}"
        raw = eu_by_teacher[..., index][valid].float()
        finite = torch.isfinite(raw)
        values = raw[finite]
        metrics[f"{prefix}/valid_tokens"] = float(raw.numel())
        metrics[f"{prefix}/nonfinite_count"] = float((~finite).sum().item())
        metrics[f"{prefix}/nonpositive_count"] = float((values <= 0).sum().item())
        for key in ("mean", "std", "min", "max", "p90"):
            metrics[f"{prefix}/{key}"] = float("nan")
        if values.numel():
            metrics.update(
                {
                    f"{prefix}/mean": values.mean().item(),
                    f"{prefix}/std": values.std(unbiased=False).item(),
                    f"{prefix}/min": values.min().item(),
                    f"{prefix}/max": values.max().item(),
                    f"{prefix}/p90": torch.quantile(values, 0.9).item(),
                }
            )
        if rewards_by_teacher is not None:
            rewards = rewards_by_teacher[..., index][valid].float()
            pairs = finite & torch.isfinite(rewards)
            metrics[f"{prefix}/reward_corr_valid_pairs"] = float(pairs.sum().item())
            corr = float("nan")
            if pairs.sum().item() >= 2:
                x, y = raw[pairs].double(), rewards[pairs].double()
                x, y = x - x.mean(), y - y.mean()
                denominator = x.norm() * y.norm()
                if denominator > 0:
                    corr = ((x * y).sum() / denominator).clamp(-1, 1).item()
            metrics[f"eu_reward_corr/{name}"] = corr
    return metrics
