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

import ast
import importlib.util
from pathlib import Path

import pytest
import torch

# Load the pure numerical utility without initializing Ray/verl's GPU stack.
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("multi_teacher", ROOT / "verl/utils/multi_teacher.py")
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)


def fixture_inputs():
    torch.manual_seed(8)
    s = torch.log_softmax(torch.randn(3, 17, 8), -1)[..., :4]
    t = torch.log_softmax(torch.randn(3, 17, 8, 3), -2)[..., :4, :]
    overlap = torch.rand_like(t) > 0.5
    mask = torch.rand(3, 17) > 0.3
    mask[2] = False
    return s, t, overlap, mask


@pytest.mark.parametrize("beta", [0.0, 0.9, 0.999])
def test_scan_matches_sequential_masked_ema_and_resets(beta):
    s, t, overlap, mask = fixture_inputs()
    weights, d = mt.trajectory_ln_weights(s, t, overlap, mask, beta=beta)
    expected = torch.zeros_like(d["learnability"])
    for b in range(s.shape[0]):
        state = None
        for pos in range(s.shape[1]):
            if mask[b, pos]:
                mass = d["overlap_mass"][b, pos]
                state = mass if state is None else beta * state + (1 - beta) * mass
                expected[b, pos] = state
    torch.testing.assert_close(d["learnability"], expected)
    # Splitting/reordering trajectories must not change weights or carry state.
    for b in range(s.shape[0]):
        alone, _ = mt.trajectory_ln_weights(s[b : b + 1], t[b : b + 1], overlap[b : b + 1], mask[b : b + 1], beta)
        torch.testing.assert_close(alone[0], weights[b])


def test_real_worker_token_alignment_and_raw_mass():
    # Execute the actual worker's pure scoring method without importing FSDP.
    tree = ast.parse((ROOT / "verl/workers/fsdp_workers.py").read_text())
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compute_teacher_top_k_log_probs"
    )
    method.decorator_list = []
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "scorer", "exec"), namespace)
    teacher_logits = torch.tensor([[0.1, 0.2, 0.6, 0.1]]).log()
    ids = torch.tensor([[1, 0]])  # Student ranking differs from teacher ranking.
    scored = namespace[method.name](None, teacher_logits, ids, 2)
    t = scored[0].reshape(1, 1, 2, 1)
    overlap = scored[2].reshape(1, 1, 2, 1)
    s = torch.tensor([[[0.5, 0.3]]]).log()
    _, d = mt.trajectory_ln_weights(s, t, overlap, torch.ones(1, 1))
    torch.testing.assert_close(t.flatten().exp(), torch.tensor([0.2, 0.1]))
    torch.testing.assert_close(d["overlap_mass"], torch.tensor([[[0.5]]]))
    assert d["overlap_mass"].item() != pytest.approx(0.5 / 0.8)


def test_js_identity_symmetry_extremes_and_conditional_support():
    s = torch.tensor([[[-10000.0, -10001.0]]])
    mask = torch.ones(1, 1)
    overlap = torch.ones(1, 1, 2, 1)
    _, same = mt.trajectory_ln_weights(s, (s - 100)[..., None], overlap, mask)
    torch.testing.assert_close(same["js"], torch.zeros(1, 1, 1))
    a = torch.tensor([[[0.0, -10000.0]]])
    b = a.flip(-1)
    _, ab = mt.trajectory_ln_weights(a, b[..., None], overlap, mask)
    _, ba = mt.trajectory_ln_weights(b, a[..., None], overlap, mask)
    torch.testing.assert_close(ab["js"], ba["js"])
    torch.testing.assert_close(ab["js"], torch.ones(1, 1, 1))


def test_softmax_temperature_zero_scores_detachment_and_reward():
    s, t, overlap, mask = fixture_inputs()
    s.requires_grad_()
    t.requires_grad_()
    cold, d = mt.trajectory_ln_weights(s, t, overlap, mask, tau=0.01)
    warm, _ = mt.trajectory_ln_weights(s, t, overlap, mask, tau=1)
    torch.testing.assert_close(cold.sum(-1), torch.ones_like(mask, dtype=torch.float))
    torch.testing.assert_close(cold, torch.softmax(d["utility"] / 0.01, -1))
    assert (cold.max(-1).values >= warm.max(-1).values - 1e-6).all()
    assert all(not x.requires_grad for x in d.values())
    uniform, _ = mt.trajectory_ln_weights(s, t, torch.zeros_like(overlap), mask)
    torch.testing.assert_close(uniform, torch.full_like(uniform, 1 / 3))
    reward, individual = mt.compute_weighted_teacher_rewards(s, t, s.softmax(-1), cold.unsqueeze(-2))
    expected = s.softmax(-1)[..., None] * (t - s[..., None])
    torch.testing.assert_close(individual, expected)
    torch.testing.assert_close(reward, (expected * cold.unsqueeze(-2)).sum(-1))
    single, _ = mt.trajectory_ln_weights(s, t[..., :1], overlap[..., :1], mask)
    torch.testing.assert_close(single, torch.ones_like(single))
    fixed, _ = mt.compute_weighted_teacher_rewards(s, t, s.softmax(-1), [1, 1, 1])
    torch.testing.assert_close(fixed, expected.mean(-1))


def test_trajectory_eu_rate_uses_all_valid_tokens_then_filters_semantics():
    student_eu = torch.arange(20, dtype=torch.float).repeat(3, 1)
    student_eu[2] = float("nan")
    response_mask = torch.ones_like(student_eu, dtype=torch.bool)
    semantic_mask = torch.ones_like(response_mask)
    semantic_mask[1, -1] = False  # The sole >=p95 token is formatting-only.

    rate, normalized_u, a_l, a_n = mt.trajectory_eu_routing_coefficients(
        student_eu, response_mask, semantic_mask
    )
    torch.testing.assert_close(rate, torch.tensor([0.05, 0.0, 0.0]))
    torch.testing.assert_close(normalized_u, torch.tensor([1.0, 0.0, 0.25]))
    torch.testing.assert_close(a_l, torch.tensor([0.1, 1.9, 1.0]))
    torch.testing.assert_close(a_n, torch.tensor([1.9, 0.1, 1.0]))

    # Padding never enters either the percentile or denominator.
    response_mask[0, :10] = False
    rate, normalized_u, a_l, a_n = mt.trajectory_eu_routing_coefficients(
        student_eu, response_mask, semantic_mask
    )
    torch.testing.assert_close(rate[0], torch.tensor(0.1))
    torch.testing.assert_close(normalized_u[0], torch.tensor(2.0))
    torch.testing.assert_close(a_l[0], torch.tensor(0.1))
    torch.testing.assert_close(a_n[0], torch.tensor(1.9))


def test_negative_eu_coefficient_gain_reverses_learnability_and_novelty():
    student_eu = torch.arange(20, dtype=torch.float).repeat(2, 1)
    response_mask = torch.ones_like(student_eu, dtype=torch.bool)
    semantic_mask = torch.ones_like(response_mask)
    semantic_mask[0, -1] = False

    positive_rate, positive_u, positive_a_l, positive_a_n = mt.trajectory_eu_routing_coefficients(
        student_eu,
        response_mask,
        semantic_mask,
        coefficient_gain=2.0,
    )
    negative_rate, negative_u, negative_a_l, negative_a_n = mt.trajectory_eu_routing_coefficients(
        student_eu,
        response_mask,
        semantic_mask,
        coefficient_gain=-2.0,
    )

    torch.testing.assert_close(negative_rate, positive_rate)
    torch.testing.assert_close(negative_u, positive_u)
    torch.testing.assert_close(negative_a_l, positive_a_n)
    torch.testing.assert_close(negative_a_n, positive_a_l)


@pytest.mark.parametrize("coefficient_gain", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_eu_coefficient_gain_fails(coefficient_gain):
    student_eu = torch.zeros((1, 20), dtype=torch.float)
    response_mask = torch.ones_like(student_eu, dtype=torch.bool)
    semantic_mask = torch.ones_like(response_mask)

    with pytest.raises(ValueError, match="EU coefficient gain must be finite"):
        mt.trajectory_eu_routing_coefficients(
            student_eu,
            response_mask,
            semantic_mask,
            coefficient_gain=coefficient_gain,
        )


@pytest.mark.parametrize(
    ("coefficient_min", "coefficient_max"),
    [(0.0, 1.9), (1.1, 1.9), (0.1, 0.9), (2.0, 1.9)],
)
def test_invalid_eu_coefficient_bounds_fail(coefficient_min, coefficient_max):
    student_eu = torch.zeros((1, 20), dtype=torch.float)
    response_mask = torch.ones_like(student_eu, dtype=torch.bool)
    semantic_mask = torch.ones_like(response_mask)

    with pytest.raises(ValueError, match="Coefficient bounds"):
        mt.trajectory_eu_routing_coefficients(
            student_eu,
            response_mask,
            semantic_mask,
            coefficient_min=coefficient_min,
            coefficient_max=coefficient_max,
        )


def test_eu_adjusted_and_clipped_routing_sanity_cases():
    learnability = torch.tensor([[[0.2, 0.4, 0.8]]]).repeat(3, 1, 1)
    novelty = torch.tensor([[[0.9, 0.3, 0.1]]]).repeat(3, 1, 1)
    a_l = torch.tensor([1.25, 1.0, 0.5])
    a_n = torch.tensor([0.75, 1.0, 1.5])

    weights, scores = mt.eu_controlled_teacher_weights(learnability, novelty, a_l, a_n, tau=0.1)
    centered_l = learnability.log() - learnability.log().mean(dim=-1, keepdim=True)
    centered_n = novelty.log() - novelty.log().mean(dim=-1, keepdim=True)
    normalized_l = centered_l / centered_l.std(unbiased=False)
    normalized_n = centered_n / centered_n.std(unbiased=False)
    expected_scores = a_l[:, None, None] * normalized_l + a_n[:, None, None] * normalized_n
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(weights, torch.softmax(expected_scores / 0.1, dim=-1))

    identical = torch.full((2, 4, 3), 0.4)
    uniform, _ = mt.eu_controlled_teacher_weights(
        identical,
        identical,
        torch.tensor([1.25, 0.5]),
        torch.tensor([0.75, 1.5]),
    )
    torch.testing.assert_close(uniform, torch.full_like(uniform, 1 / 3))

    cold, _ = mt.eu_controlled_teacher_weights(
        learnability[:1], novelty[:1], a_l[:1], a_n[:1], tau=0.05
    )
    warm, _ = mt.eu_controlled_teacher_weights(
        learnability[:1], novelty[:1], a_l[:1], a_n[:1], tau=0.5
    )
    assert cold.argmax(-1).item() == warm.argmax(-1).item()
    assert cold.max().item() > warm.max().item()


def test_log_metrics_use_one_batch_scale_per_metric_and_ignore_padding():
    learnability = torch.tensor(
        [
            [[0.2, 0.4, 0.8], [0.9, 0.3, 0.1]],
            [[0.7, 0.2, 0.1], [0.25, 0.5, 1.0]],
        ]
    )
    novelty = torch.tensor(
        [
            [[0.8, 0.5, 0.1], [0.1, 0.3, 0.9]],
            [[0.2, 0.3, 0.9], [0.9, 0.4, 0.2]],
        ]
    )
    response_mask = torch.tensor([[True, False], [True, True]])

    d = mt._standardize_teacher_log_metrics(learnability, novelty, response_mask)
    centered_l = learnability.log() - learnability.log().mean(dim=-1, keepdim=True)
    centered_n = novelty.log() - novelty.log().mean(dim=-1, keepdim=True)
    expected_l_scale = centered_l[response_mask].std(unbiased=False)
    expected_n_scale = centered_n[response_mask].std(unbiased=False)

    torch.testing.assert_close(d["learnability_scale"], expected_l_scale.expand(2))
    torch.testing.assert_close(d["novelty_scale"], expected_n_scale.expand(2))
    torch.testing.assert_close(
        d["normalized_log_learnability"][response_mask], centered_l[response_mask] / expected_l_scale
    )
    torch.testing.assert_close(d["normalized_log_novelty"][response_mask], centered_n[response_mask] / expected_n_scale)
    torch.testing.assert_close(
        d["normalized_log_learnability"][response_mask].std(unbiased=False), torch.tensor(1.0)
    )
    torch.testing.assert_close(d["normalized_log_novelty"][response_mask].std(unbiased=False), torch.tensor(1.0))
    assert not d["normalized_log_learnability"][~response_mask].any()
    assert not d["normalized_log_novelty"][~response_mask].any()


def test_log_metric_scales_all_reduce_sufficient_statistics(monkeypatch):
    learnability = torch.tensor([[[0.2, 0.4, 0.8]], [[0.7, 0.2, 0.1]]])
    novelty = torch.tensor([[[0.8, 0.5, 0.1]], [[0.2, 0.3, 0.9]]])
    response_mask = torch.ones(2, 1, dtype=torch.bool)

    def moments(local_l, local_n):
        centered_l = local_l.log() - local_l.log().mean(dim=-1, keepdim=True)
        centered_n = local_n.log() - local_n.log().mean(dim=-1, keepdim=True)
        return torch.stack(
            (
                centered_l.double().sum(),
                centered_l.double().square().sum(),
                centered_l.new_tensor(centered_l.numel(), dtype=torch.double),
                centered_n.double().sum(),
                centered_n.double().square().sum(),
                centered_n.new_tensor(centered_n.numel(), dtype=torch.double),
            )
        )

    monkeypatch.setattr(mt.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(mt.torch.distributed, "is_initialized", lambda: True)
    rank_scales = []
    for local_rank, remote_rank in ((0, 1), (1, 0)):
        remote_stats = moments(learnability[remote_rank : remote_rank + 1], novelty[remote_rank : remote_rank + 1])

        def fake_all_reduce(stats, op, remote_stats=remote_stats):
            stats.add_(remote_stats)

        monkeypatch.setattr(mt.torch.distributed, "all_reduce", fake_all_reduce)
        d = mt._standardize_teacher_log_metrics(
            learnability[local_rank : local_rank + 1],
            novelty[local_rank : local_rank + 1],
            response_mask[local_rank : local_rank + 1],
            synchronize_stats=True,
        )
        rank_scales.append(torch.stack((d["learnability_scale"][0], d["novelty_scale"][0])))

    global_centered_l = learnability.log() - learnability.log().mean(dim=-1, keepdim=True)
    global_centered_n = novelty.log() - novelty.log().mean(dim=-1, keepdim=True)
    expected = torch.stack((global_centered_l.std(unbiased=False), global_centered_n.std(unbiased=False)))
    torch.testing.assert_close(rank_scales[0], expected)
    torch.testing.assert_close(rank_scales[1], expected)


def test_actual_student_forward_computes_eu_before_temperature():
    from types import SimpleNamespace

    eu_spec = importlib.util.spec_from_file_location(
        "epistemic_uncertainty", ROOT / "verl/utils/epistemic_uncertainty.py"
    )
    eu_module = importlib.util.module_from_spec(eu_spec)
    eu_spec.loader.exec_module(eu_module)
    tree = ast.parse((ROOT / "verl/workers/actor/dp_actor.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_forward_micro_batch"
    )
    method.decorator_list = []
    namespace = {
        "torch": torch,
        "compute_token_level_eu": eu_module.compute_token_level_eu,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "actor", "exec"), namespace)

    class Model:
        def __call__(self, input_ids, **kwargs):
            positions = torch.arange(input_ids.shape[1], dtype=torch.float)[None, :, None]
            vocabulary = torch.arange(32, dtype=torch.float)[None, None, :]
            return SimpleNamespace(logits=(vocabulary + 2 * positions).expand(input_ids.shape[0], -1, -1).clone())

    actor = SimpleNamespace(
        actor_module=Model(),
        device_name="cpu",
        use_remove_padding=False,
        use_fused_kernels=False,
        config=SimpleNamespace(entropy_checkpointing=False),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
        "attention_mask": torch.ones(1, 6),
        "position_ids": torch.arange(6).unsqueeze(0),
        "responses": torch.tensor([[4, 5, 6]]),
    }
    _, _, _, _, student_eu = namespace[method.name](
        actor, batch, temperature=5.0, top_k=4, compute_eu=True
    )
    raw = actor.actor_module(batch["input_ids"]).logits[:, -4:-1]
    torch.testing.assert_close(student_eu, eu_module.compute_token_level_eu(raw))
    assert not torch.allclose(student_eu, eu_module.compute_token_level_eu(raw / 5.0))


@pytest.mark.parametrize("beta,tau", [(1, 0.1), (-0.1, 0.1), (float("nan"), 0.1), (0.9, 0), (0.9, float("inf"))])
def test_bad_parameters(beta, tau):
    with pytest.raises(ValueError):
        mt.trajectory_ln_weights(*fixture_inputs(), beta=beta, tau=tau)


def test_nonfinite_valid_scores_fail_but_padding_is_ignored():
    s, t, overlap, mask = fixture_inputs()
    s[~mask] = float("nan")
    t[~mask] = float("inf")
    weights, _ = mt.trajectory_ln_weights(s, t, overlap, mask)
    assert torch.isfinite(weights).all()
    t[mask] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        mt.trajectory_ln_weights(s, t, overlap, mask)


@pytest.mark.parametrize("mode", ["fixed", "ln_softmax", "eu_ln_softmax", "single"])
def test_actual_actor_reward_path_and_batch_partition(mode):
    from types import SimpleNamespace

    class Data:
        @staticmethod
        def from_dict(tensors):
            return tensors

    tree = ast.parse((ROOT / "verl/workers/actor/dp_actor.py").read_text())
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute_distillation_reward"
    )
    method.decorator_list = []
    ns = {
        "torch": torch,
        "DataProto": Data,
        "get_device_id": lambda: "cpu",
        "compute_weighted_teacher_rewards": mt.compute_weighted_teacher_rewards,
        "trajectory_eu_ln_weights": mt.trajectory_eu_ln_weights,
        "trajectory_ln_weights": mt.trajectory_ln_weights,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "actor", "exec"), ns)
    s, t, overlap, mask = fixture_inputs()
    batch = {
        "student_top_k_ids": torch.zeros_like(s, dtype=torch.long),
        "student_top_k_log_probs": s,
        "response_mask": mask,
    }
    if mode == "single":
        batch["teacher_on_student_log_probs"] = t[..., 0]
    else:
        batch.update(
            teacher_on_student_log_probs_by_teacher=t,
            teacher_overlap_mask_by_teacher=overlap,
            student_eu=torch.arange(s.shape[1], dtype=torch.float).expand(s.shape[0], -1),
            semantic_token_mask=torch.ones_like(mask),
        )
    meta = {
        "micro_batch_size": 1,
        "temperature": 0.6,
        "use_dynamic_bsz": True,
        "teacher_weights": [1, 1, 1],
        "teacher_weight_mode": mode if mode != "single" else "fixed",
        "teacher_ema_beta": 0.9,
        "teacher_selection_tau": 0.1,
        "teacher_eu_coefficient_gain": 2.0,
        "teacher_eu_normalized_center": 0.25,
        "teacher_eu_coefficient_min": 0.2,
        "teacher_eu_coefficient_max": 1.8,
    }
    actor = SimpleNamespace(actor_module=SimpleNamespace(eval=lambda: None))
    run = lambda b: ns[method.name](actor, SimpleNamespace(batch=b, meta_info=meta))
    result = run(batch)
    if mode == "ln_softmax":
        w, _ = mt.trajectory_ln_weights(s, t, overlap, mask)
        expected = ((t - s[..., None]) * s.softmax(-1)[..., None] * w[..., None, :]).sum(-1)
        expected = expected.masked_fill(~mask[..., None], 0)
        torch.testing.assert_close(result["dynamic_teacher_weight"], w)
    elif mode == "eu_ln_softmax":
        w, d = mt.trajectory_eu_ln_weights(
            s,
            t,
            overlap,
            mask,
            batch["student_eu"],
            batch["semantic_token_mask"],
            coefficient_gain=2.0,
            coefficient_min=0.2,
            coefficient_max=1.8,
        )
        expected = ((t - s[..., None]) * s.softmax(-1)[..., None] * w[..., None, :]).sum(-1)
        expected = expected.masked_fill(~mask[..., None], 0)
        torch.testing.assert_close(result["dynamic_teacher_weight"], w)
        torch.testing.assert_close(
            result["dynamic_teacher_eu_normalized_uncertainty"], d["eu_normalized_uncertainty"]
        )
        torch.testing.assert_close(
            result["dynamic_teacher_learnability_coefficient"], d["learnability_coefficient"]
        )
        torch.testing.assert_close(result["dynamic_teacher_novelty_coefficient"], d["novelty_coefficient"])
        torch.testing.assert_close(result["dynamic_teacher_learnability_scale"], d["learnability_scale"])
        torch.testing.assert_close(result["dynamic_teacher_novelty_scale"], d["novelty_scale"])
        torch.testing.assert_close(
            result["dynamic_teacher_normalized_log_learnability"][mask].std(unbiased=False), torch.tensor(1.0)
        )
        torch.testing.assert_close(
            result["dynamic_teacher_normalized_log_novelty"][mask].std(unbiased=False), torch.tensor(1.0)
        )
        assert result["dynamic_teacher_log_routing_score"].shape == (3, 17, 3)
    elif mode == "fixed":
        expected = ((t - s[..., None]) * s.softmax(-1)[..., None]).mean(-1)
    else:
        expected = (t[..., 0] - s) * s.softmax(-1)
    torch.testing.assert_close(result["rm_scores"], expected)
    if mode != "eu_ln_softmax":
        split = [run({k: v[b : b + 1] for k, v in batch.items()})["rm_scores"] for b in range(3)]
        torch.testing.assert_close(torch.cat(split), result["rm_scores"])
