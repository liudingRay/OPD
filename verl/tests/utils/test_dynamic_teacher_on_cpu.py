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


@pytest.mark.parametrize("mode", ["fixed", "ln_softmax", "single"])
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
        batch.update(teacher_on_student_log_probs_by_teacher=t, teacher_overlap_mask_by_teacher=overlap)
    meta = {
        "micro_batch_size": 1,
        "temperature": 0.6,
        "use_dynamic_bsz": True,
        "teacher_weights": [1, 1, 1],
        "teacher_weight_mode": mode if mode != "single" else "fixed",
        "teacher_ema_beta": 0.9,
        "teacher_selection_tau": 0.1,
    }
    actor = SimpleNamespace(actor_module=SimpleNamespace(eval=lambda: None))
    run = lambda b: ns[method.name](actor, SimpleNamespace(batch=b, meta_info=meta))
    result = run(batch)
    if mode == "ln_softmax":
        w, _ = mt.trajectory_ln_weights(s, t, overlap, mask)
        expected = ((t - s[..., None]) * s.softmax(-1)[..., None] * w[..., None, :]).sum(-1)
        expected = expected.masked_fill(~mask[..., None], 0)
        torch.testing.assert_close(result["dynamic_teacher_weight"], w)
    elif mode == "fixed":
        expected = ((t - s[..., None]) * s.softmax(-1)[..., None]).mean(-1)
    else:
        expected = (t[..., 0] - s) * s.softmax(-1)
    torch.testing.assert_close(result["rm_scores"], expected)
    split = [run({k: v[b : b + 1] for k, v in batch.items()})["rm_scores"] for b in range(3)]
    torch.testing.assert_close(torch.cat(split), result["rm_scores"])
