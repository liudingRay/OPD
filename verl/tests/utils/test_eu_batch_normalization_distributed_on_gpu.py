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

"""Two-rank numerical check for globally standardized EU teacher routing.

Run with::

    torchrun --standalone --nnodes=1 --nproc-per-node=2 \
      verl/tests/utils/test_eu_batch_normalization_distributed_on_gpu.py

Set ``EU_TEST_DEVICE=cpu`` to exercise the same collective with Gloo.
"""

import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("multi_teacher", ROOT / "verl/utils/multi_teacher.py")
multi_teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_teacher)


def main() -> None:
    force_cpu = os.environ.get("EU_TEST_DEVICE") == "cpu"
    if not force_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; set EU_TEST_DEVICE=cpu for the Gloo check")
    backend = "gloo" if force_cpu else "nccl"
    dist.init_process_group(backend=backend)
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError(f"Expected exactly two ranks, got {dist.get_world_size()}")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        if force_cpu:
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

        all_learnability = torch.tensor(
            [
                [[0.20, 0.40, 0.80], [0.70, 0.50, 0.10]],
                [[0.60, 0.30, 0.15], [0.10, 0.80, 0.40]],
            ],
            device=device,
        )
        all_novelty = torch.tensor(
            [
                [[0.90, 0.30, 0.10], [0.20, 0.50, 0.80]],
                [[0.15, 0.45, 0.90], [0.80, 0.20, 0.40]],
            ],
            device=device,
        )
        all_response_mask = torch.tensor([[True, False], [True, True]], device=device)
        all_a_l = torch.tensor([1.25, 0.50], device=device)
        all_a_n = torch.tensor([0.75, 1.50], device=device)

        local_slice = slice(rank, rank + 1)
        weights, score, diagnostics = multi_teacher.eu_controlled_teacher_weights(
            all_learnability[local_slice],
            all_novelty[local_slice],
            all_a_l[local_slice],
            all_a_n[local_slice],
            response_mask=all_response_mask[local_slice],
            tau=0.1,
            synchronize_stats=True,
            return_normalization=True,
        )

        centered_l = all_learnability.log() - all_learnability.log().mean(dim=-1, keepdim=True)
        centered_n = all_novelty.log() - all_novelty.log().mean(dim=-1, keepdim=True)
        expected_l_scale = centered_l[all_response_mask].std(unbiased=False)
        expected_n_scale = centered_n[all_response_mask].std(unbiased=False)
        expected_normalized_l = centered_l[local_slice] / expected_l_scale
        expected_normalized_n = centered_n[local_slice] / expected_n_scale
        local_mask = all_response_mask[local_slice]
        expected_normalized_l = expected_normalized_l.masked_fill(~local_mask[..., None], 0.0)
        expected_normalized_n = expected_normalized_n.masked_fill(~local_mask[..., None], 0.0)
        expected_score = (
            all_a_l[local_slice, None, None] * expected_normalized_l
            + all_a_n[local_slice, None, None] * expected_normalized_n
        )
        expected_weights = torch.softmax(expected_score / 0.1, dim=-1)

        torch.testing.assert_close(diagnostics["learnability_scale"][0], expected_l_scale)
        torch.testing.assert_close(diagnostics["novelty_scale"][0], expected_n_scale)
        torch.testing.assert_close(diagnostics["normalized_log_learnability"], expected_normalized_l)
        torch.testing.assert_close(diagnostics["normalized_log_novelty"], expected_normalized_n)
        torch.testing.assert_close(score, expected_score)
        torch.testing.assert_close(weights, expected_weights)

        local_scales = torch.stack(
            (diagnostics["learnability_scale"][0], diagnostics["novelty_scale"][0])
        )
        gathered_scales = [torch.empty_like(local_scales) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_scales, local_scales)
        for other_scales in gathered_scales:
            torch.testing.assert_close(other_scales, local_scales)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if rank == 0:
            report = {
                "status": "PASS",
                "backend": backend,
                "world_size": dist.get_world_size(),
                "valid_token_count": int(all_response_mask.sum().item()),
                "teacher_count": all_learnability.shape[-1],
                "learnability_scale": expected_l_scale.item(),
                "novelty_scale": expected_n_scale.item(),
                "rank0_weights": weights[local_mask].detach().cpu().tolist(),
            }
            print("EU_BATCH_NORMALIZATION_DISTRIBUTED_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
