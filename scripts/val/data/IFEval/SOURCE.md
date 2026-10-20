# IFEval data provenance

- Dataset: `google/IFEval`
- Upstream revision: `966cd89545d6b6acfd7638bc708b98261ca58e84`
- Upstream file: `ifeval_input_data.jsonl`
- Source URL: <https://huggingface.co/datasets/google/IFEval/blob/966cd89545d6b6acfd7638bc708b98261ca58e84/ifeval_input_data.jsonl>
- Downloaded: 2026-10-19
- License: Apache-2.0, as declared by the upstream dataset card
- Paper: <https://arxiv.org/abs/2311.07911>

The local JSONL is an unmodified copy of the file at the pinned upstream
revision. It contains 541 prompts and 834 verifiable-instruction instances.
Each record contains `key`, `prompt`, `instruction_id_list`, and `kwargs`.

Before using a replacement dataset revision, update this file and
`SHA256SUMS`, then rerun the structural validation for row count, unique keys,
unique prompts, required fields, and matching lengths of `instruction_id_list`
and `kwargs`.
