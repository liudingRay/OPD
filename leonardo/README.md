# Leonardo-Specific OPD Guidance

This directory contains the current CINECA Leonardo launchers and older top-level training scripts retained with the Leonardo workflow for reference. Read this file together with the workspace `AGENTS.md` before editing or running a launcher. The shared scientific definition of Curriculum OPD remains in `AGENTS.md`.

## Leonardo Script Map

| Script | Function |
| --- | --- |
| `leonardo/curriculum-opd` | Primary Leonardo two-node, four-A100-per-node Curriculum OPD launcher. It contains the current Stage-1/2/3 training configuration, starts the shared Ray cluster, and invokes `python -m verl.trainer.main_ppo` directly. |
| `leonardo/opd_pilot_2node_4xa100.sh` | Older two-node OPD reproduction settings for an already-started Ray cluster; it is not called by the current `curriculum-opd`. |
| `leonardo/opd_pilot_4xa100.sh` | Older single-node, four-A100 OPD pilot settings. |
| `leonardo/on_policy_distillation.sh` | Legacy general OPD wrapper used by the old `opd_pilot_*` scripts. It converts environment variables into `verl.trainer.main_ppo` Hydra overrides, enables the teacher reward model and token-level OPD estimator, manages an optional local Ray head, and retains obsolete `account=test` / `partition=TEST1` defaults. Current Curriculum OPD launchers do not call it. |
| `leonardo/grpo.sh` | Legacy monolithic GRPO reference. It uses `ADV_ESTIMATOR=grpo`, disables the reward model and thinking mode, assumes eight GPUs, starts its own Ray head, logs to SwanLab, and retains obsolete `account=test` / `partition=TEST1` directives. It is not the current Leonardo GRPO launcher and must not be submitted unchanged. |
| `leonardo/submit_eval_qwen3_thinking_baselines_1xa100.sbatch` | Serial thinking-mode `avg@16` evaluation of official Qwen3-1.7B/4B/8B on Leonardo. |
| `leonardo/submit_eval_qwen3_model_1xa100.sbatch` | Evaluate one local Qwen3 model on Leonardo, selected with `MODEL_NAME` and `MODEL_PATH`. |
| `leonardo/submit_eval_qwen3_14b_task_2xa100.sbatch` | Evaluate one 14B benchmark per Leonardo job; two GPUs generate eight samples each and merge to 16. |
| `leonardo/submit_eval_qwen3_1p7b_opd_1xa100.sbatch` | Evaluate a merged 1.7B OPD checkpoint on Leonardo. |
| `leonardo/submit_merge_fsdp_actor.sbatch` | Validate and merge a verl FSDP actor on Leonardo; select it with `RUN_DIR`, `EXPECTED_STEP`, and `MERGED_MODEL_NAME`. |
| `leonardo/submit_sft_teacher_rollout_qwen3_4b_4xa100.sbatch` | Auxiliary/legacy resumable Qwen3-4B rollout for the Leonardo SFT pipeline. |
| `leonardo/submit_sft_qwen3_1p7b_2node_4xa100.sbatch` | Auxiliary/legacy two-node 1.7B cold-start SFT launcher for Leonardo. |
| `leonardo/submit_grpo_qwen3_14b_2node_4xa100.sbatch` / `leonardo/grpo_pilot_qwen3_14b_2node_4xa100.sh` / `leonardo/grpo_qwen3_14b_base.sh` | Leonardo two-node Qwen3-14B GRPO submission wrapper and its training settings/runtime. |
| `leonardo/submit_multinode_2x1a100_smoke.sbatch` | Leonardo Slurm placement and GPU-visibility smoke test; it does not train. |
| `leonardo/opd-cu118.constraints.txt` | Version constraints for the Leonardo CUDA 11.8 OPD environment. |

## Leonardo Boundaries

- Treat the `#SBATCH` account, partition, GPU request, memory, output paths, project paths, Conda paths, and cache roots in these launchers as Leonardo-specific.
- Treat `on_policy_distillation.sh` and `grpo.sh` as legacy references, not current Leonardo submission scripts; their `TEST1` Slurm directives and old experiment defaults must be replaced before any attempted reuse.
- Inspect the selected launcher and current Leonardo allocation before reusing its defaults; several legacy scripts retain old accounts, paths, model choices, or experiment names.
- Keep the Curriculum OPD scientific stages aligned with `AGENTS.md`; resource adaptations must not silently change the experiment definition.
- Run `bash -n` after changing a shell or Slurm script. Use `DRY_RUN=1` when the launcher provides it, and grade only completed evaluation outputs with the matching tokenizer.
