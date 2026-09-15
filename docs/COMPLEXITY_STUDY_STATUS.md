# Complexity Study Status

Read first:
- `instruction/AICO_复杂度研究计划_2026-09-14.md`
- `instruction/AICO_Intellig_相关工作调研_2026-09-13.md`
- `results/complexity_study/shared_subspace_synthetic_large_v1.json`
- `results/complexity_study/shared_subspace_mismatch_large_v1.json`
- `results/complexity_study/shared_subspace_mismatch_large_v2.json`
- `results/complexity_study/archive_family_audit.json`

Do not read first:
- `results/curated_datasets/instances.jsonl`
- `results/curated_datasets/run_summaries.jsonl`
- `results/curated_datasets/runs_light.jsonl`
- `results/curated_datasets/checkpoints.jsonl`
- raw `results/` tree

Completed:
- archive family audit
- synthetic baseline
- synthetic large scaling sweep
- mismatch sweep v1
- mismatch sweep v2

Scale:
- scaling large sweep: about `11,700` runs
- mismatch v1: about `36,000` runs
- mismatch v2: about `36,000` runs

Conclusions:
- shared subspace strongly beats scratch in the controlled synthetic setting
- history cost vs downstream gain already shows a frontier
- mismatch v1 shows a clear residual-strength trend
- mismatch v2 is better for summary, but angle effects are still not clean
- real AICO families still need event-level audit before becoming the main benchmark

Selected numbers:
- `d=64`: scratch regret `1.4003`, estimated regret `0.0218`
- `d=256`: scratch regret `1.2986`, estimated regret `0.0104`
- mismatch v1 success@`0.005`: `0.9358 -> 0.8109 -> 0.3218 -> 0.1200` as residual weight goes `0.00 -> 0.02 -> 0.05 -> 0.10`
- mismatch v2 success@`0.005`: `0.9297 -> 0.9196 -> 0.8491 -> 0.7057` as ambient curvature goes `0.005 -> 0.01 -> 0.02 -> 0.05`

Manual follow-up families:
- `sns_flow_family`
- `advection_v1`
- `blackscholes_v1`
- `heat_time_v1`
- `sns_v3`
