# Complexity Study Status

Read first:
- `instruction/AICO_复杂度研究计划_2026-09-14.md`
- `instruction/AICO_Intellig_相关工作调研_2026-09-13.md`
- `instruction/AICO_v4核验_2026-09-15.md`
- `instruction/AICO_大规模扩跑补强_2026-09-15.md`
- `instruction/AICO_事件级审计与高维v4补强_2026-09-15.md`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_k3.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_k5.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_budget360.json`
- `results/complexity_study/shared_subspace_synthetic_xlarge_v2.json`
- `results/complexity_study/shared_subspace_mismatch_xlarge_v2_d128.json`
- `results/complexity_study/shared_subspace_strict_v4_large_v1.json`
- `results/complexity_study/archive_event_level_audit.json`
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
- archive event-level trace audit
- synthetic baseline
- synthetic large scaling sweep
- mismatch sweep v1
- mismatch sweep v2
- strict v3 pilot sweep
- strict v4 verified pilot sweep
- strict v4 high-dimensional pilot
- strict v4 large stress sweep
- strict v4 xlarge sweep (`k=3`)
- strict v4 xlarge sweep (`k=5`)
- strict v4 xlarge sweep (`budget=360`)
- synthetic xlarge scaling sweep through `d=512`
- mismatch v2 xlarge sweep at `d=128`

Scale:
- scaling large sweep: about `11,700` runs
- mismatch v1: about `36,000` runs
- mismatch v2: about `36,000` runs
- strict v3 pilot: `240` task runs per method across `d in {32,64}`
- strict v4 verified pilot: `240` task runs per method across `d in {32,64}`, plus validation tuning tasks
- strict v4 high-dimensional pilot: `192` task runs per method across `d in {128,256}`, plus validation tuning tasks
- strict v4 large stress sweep: `1,920` task runs per method across `d in {32,64,128,256}`, plus validation tuning tasks
- strict v4 xlarge (`k=3`): `3,600` task runs per method across `d in {32,64,128,256}`, plus validation tuning tasks
- strict v4 xlarge (`k=5`): `2,160` task runs per method across `d in {64,128,256}`, plus validation tuning tasks
- strict v4 xlarge (`budget=360`): `2,160` task runs per method across `d in {64,128,256}`, plus validation tuning tasks
- synthetic xlarge scaling sweep: `7,200` task runs across `d in {16,32,64,128,256,512}`
- mismatch v2 xlarge: `14,400` task runs per method over `120` misspecification conditions at `d=128`
- archive event-level trace audit: `62,558` runs and about `4.11M` iteration events across five candidate families

Conclusions:
- shared subspace strongly beats scratch in the controlled synthetic setting
- history cost vs downstream gain already shows a frontier
- mismatch v1 shows a clear residual-strength trend
- mismatch v2 is better for summary, but angle effects are still not clean
- v3 is a better pilot protocol than v1/v2 for amortization reporting, but it is still not the final fair-comparison protocol
- v4 verifies a common first evaluation point for non-warm-start methods and preserves the structure-transfer advantage under tuned random-direction baselines
- the larger v4 sweep preserves the same pattern through `d=256`: estimated subspace stays near `0.0067` regret while the tested no-history baselines deteriorate sharply
- the xlarge `k=3` v4 sweep keeps the same pattern at larger sample size: at `d=256`, estimated subspace remains near `0.0067` regret while the three tested baselines stay between `0.3018` and `0.3883`
- the xlarge `k=5` v4 sweep shows that the gain is not confined to the `k=3` setting: at `d=256`, estimated subspace stays near `0.0080` while all tested baselines remain above `0.51`
- doubling the target-task budget to `360` improves the no-history baselines somewhat, but it does not remove the gap at `d=128` or `d=256`; the structure-aware method still reaches success@`0.02` = `1.0`
- the synthetic xlarge sweep extends the scaling evidence through `d=512`, where scratch regret reaches `2.0951` while estimated-subspace regret remains `0.0162`
- the xlarge mismatch v2 sweep shows gradual degradation rather than immediate collapse under misspecification: mean success@`0.02` falls from `0.8782` at ambient curvature `0.005` to `0.7049` at `0.1`
- event-level audit now shows that `advection_v1`, `blackscholes_v1`, `heat_time_v1`, and `sns_flow_family` retain full iteration traces with complete `x_json`, contiguous `eval_index`, and exact `run_summary` count alignment
- `sns_v3` is close to usable at event level, but one source experiment still misses `114` runs worth of `iteration_event` records

Important limits:
- current evidence supports a specific synthetic setup, not a general query-complexity claim
- the high-dimensional scratch baseline is a coordinate-difference method that is weak under a budget of 180
- current amortized-threshold summaries average only over successful hits and must not be read as full-task payback
- in `v3`, subspace methods still evaluate the first point after projecting `x0` into the chosen subspace
- in `v3`, the warm-start baseline still uses an idealized history optimum rather than the same observation interface as structure estimation
- `v4` fixes the key `v3` protocol issues and remains strong in higher dimensions, but it still uses a controlled synthetic family and a limited set of no-history baselines
- the xlarge sweeps reduce small-sample uncertainty, but they still stay within one shared-subspace synthetic family rather than testing broad out-of-distribution generalization
- real archive evidence is now event-level for the five follow-up families, but those families remain low-dimensional (`1` to `4` observed coordinates) and therefore cannot carry a `d=64` or `d=256` real-task claim
- `sns_v3` has a localized archive gap in `results/sns_v3_full_240_m30_t90_r3_more1`, so it should not yet be treated as a fully clean event-level benchmark

Selected numbers:
- `d=64`: scratch regret `1.4003`, estimated regret `0.0218`
- `d=256`: scratch regret `1.2986`, estimated regret `0.0104`
- mismatch v1 success@`0.005`: `0.9358 -> 0.8109 -> 0.3218 -> 0.1200` as residual weight goes `0.00 -> 0.02 -> 0.05 -> 0.10`
- mismatch v2 success@`0.005`: `0.9297 -> 0.9196 -> 0.8491 -> 0.7057` as ambient curvature goes `0.005 -> 0.01 -> 0.02 -> 0.05`
- strict v3 (`d=64`): coordinate scratch regret `0.1498`, random-direction scratch regret `0.2221`, history-init random-direction regret `0.2156`, estimated-subspace regret `0.0020`
- strict v4 (`d=64`): coordinate scratch regret `0.1598`, tuned random-direction scratch regret `0.1439`, history-probe-init random-direction regret `0.0780`, estimated-subspace regret `0.0065`
- strict v4 large (`d=128`): coordinate scratch regret `0.1485`, tuned random-direction scratch regret `0.2303`, history-probe-init random-direction regret `0.1430`, estimated-subspace regret `0.0068`
- strict v4 large (`d=256`): coordinate scratch regret `0.3516`, tuned random-direction scratch regret `0.3929`, history-probe-init random-direction regret `0.2981`, estimated-subspace regret `0.0067`
- strict v4 large success@`0.02` at `d=256`: coordinate scratch `0.0083`, tuned random-direction scratch `0.0063`, history-probe-init random-direction `0.0063`, estimated-subspace `1.0000`
- strict v4 xlarge (`k=3`, `d=256`): coordinate scratch regret `0.3569`, tuned random-direction scratch regret `0.3883`, history-probe-init random-direction regret `0.3018`, estimated-subspace regret `0.0067`
- strict v4 xlarge (`k=3`, `d=256`) success@`0.02`: coordinate scratch `0.0056`, tuned random-direction scratch `0.0033`, history-probe-init random-direction `0.0067`, estimated-subspace `0.9956`
- strict v4 xlarge (`k=5`, `d=256`): coordinate scratch regret `0.6033`, tuned random-direction scratch regret `0.6711`, history-probe-init random-direction regret `0.5150`, estimated-subspace regret `0.0080`
- strict v4 xlarge (`budget=360`, `d=256`): coordinate scratch regret `0.1595`, tuned random-direction scratch regret `0.2913`, history-probe-init random-direction regret `0.1519`, estimated-subspace regret `0.0064`
- synthetic xlarge (`d=512`): scratch regret `2.0951`, oracle regret `0.0154`, estimated regret `0.0162`
- mismatch v2 xlarge mean success@`0.02`: `0.8782 -> 0.8698 -> 0.8581 -> 0.7992 -> 0.7049` as ambient curvature goes `0.005 -> 0.01 -> 0.02 -> 0.05 -> 0.1`
- event-level audit: `sns_flow_family` covers `20,512` runs with exact event-count alignment; `advection_v1`, `blackscholes_v1`, and `heat_time_v1` each cover `3,600` runs with full trace integrity; `sns_v3` covers `31,132 / 31,246` runs

Read with v3:
- `instruction/AICO_v3核验_2026-09-15.md`
- `results/complexity_study/shared_subspace_strict_v3.json`

Read with v4:
- `instruction/AICO_v4核验_2026-09-15.md`
- `results/complexity_study/shared_subspace_strict_v4.json`
- `results/complexity_study/shared_subspace_strict_v4_highdim_pilot.json`
- `results/complexity_study/shared_subspace_strict_v4_large_v1.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_k3.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_k5.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_budget360.json`

Manual follow-up families:
- `sns_flow_family`
- `advection_v1`
- `blackscholes_v1`
- `heat_time_v1`
- `sns_v3`

Event-level ready families:
- `sns_flow_family`
- `advection_v1`
- `blackscholes_v1`
- `heat_time_v1`

Event-level follow-up gap:
- `sns_v3`: `114` missing `iteration_event` runs in `results/sns_v3_full_240_m30_t90_r3_more1`
