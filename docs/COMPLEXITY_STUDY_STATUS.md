# Complexity Study Status

Read first:
- `instruction/AICO_复杂度研究计划_2026-09-14.md`
- `instruction/AICO_Intellig_相关工作调研_2026-09-13.md`
- `instruction/AICO_v4核验_2026-09-15.md`
- `instruction/AICO_大规模扩跑补强_2026-09-15.md`
- `instruction/AICO_v4修正重跑_2026-09-16.md`
- `instruction/AICO_事件级审计与高维v4补强_2026-09-15.md`
- `results/complexity_study/shared_subspace_strict_v4_repaired_2026-09-16.json`
- `results/complexity_study/shared_subspace_strict_v4_large_v1_repaired_2026-09-16.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_k3_repaired_2026-09-16.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_k5_repaired_2026-09-16.json`
- `results/complexity_study/shared_subspace_strict_v4_xlarge_budget360_repaired_2026-09-16.json`
- `results/complexity_study/shared_subspace_history_noise_d256_2026-09-16.json`
- `results/complexity_study/shared_subspace_platform_ablation_2026-09-16.json`
- `results/complexity_study/shared_subspace_mismatch_noise_d256_2026-09-16.json`
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
- strict v4 repaired rerun with expanded random-direction step search
- strict v4 large repaired rerun
- strict v4 xlarge repaired reruns (`k=3`, `k=5`, `budget=360`)
- history-noise sweep at `d=256`
- platform ablation sweep
- mismatch-plus-noise sweep at `d=256`
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
- strict v4 repaired rerun: `600` task runs per method across `d in {32,64}`, plus validation tuning tasks
- strict v4 large repaired rerun: `1,920` task runs per method across `d in {32,64,128,256}`, plus validation tuning tasks
- history-noise sweep at `d=256`: `12` history/noise settings, `480` downstream target tasks per setting
- platform ablation: `54` settings over `d in {64,256}`, each with `720` downstream target tasks per method variant
- mismatch-plus-noise sweep at `d=256`: `64` settings, each with `720` downstream target tasks per method
- synthetic xlarge scaling sweep: `7,200` task runs across `d in {16,32,64,128,256,512}`
- mismatch v2 xlarge: `14,400` task runs per method over `120` misspecification conditions at `d=128`
- archive event-level trace audit: `62,558` runs and about `4.11M` iteration events across five candidate families

Latest repaired update (`2026-09-16`):
- the strict `v4` scripts now use an expanded random-direction step grid (`0.05, 0.1, 0.2, 0.4, 0.8, 1.6`) and correct the first-hit accounting bug in per-query best traces
- after the repair, the tuned random-direction baselines improve substantially, but the structure-transfer advantage remains clear at both pilot and xlarge scale
- the repaired `k=3` xlarge sweep now gives at `d=256`: scratch-random regret `0.2151`, history-init-random regret `0.1810`, estimated-subspace regret `0.0067`
- the repaired `k=5` xlarge sweep now gives at `d=256`: scratch-random regret `0.4336`, history-init-random regret `0.4001`, estimated-subspace regret `0.0080`
- the repaired `budget=360` xlarge sweep now gives at `d=256`: scratch-random regret `0.1777`, history-init-random regret `0.1320`, estimated-subspace regret `0.0064`
- the new history-noise sweep at `d=256` shows graceful degradation rather than collapse: with `8` history tasks and `2` probes per task, success@`0.02` stays at `0.9792` even at noise std `0.1`
- the new platform ablation shows that the old `~0.006` floor is not purely a structure-estimation limit; anchor and search-radius choices create much of the observed plateau
- the new mismatch-plus-noise sweep confirms that moderate history noise and moderate misspecification do not erase the transfer gain, but angle effects are not yet monotone enough for a clean boundary claim

Conclusions:
- shared subspace strongly beats scratch in the controlled synthetic setting
- history cost vs downstream gain already shows a frontier
- mismatch v1 shows a clear residual-strength trend
- mismatch v2 is better for summary, but angle effects are still not clean
- v3 is a better pilot protocol than v1/v2 for amortization reporting, but it is still not the final fair-comparison protocol
- repaired `v4` still verifies a common first evaluation point for non-warm-start methods and preserves the structure-transfer advantage under substantially stronger tuned random-direction baselines
- the repaired large `v4` sweep preserves the same pattern through `d=256`: estimated subspace stays near `0.0066` regret while the tuned no-history random-direction baselines remain at `0.1111` (`d=128`) and `0.2149` (`d=256`)
- the repaired xlarge `k=3` sweep keeps the same pattern at larger sample size: at `d=256`, estimated subspace remains near `0.0067` regret while the tuned random-direction baselines stay at `0.1810` to `0.2151`
- the repaired xlarge `k=5` sweep shows that the gain is not confined to the `k=3` setting: at `d=256`, estimated subspace stays near `0.0080` while the tuned random-direction baselines remain around `0.40` to `0.43`
- doubling the target-task budget to `360` improves the no-history baselines somewhat, but it does not remove the gap at `d=128` or `d=256`; the structure-aware method still reaches success@`0.02` = `1.0`
- the history-noise sweep at `d=256` shows that moderate additive noise on history gradients enlarges projector error much faster than it harms downstream success, so the current synthetic transfer gain is noise-tolerant in the tested regime
- the platform ablation shows that the old `~0.006` floor partly comes from anchor and radius choices rather than from subspace-estimation error alone
- the synthetic xlarge sweep extends the scaling evidence through `d=512`, where scratch regret reaches `2.0951` while estimated-subspace regret remains `0.0162`
- the xlarge mismatch v2 sweep shows gradual degradation rather than immediate collapse under misspecification: mean success@`0.02` falls from `0.8782` at ambient curvature `0.005` to `0.7049` at `0.1`
- the mismatch-plus-noise sweep at `d=256` shows that the transfer gain survives jointly perturbed history quality and subspace mismatch, but the angle-degradation pattern is not monotone enough yet for a clean theorem-style boundary statement
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
- the repaired random-direction baselines are stronger than before, so pre-`2026-09-16` gap magnitudes should not be cited as the current canonical effect sizes
- the platform ablation shows that the old `~0.006` plateau must not be read as “structure learning has reached the intrinsic optimum”; part of that floor is optimizer-side
- the mismatch-plus-noise sweep is encouraging, but the angle response is not monotone enough yet to support a sharp misspecification law
- real archive evidence is now event-level for the five follow-up families, but those families remain low-dimensional (`1` to `4` observed coordinates) and therefore cannot carry a `d=64` or `d=256` real-task claim
- `sns_v3` has a localized archive gap in `results/sns_v3_full_240_m30_t90_r3_more1`, so it should not yet be treated as a fully clean event-level benchmark

Selected numbers:
- `d=64`: scratch regret `1.4003`, estimated regret `0.0218`
- `d=256`: scratch regret `1.2986`, estimated regret `0.0104`
- mismatch v1 success@`0.005`: `0.9358 -> 0.8109 -> 0.3218 -> 0.1200` as residual weight goes `0.00 -> 0.02 -> 0.05 -> 0.10`
- mismatch v2 success@`0.005`: `0.9297 -> 0.9196 -> 0.8491 -> 0.7057` as ambient curvature goes `0.005 -> 0.01 -> 0.02 -> 0.05`
- strict v3 (`d=64`): coordinate scratch regret `0.1498`, random-direction scratch regret `0.2221`, history-init random-direction regret `0.2156`, estimated-subspace regret `0.0020`
- strict v4 repaired (`d=64`): coordinate scratch regret `0.1598`, tuned random-direction scratch regret `0.0580`, history-probe-init random-direction regret `0.0560`, estimated-subspace regret `0.0065`
- strict v4 large repaired (`d=128`): coordinate scratch regret `0.1576`, tuned random-direction scratch regret `0.1111`, history-probe-init random-direction regret `0.0990`, estimated-subspace regret `0.0067`
- strict v4 large repaired (`d=256`): coordinate scratch regret `0.3498`, tuned random-direction scratch regret `0.2149`, history-probe-init random-direction regret `0.1869`, estimated-subspace regret `0.0066`
- strict v4 large repaired success@`0.02` at `d=256`: coordinate scratch `0.0083`, tuned random-direction scratch `0.0437`, history-probe-init random-direction `0.1021`, estimated-subspace `1.0000`
- strict v4 xlarge repaired (`k=3`, `d=256`): coordinate scratch regret `0.3569`, tuned random-direction scratch regret `0.2151`, history-probe-init random-direction regret `0.1810`, estimated-subspace regret `0.0067`
- strict v4 xlarge repaired (`k=3`, `d=256`) success@`0.02`: coordinate scratch `0.0056`, tuned random-direction scratch `0.0533`, history-probe-init random-direction `0.1011`, estimated-subspace `0.9956`
- strict v4 xlarge repaired (`k=5`, `d=256`): coordinate scratch regret `0.6033`, tuned random-direction scratch regret `0.4336`, history-probe-init random-direction regret `0.4001`, estimated-subspace regret `0.0080`
- strict v4 xlarge repaired (`budget=360`, `d=256`): coordinate scratch regret `0.1595`, tuned random-direction scratch regret `0.1777`, history-probe-init random-direction regret `0.1320`, estimated-subspace regret `0.0064`
- history-noise (`d=256`, `8` history tasks, `2` probes, noise std `0.0`): projector error `0.0074`, estimated-subspace regret `0.0079`, success@`0.02` `0.9688`
- history-noise (`d=256`, `8` history tasks, `2` probes, noise std `0.1`): projector error `0.1709`, estimated-subspace regret `0.0198`, success@`0.02` `0.9792`
- platform ablation (`d=64`, budget `180`, radius `0.5`, `fd_eps=1e-3`): estimated-anchor `0.0023`, estimated-free `0.0001`, oracle-anchor `0.0023`, oracle-free `0.0004`
- mismatch-plus-noise (`d=256`, angle `45`, ambient `0.01`, noise std `0.1`): projector error `0.7257`, estimated-subspace regret `0.0127`, success@`0.02` `0.8778`
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
