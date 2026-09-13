# AICO-Intellig Dataset Inventory

Generated at: `2026-09-05T07:31:03+00:00`

## Scope

This inventory is a non-destructive organization layer. It does not move or rewrite raw results. Instead, it records canonical file locations, aggregate record counts, and directory roles for the current repository state.

## Canonical Interfaces

- `problem_instance.jsonl`: canonical instance-level structural metadata.
- `run_summary.jsonl`: canonical run-level summary table.
- `iteration_event.jsonl`: canonical fine-grained trajectory events.
- `run_checkpoint.jsonl`: materialized checkpoint cache derived from iteration events.
- `dataset.jsonl`: training-oriented instance view with labels and candidate aggregates.
- `runs.jsonl`: lightweight run view used by ranker-style pipelines.

## Current Scale

- Discovered data-bearing experiment directories: `51`
- Aggregate JSONL record counts: `dataset.jsonl` = `7872`, `problem_instance.jsonl` = `7342`, `runs.jsonl` = `258552`, `run_summary.jsonl` = `123419`, `iteration_event.jsonl` = `6265280`, `run_checkpoint.jsonl` = `4327644`
- Manifest classes: `derived_view` = `4`, `lightweight_dataset` = `3`, `raw_experiment` = `26`, `sharded_experiment` = `18`

## Results Root Overview

| Entry | Role | Files | Size (MB) |
| --- | --- | --- | --- |
| _bucketed_sns_ranker_v1v3 | experiment_bundle | 13 | 291.73 |
| _bucketed_sns_ranker_v1v3_train | container | 15 | 218.29 |
| _global_split_sns_ranker_v1v3 | container | 2 | 0.01 |
| _merged_sns_ranker_v1v3 | experiment_bundle | 3 | 13.6 |
| _merged_sns_ranker_v1v3_richfeat | experiment_bundle | 5 | 408.19 |
| _merged_sns_ranker_v1v3_richfeat_train | experiment_bundle | 5 | 218.29 |
| _merged_sns_ranker_v1v3_staticfeat | experiment_bundle | 3 | 14.07 |
| algo_landscape_20260711_101558 | container | 0 | 0.0 |
| algo_landscape_20260711_101614 | report_bundle | 6 | 0.31 |
| algo_landscape_latest | report_bundle | 6 | 0.3 |
| corca_test_v1 | experiment_bundle | 23 | 0.15 |
| curve_policy_eval20_20260710_142137 | model_bundle | 1 | 0.0 |
| curve_policy_eval20_20260710_142353 | model_bundle | 3 | 6.54 |
| curve_policy_eval20_latest | model_bundle | 3 | 6.56 |
| ex_advection2d_family_v1_180_m100_t3600_seed0_20260708_203300 | experiment_bundle | 11 | 0.55 |
| ex_advection2d_family_v1_200_m100_t3600_seed0_20260709_103848 | experiment_bundle | 11 | 0.61 |
| ex_advection2d_family_v1_instances200_config.json | config_or_summary_file | 1 | 0.0 |
| ex_advection2d_v1_full_200_m100_t3600_r1_20260709_231148 | experiment_bundle | 10848 | 552.56 |
| ex_blackscholes2d_family_v1_180_m100_t3600_seed0_20260708_203300 | experiment_bundle | 11 | 0.57 |
| ex_blackscholes2d_family_v1_200_m100_t3600_seed0_20260709_103842 | experiment_bundle | 11 | 0.64 |
| ex_blackscholes2d_family_v1_instances200_config.json | config_or_summary_file | 1 | 0.0 |
| ex_blackscholes2d_v1_full_200_m100_t3600_r1_20260709_104116 | experiment_bundle | 10848 | 567.03 |
| ex_heat_time_family_v1_180_m100_t3600_seed0_20260708_203300 | experiment_bundle | 11 | 0.55 |
| ex_heat_time_family_v1_200_m100_t3600_seed0_20260709_103835 | experiment_bundle | 11 | 0.6 |
| ex_heat_time_family_v1_instances200_config.json | config_or_summary_file | 1 | 0.0 |
| ex_heat_time_v1_full_200_m100_t3600_r1_ | experiment_bundle | 10848 | 538.16 |
| expansion_20260822 | container | 93210 | 3123.28 |
| expansion_20260826 | container | 20223 | 459.58 |
| heat_family_protocol_v1_instances180 | experiment_bundle | 11 | 1.83 |
| heat_v1_full_180_m30_t90_r3 | experiment_bundle | 29208 | 738.9 |
| iaea_family_protocol_v1_instances180 | experiment_bundle | 11 | 0.91 |
| iaea_v1_full_180_m20_t90_r2 | experiment_bundle | 29196 | 511.19 |
| iaea_v1_full_180_m30_t90_r3_20260708_101221 | experiment_bundle | 16 | 0.09 |
| iaea_v1_full_180_m30_t90_r3_20260708_102705 | experiment_bundle | 29208 | 573.91 |
| massive_pde_database_v1 | experiment_bundle | 2707 | 86.6 |
| massive_pde_extensions_v1 | experiment_bundle | 2111 | 33.41 |
| massive_sweep.log | log_file | 1 | 5.1 |
| massive_sweep2.log | log_file | 1 | 5.15 |
| massive_sweep_extensions.log | log_file | 1 | 2.51 |
| massive_sweep_fixed.log | log_file | 1 | 0.04 |
| model_algo_ranker_bucket_constrained_rerank_pool9_warm10.joblib | model_file | 1 | 15.25 |
| model_algo_ranker_bucket_constrained_rerank_pool9_warm10_catboost_noleak.joblib | model_file | 1 | 2.4 |
| model_algo_ranker_bucket_constrained_rerank_pool9_warm10_noleak.joblib | model_file | 1 | 13.5 |
| model_algo_ranker_bucket_constrained_static_pool9.joblib | model_file | 1 | 15.25 |
| model_algo_ranker_bucket_constrained_static_pool9_catboost_noleak.joblib | model_file | 1 | 2.4 |
| model_algo_ranker_bucket_constrained_static_pool9_noleak.joblib | model_file | 1 | 13.5 |
| model_algo_ranker_bucket_design_minmax_rerank_pool9_warm10.joblib | model_file | 1 | 9.08 |
| model_algo_ranker_bucket_design_minmax_rerank_pool9_warm10_catboost_noleak.joblib | model_file | 1 | 2.41 |
| model_algo_ranker_bucket_design_minmax_rerank_pool9_warm10_noleak.joblib | model_file | 1 | 8.83 |
| model_algo_ranker_bucket_design_minmax_static_pool9.joblib | model_file | 1 | 9.08 |
| model_algo_ranker_bucket_design_minmax_static_pool9_catboost_noleak.joblib | model_file | 1 | 2.41 |
| model_algo_ranker_bucket_design_minmax_static_pool9_noleak.joblib | model_file | 1 | 8.83 |
| model_algo_ranker_bucket_target_matching_rerank_pool9_warm10.joblib | model_file | 1 | 62.26 |
| model_algo_ranker_bucket_target_matching_rerank_pool9_warm10_catboost_noleak.joblib | model_file | 1 | 2.43 |
| model_algo_ranker_bucket_target_matching_rerank_pool9_warm10_noleak.joblib | model_file | 1 | 62.57 |
| model_algo_ranker_bucket_target_matching_static_pool9.joblib | model_file | 1 | 62.26 |
| model_algo_ranker_bucket_target_matching_static_pool9_catboost_noleak.joblib | model_file | 1 | 2.43 |
| model_algo_ranker_bucket_target_matching_static_pool9_noleak.joblib | model_file | 1 | 62.57 |
| model_algo_ranker_merged_warm10.joblib | model_file | 1 | 127.29 |
| model_algo_ranker_pool9.joblib | model_file | 1 | 17.07 |
| model_algo_ranker_pool9_warm10.joblib | model_file | 1 | 20.3 |
| model_algo_ranker_pool9_warm30.joblib | model_file | 1 | 19.25 |
| model_algo_ranker_sns_v1v3_richfeat_pool9_warm10.joblib | model_file | 1 | 86.25 |
| model_algo_ranker_sns_v1v3_richfeat_pool9_warm10_noleak.joblib | model_file | 1 | 85.69 |
| model_algo_ranker_sns_v1v3_richfeat_warm10.joblib | model_file | 1 | 234.88 |
| model_algo_ranker_sns_v1v3_static_pool9.joblib | model_file | 1 | 72.47 |
| model_algo_ranker_sns_v1v3_static_pool9_noleak.joblib | model_file | 1 | 85.69 |
| model_algo_ranker_sns_v1v3_warm10.joblib | model_file | 1 | 190.31 |
| model_algo_selector_all.joblib | model_file | 1 | 0.01 |
| model_algo_selector_pool9.joblib | model_file | 1 | 0.01 |
| pairwise_curve_policy_eval10_20260715_101627 | model_bundle | 2 | 0.53 |
| pairwise_curve_policy_eval15_20260715_101714 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval15_20260715_102311 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval16_20260715_102357 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval17_20260715_102443 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval18_20260715_102529 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval19_20260715_102615 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval20_20260714_174639 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval20_20260714_174746 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval20_20260715_101801 | model_bundle | 2 | 0.53 |
| pairwise_curve_policy_eval20_20260715_102702 | model_bundle | 2 | 0.53 |
| pairwise_curve_policy_eval21_20260715_102749 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval22_20260715_102836 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval23_20260715_102924 | model_bundle | 2 | 0.54 |
| pairwise_curve_policy_eval24_20260715_103012 | model_bundle | 2 | 0.53 |
| pairwise_curve_policy_eval25_20260715_101849 | model_bundle | 2 | 0.53 |
| pairwise_curve_policy_eval25_20260715_103059 | model_bundle | 2 | 0.53 |
| pairwise_curve_policy_eval30_20260715_101936 | model_bundle | 2 | 0.53 |
| pairwise_curve_policy_eval5_20260715_101543 | model_bundle | 2 | 0.53 |
| paper_stats_20260722 | report_bundle | 2 | 0.0 |
| rule_switch_policy_latest.json | config_or_summary_file | 1 | 0.0 |
| sns_family_protocol_v1_instances150 | experiment_bundle | 6 | 0.22 |
| sns_family_protocol_v1_instances500 | experiment_bundle | 11 | 1.46 |
| sns_family_protocol_v1_probe10 | experiment_bundle | 246 | 0.63 |
| sns_family_protocol_v1_smoke | experiment_bundle | 6 | 0.09 |
| sns_family_protocol_v3_instances160 | experiment_bundle | 11 | 0.75 |
| sns_family_protocol_v3_instances240_more1 | experiment_bundle | 11 | 1.3 |
| sns_family_protocol_v3_instances240_more1_config.json | config_or_summary_file | 1 | 0.0 |
| sns_v1_full18_i150_r2 | experiment_bundle | 4973 | 69.07 |
| sns_v1_full_150_m30_t60_r3 | experiment_bundle | 24312 | 429.09 |
| sns_v1_full_150_m40_t90_r4_20260429_234621 | experiment_bundle | 32448 | 694.96 |
| sns_v3_full_160_m30_t90_r3 | experiment_bundle | 25968 | 741.11 |
| sns_v3_full_240_m30_t90_r3_more1 | experiment_bundle | 19599 | 507.06 |
| static_ranker_baseline_latest | model_bundle | 13 | 931.75 |
| submission_snapshot_latest | report_bundle | 8 | 0.01 |
| sweep_logs | log_bundle | 8 | 356.91 |
| test_corca_gen | experiment_bundle | 65 | 0.71 |
| test_corca_gen2 | experiment_bundle | 65 | 0.74 |
| test_corca_gen3 | experiment_bundle | 65 | 0.73 |
| theory_validation_latest | container | 7 | 0.05 |
| thmf_v1_full_180_m30_t90_r3_20260708_101210 | experiment_bundle | 16 | 0.19 |
| thmf_v1_full_180_m30_t90_r3_20260708_102705 | experiment_bundle | 29208 | 743.87 |
| training_more3_model | model_bundle | 1 | 0.0 |
| training_train | model_bundle | 1 | 0.0 |

## Canonical Experiment Index

| Path | Class | Shards | dataset | problem | runs | run_summary | checkpoints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results/_merged_sns_ranker_v1v3 | derived_view | 0 | 310 | 0 | 19440 | 0 | 0 |
| results/_merged_sns_ranker_v1v3_richfeat | derived_view | 0 | 310 | 310 | 19440 | 0 | 523260 |
| results/_merged_sns_ranker_v1v3_richfeat_train | derived_view | 0 | 232 | 232 | 14544 | 0 | 276336 |
| results/_merged_sns_ranker_v1v3_staticfeat | derived_view | 0 | 310 | 310 | 19440 | 0 | 0 |
| results/sns_family_protocol_v1_instances150 | lightweight_dataset | 0 | 150 | 0 | 0 | 0 | 0 |
| results/sns_family_protocol_v1_probe10 | lightweight_dataset | 0 | 10 | 0 | 80 | 0 | 0 |
| results/sns_family_protocol_v1_smoke | lightweight_dataset | 0 | 50 | 0 | 0 | 0 | 0 |
| results/_bucketed_sns_ranker_v1v3/constrained | raw_experiment | 0 | 50 | 50 | 2700 | 0 | 51300 |
| results/_bucketed_sns_ranker_v1v3/design_minmax | raw_experiment | 0 | 39 | 39 | 2106 | 0 | 40014 |
| results/_bucketed_sns_ranker_v1v3/target_matching | raw_experiment | 0 | 221 | 221 | 14634 | 0 | 278046 |
| results/_bucketed_sns_ranker_v1v3_train/constrained | raw_experiment | 0 | 34 | 34 | 1836 | 0 | 34884 |
| results/_bucketed_sns_ranker_v1v3_train/design_minmax | raw_experiment | 0 | 30 | 30 | 1620 | 0 | 30780 |
| results/_bucketed_sns_ranker_v1v3_train/target_matching | raw_experiment | 0 | 168 | 168 | 11088 | 0 | 210672 |
| results/corca_test_v1 | raw_experiment | 0 | 2 | 2 | 4 | 4 | 76 |
| results/ex_advection2d_family_v1_180_m100_t3600_seed0_20260708_203300 | raw_experiment | 0 | 180 | 180 | 0 | 0 | 0 |
| results/ex_advection2d_family_v1_200_m100_t3600_seed0_20260709_103848 | raw_experiment | 0 | 200 | 200 | 0 | 0 | 0 |
| results/ex_blackscholes2d_family_v1_180_m100_t3600_seed0_20260708_203300 | raw_experiment | 0 | 180 | 180 | 0 | 0 | 0 |
| results/ex_blackscholes2d_family_v1_200_m100_t3600_seed0_20260709_103842 | raw_experiment | 0 | 200 | 200 | 0 | 0 | 0 |
| results/ex_heat_time_family_v1_180_m100_t3600_seed0_20260708_203300 | raw_experiment | 0 | 180 | 180 | 0 | 0 | 0 |
| results/ex_heat_time_family_v1_200_m100_t3600_seed0_20260709_103835 | raw_experiment | 0 | 200 | 200 | 0 | 0 | 0 |
| results/heat_family_protocol_v1_instances180 | raw_experiment | 0 | 180 | 180 | 0 | 0 | 0 |
| results/iaea_family_protocol_v1_instances180 | raw_experiment | 0 | 180 | 180 | 0 | 0 | 0 |
| results/massive_pde_database_v1 | raw_experiment | 0 | 16 | 3 | 878 | 818 | 17063 |
| results/massive_pde_extensions_v1 | raw_experiment | 0 | 12 | 11 | 682 | 636 | 13292 |
| results/sns_family_protocol_v1_instances500 | raw_experiment | 0 | 500 | 500 | 0 | 0 | 0 |
| results/sns_family_protocol_v3_instances160 | raw_experiment | 0 | 160 | 160 | 0 | 0 | 0 |
| results/sns_family_protocol_v3_instances240_more1 | raw_experiment | 0 | 240 | 240 | 0 | 0 | 0 |
| results/sns_v1_full18_i150_r2 | raw_experiment | 0 | 45 | 42 | 1652 | 1612 | 31416 |
| results/sns_v1_full_150_m30_t60_r3 | raw_experiment | 0 | 150 | 150 | 8100 | 8100 | 153900 |
| results/static_ranker_baseline_latest | raw_experiment | 0 | 782 | 782 | 28063 | 0 | 534009 |
| results/test_corca_gen | raw_experiment | 0 | 1 | 1 | 18 | 18 | 342 |
| results/test_corca_gen2 | raw_experiment | 0 | 1 | 1 | 18 | 18 | 342 |
| results/test_corca_gen3 | raw_experiment | 0 | 1 | 1 | 18 | 18 | 342 |
| results/ex_advection2d_v1_full_200_m100_t3600_r1_20260709_231148 | sharded_experiment | 4 | 200 | 200 | 3600 | 3600 | 68400 |
| results/ex_blackscholes2d_v1_full_200_m100_t3600_r1_20260709_104116 | sharded_experiment | 4 | 200 | 200 | 3600 | 3600 | 68400 |
| results/ex_heat_time_v1_full_200_m100_t3600_r1_ | sharded_experiment | 4 | 200 | 200 | 3600 | 3600 | 68400 |
| results/expansion_20260822/cheap_xfamilies_pool9_r3 | sharded_experiment | 3 | 303 | 303 | 8181 | 8181 | 155439 |
| results/expansion_20260822/iaea_v1_pool9_r3 | sharded_experiment | 2 | 22 | 22 | 594 | 594 | 11286 |
| results/expansion_20260822/sns_v3_full_240_m40_t120_r4 | sharded_experiment | 4 | 240 | 240 | 17280 | 17280 | 328320 |
| results/expansion_20260822/thmf_v1_pool9_r3 | sharded_experiment | 2 | 184 | 184 | 4968 | 4968 | 94392 |
| results/expansion_20260826/ex_heat_time_v1_pool9_r3 | sharded_experiment | 2 | 44 | 44 | 1188 | 1188 | 22572 |
| results/expansion_20260826/ex_pde_v1_pool9_r3 | sharded_experiment | 2 | 205 | 205 | 5535 | 5535 | 105165 |
| results/heat_v1_full_180_m30_t90_r3 | sharded_experiment | 4 | 180 | 180 | 9720 | 9720 | 184680 |
| results/iaea_v1_full_180_m20_t90_r2 | sharded_experiment | 3 | 180 | 180 | 9720 | 9720 | 184680 |
| results/iaea_v1_full_180_m30_t90_r3_20260708_102705 | sharded_experiment | 4 | 67 | 63 | 9720 | 9720 | 184680 |
| results/sns_v1_full_150_m40_t90_r4_20260429_234621 | sharded_experiment | 4 | 150 | 150 | 10800 | 10800 | 205200 |
| results/sns_v3_full_160_m30_t90_r3 | sharded_experiment | 4 | 160 | 160 | 8640 | 8640 | 164160 |
| results/sns_v3_full_240_m30_t90_r3_more1 | sharded_experiment | 4 | 98 | 105 | 5325 | 5329 | 101116 |
| results/thmf_v1_full_180_m30_t90_r3_20260708_102705 | sharded_experiment | 4 | 115 | 119 | 9720 | 9720 | 184680 |

## Industrial Runtime Assets

The `folderA/preciseFZ` tree is not part of the same JSONL dataset interface. It contains runtime assets, databanks, workspaces, and outputs for CORCA/preciseFZ integration.

| Entry | Kind | Files | Size (MB) |
| --- | --- | --- | --- |
| .dep.inc | file | 1 | 0.0 |
| .make.state.Debug | file | 1 | 0.24 |
| apply | dir | 161 | 51.98 |
| databank | dir | 44 | 1322.61 |
| del.py | file | 1 | 0.0 |
| forecast | dir | 196 | 141.2 |
| result | dir | 4 | 16.13 |
| tracking | dir | 112 | 69.0 |

## Recommended Read Order

- For dataset auditing: start from `results/curated_datasets/index.csv`.
- For an individual sweep: open the matching manifest under `results/manifests/`.
- For instance-level features: prefer `problem_instance.jsonl` over `dataset.jsonl`.
- For run-level statistics: prefer `run_summary.jsonl` over `runs.jsonl` whenever available.
- For early-trajectory policy work: treat `iteration_event.jsonl` as source and `run_checkpoint.jsonl` as cache.
