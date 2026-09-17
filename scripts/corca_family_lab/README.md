# CORCA Family Lab

This directory is an isolated staging area for CORCA-family experiments.

Scope:
- add new scripts only;
- call existing CORCA simulators as black boxes;
- keep outputs under `results/corca_family_lab/`;
- avoid changing the main data-generation and optimization flows until a lab script is validated.

Current entry points:
- `parse_corca_outputs.py`: parse `SimuOutBurnup.out` and `SimuOutXenon.out` into compact JSON summaries;
- `build_corca_family_protocol.py`: generate small protocol manifests for `corca_state_family_v1` and `corca_transient_family_v1`.
- `objectives.py`: score state, burnup, and xenon outputs with lab-only objective wrappers;
- `state_family_cases.py`: define grouped CORCA state regimes and rod-pattern candidate sets;
- `transient_family_cases.py`: define burnup and xenon schedule candidates for transient families;
- `run_corca_family_pilot.py`: run small isolated pilot cases and write summaries to `results/corca_family_lab/`.
- `run_corca_family_sweep.py`: run family-level sweeps from the lab protocol and aggregate per-problem summaries.

Guardrails:
- do not modify `scripts/generate_classification_data.py` from this lab;
- do not modify `src/optimizer.py` from this lab;
- do not overwrite files under `templates/corcasim/`;
- keep new templates under `templates/corcasim_lab/`.
