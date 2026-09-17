# CORCA Family Lab Guardrails

## Goal

Extend CORCA-related task families inside `AICO-Intellig` without destabilizing the
existing archive, generation scripts, or optimization entry points.

## Isolation Policy

All exploratory work should start in:
- `scripts/corca_family_lab/`
- `templates/corcasim_lab/`
- `results/corca_family_lab/`

The following files are out of scope for the bootstrap stage unless a later review
explicitly approves integration:
- `scripts/generate_classification_data.py`
- `src/optimizer.py`
- `templates/corcasim/default.json`
- `templates/corcasim/burnup.json`
- `templates/corcasim/xenon.json`

## Bootstrap Deliverables

1. Lightweight parser for `SimuOutBurnup.out` and `SimuOutXenon.out`
2. Protocol manifest for:
   - `corca_state_family_v1`
   - `corca_transient_family_v1`
3. Small pilots written only to `results/corca_family_lab/`

## Expansion Order

1. Parse existing outputs correctly
2. Define grouped parameter families without touching the main generator
3. Run pilots via standalone scripts
4. Revisit framework integration only after pilots show stable value

## Current Rationale

The simulator layer already exposes richer structure than the current curated CORCA
instances capture. The immediate bottleneck is task-family design, not simulator
availability. Therefore, the lab should first add wrappers and manifests rather than
modify the main system.
