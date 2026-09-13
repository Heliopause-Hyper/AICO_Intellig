# Theory Validation Summary

## Scope

This note summarizes the first theory-oriented experiment pack for AICO-Intellig:

- **E1**: utility misalignment under LOFO
- **E2**: one-sided harmful-switch calibration
- **E5**: affine robustness audit

All three experiments were run directly from the current working tree on 2026-08-18.

## E1: Utility Misalignment

Script:

- `scripts/paper_experiments/run_theory_utility_misalignment.py`

Output:

- `results/theory_validation_latest/utility_misalignment.json`

Observed results:

- **Static Top-1**
  - accuracy: `35.97%`
  - mean gain: `-0.359`
  - loss rate: `14.49%`
  - switch rate: `64.30%`
  - CVaR_10: `-5.450`
- **Trajectory Top-1**
  - accuracy: `44.27%`
  - mean gain: `-0.685`
  - loss rate: `22.13%`
  - switch rate: `74.57%`
  - CVaR_10: `-8.758`
- **Pairwise Gain Router**
  - accuracy: `17.65%`
  - mean gain: `+0.029`
  - loss rate: `14.23%`
  - switch rate: `39.79%`
  - CVaR_10: `-0.776`

Interpretation:

- This is the cleanest empirical support so far for the decision-theoretic reframing.
- Higher top-1 accuracy does **not** imply higher deployment utility.
- In fact, the worst utility comes from the highest-accuracy model in this comparison.
- The pairwise router achieves the lowest accuracy but the only positive expected gain.

Theory implication:

- This strongly supports **Proposition 1** and **Corollary 1** in `theory_innovation_blueprint.md`.
- The expensive-routing problem should not be written as absolute top-1 prediction.

## E2: Harmful-Switch Calibration

Script:

- `scripts/paper_experiments/run_theory_harmful_switch_calibration.py`

Output:

- `results/theory_validation_latest/harmful_switch_calibration.json`

Observed results:

- **alpha = 0.05**
  - switch mass: `0.00%`
  - harmful-switch mass: `0.00%`
  - mean gain: `+0.000`
- **alpha = 0.10**
  - switch mass: `0.00%`
  - harmful-switch mass: `0.00%`
  - mean gain: `+0.000`
- **alpha = 0.20**
  - switch mass: `2.16%`
  - harmful-switch mass: `0.91%`
  - harmful-switch rate | switched: `42.00%`
  - mean gain: `-0.007`
  - harmful mass / alpha: `0.045`
- **alpha = 0.30**
  - switch mass: `15.24%`
  - harmful-switch mass: `5.87%`
  - harmful-switch rate | switched: `38.53%`
  - mean gain: `-0.002`
  - harmful mass / alpha: `0.196`

Interpretation:

- The current one-sided conformal gate is extremely conservative for `alpha <= 0.1` under LOFO.
- Once the gate starts switching (`alpha >= 0.2`), the **marginal harmful-switch mass** remains far below `alpha`.
- However, the **conditional harmful-switch rate given switching** remains high.

Theory implication:

- These results are consistent with the correct marginal statement
  - `P(harmful switch and trigger) <= alpha`
- They do **not** support a stronger conditional claim such as
  - `P(harmful switch | trigger) <= alpha`
- This validates the cautious formulation adopted in `theory_innovation_blueprint.md`.

Method implication:

- If conformal gating remains central in the paper, it should be framed as a **safe abstention mechanism**, not as a high-utility switch policy in its current implementation.

## E3: K Trade-Off Decomposition and Sequential Stopping

Script:

- `scripts/paper_experiments/run_theory_k_tradeoff_decomposition.py`

Output:

- `results/theory_validation_latest/k_tradeoff_decomposition.json`

Observed results (fixed `K`, conformal lower-bound gate with `alpha=0.1`):

- **K=5**
  - mean gain: `+0.009`
  - switch rate: `3.16%`
  - CVaR_10: `-0.0017`
  - remaining budget fraction (mean): `68.1%`
- **K=10**
  - mean gain: `+0.020`
  - switch rate: `7.62%`
  - CVaR_10: `-0.0067`
  - remaining budget fraction (mean): `50.3%`
- **K=20**
  - mean gain: `+0.037`
  - switch rate: `9.34%`
  - CVaR_10: `-0.0071`
  - remaining budget fraction (mean): `22.9%`
- **K=30**
  - mean gain: `+0.044`
  - switch rate: `13.84%`
  - CVaR_10: `-0.0215`
  - remaining budget fraction (mean): `18.2%`

Observed results (sequential stopping over `K ∈ {5,10,15,20,30}` with the same `alpha=0.1` gate):

- mean gain: `+0.064`
- switch rate: `25.17%`
- CVaR_10: `-0.0357`
- median `K` used: `30`

Interpretation:

- Larger `K` increases switching frequency and mean gain, but also increases tail downside (CVaR becomes more negative).
- The current sequential stopping rule is not “early” in practice (median `K=30`), and its tail risk is substantially worse than any fixed-`K` conformal policy.

Theory implication:

- This produces the concrete decomposition we need for **Proposition 4**: `K` is simultaneously a diagnostic-information lever and a budget-allocation decision, and the risk/utility frontier shifts materially as `K` changes.
- The sequential policy results also indicate that the *stopping rule itself* is part of the theoretical object, not a minor implementation detail.

## E4: Tail-Risk Objective (CVaR-Optimal Thresholding)

Script:

- `scripts/paper_experiments/run_theory_tail_risk_objective.py`

Output:

- `results/theory_validation_latest/tail_risk_objective.json`

Observed results (`K=20`, LOFO):

- **tau0_mean_opt** (baseline threshold `tau=0`)
  - mean gain: `+0.013`
  - switch rate: `59.0%`
  - CVaR_10: `-1.131`
- **tau_star_mean** (threshold selected to maximize mean gain on calibration split)
  - mean gain: `+0.030`
  - switch rate: `77.5%`
  - CVaR_10: `-1.237`
- **tau_star_cvar** (threshold selected to maximize CVaR_10 on calibration split)
  - mean gain: `+0.024`
  - switch rate: `6.9%`
  - CVaR_10: `-0.0029`
- **conformal_lb** (lower-bound certified switch, `alpha=0.1`)
  - mean gain: `+0.049`
  - switch rate: `15.0%`
  - CVaR_10: `-0.0056`

Interpretation:

- Optimizing the tail objective (CVaR) yields a qualitatively different operating point than optimizing mean gain: switch rate collapses, tail risk becomes near-zero, and conditional gain per switch rises sharply.
- In this dataset, the conformal lower-bound policy achieves both high mean gain and low tail risk, consistent with framing conformal as a risk-sensitive abstention device rather than an accuracy-driven selector.

Theory implication:

- This supports the decision-theoretic shift from “predict the best algorithm” to “optimize a risk functional of net gain with abstention”.
- It also gives a clean pathway to write the router as a **risk-constrained (or risk-regularized) threshold rule** without introducing stronger assumptions than the one-sided calibration already requires.

## E6: Temporal Generalization on the Massive Archive (Checkpoint-Based)

Script:

- `scripts/paper_experiments/run_theory_temporal_generalization_massive.py`

Output:

- `results/theory_validation_latest/temporal_generalization_massive.json`

Setup:

- Train: `massive_pde_database_v1` (+ extensions / protocol runs), Test: `expansion_20260826` (PDE + HEAT-TIME).
- Features are taken from `run_checkpoint.jsonl` at `K=20` (so we reuse archived intermediate checkpoints without replaying trajectories).

Observed results (temporal holdout test, `K=20`):

- **tau0** (threshold `0`)
  - mean gain: `+0.055`
  - switch rate: `13.0%`
  - CVaR_10: `-0.0011`
- **conformal_lb** (`alpha=0.1`)
  - mean gain: `+0.055`
  - switch rate: `13.0%`
  - CVaR_10: `-0.0011`
- **cost-regularized conformal** (`lower_bound > lambda*(K/B)`)
  - `lambda=0.10`: mean gain `+0.034`, switch rate `6.5%`, CVaR_10 `0.0`
  - `lambda>=0.20`: no switching on this temporal holdout set

Interpretation:

- This uses the existing large historical archive directly (via checkpoints) and tests on a newer expansion cohort, so it is closer to a deployment-like “train on past archive → route on new instances” protocol.
- The opportunity-cost regularizer behaves as intended: increasing `lambda` shrinks the trigger set and collapses tail risk to `0.0` once switching becomes rare.

## E5: Affine Robustness Audit

Script:

- `scripts/paper_experiments/run_theory_affine_invariance_audit.py`

Output:

- `results/theory_validation_latest/affine_invariance_audit.json`

Observed route consistency:

- **multiplicative scaling**
  - `(a=0.5, b=0.0)`: route consistency `100.00%`
  - `(a=2.0, b=0.0)`: route consistency `100.00%`
- **additive shift**
  - `(a=1.0, b=10.0)`: route consistency `84.72%`
  - `(a=3.0, b=5.0)`: route consistency `89.98%`

Observed feature behavior:

- Most budget, feasibility, time, and combinatorial features remain unchanged.
- The most unstable dimensions under additive shift are:
  - `rel_best_gain`
  - `rel_auc_gain`
  - in some cases `diff_stagnation`

Interpretation:

- The current implementation is close to **multiplicative scale robustness**.
- It is **not** strictly positive-affine invariant in its present form.
- Additive objective shifts can materially perturb pairwise normalized gain features and downstream routing decisions.

Theory implication:

- **Proposition 5** should not currently be stated as full positive-affine invariance for the implemented feature pipeline.
- A more accurate near-term claim is:
  - the current trajectory representation is robust to multiplicative rescaling,
  - but not yet invariant to general positive affine transformations.

Engineering implication:

- If we want a stronger theorem here, we likely need to redesign the affected features so that they depend on:
  - rank-based progress,
  - ratio-of-differences terms,
  - or shift-invariant normalized improvements,
  instead of absolute best-loss surrogates.

## Overall Assessment

The theory validation pack produces three important conclusions:

1. **The strongest theory-supported claim is the decision-theoretic reframing.**
   - E1 gives direct empirical evidence that top-1 accuracy and deployment utility diverge sharply.

2. **The conformal story is defensible, but only in the marginal selective-risk sense.**
   - E2 supports cautious harmful-switch-mass language.
   - It does not support aggressive conditional-risk claims.

3. **The current feature pipeline is not yet theoretically clean enough for a full affine-invariance theorem.**
   - E5 exposes the exact gap between current implementation and the stronger desired proposition.

## Recommended Next Steps

If the paper is to emphasize theory, the next steps should be:

1. Keep **relative-gain routing with abstention** as the main theoretical center.
2. Keep **one-sided conformal gating** as a selective-risk mechanism, with strictly marginal language.
3. Rework the most shift-sensitive trajectory features before claiming full affine invariance.
4. Add one more theory-facing experiment for `K`:
   - remaining budget fraction,
   - conditional gain given switching,
   - and switch utility decomposition.

Among all theory directions, the cleanest package now is:

- Bayes-optimal relative-gain routing
- top-1 / utility misalignment
- marginal harmful-switch control
- budget-information trade-off for `K`

This package is already much stronger than a generic algorithm-selection story and remains aligned with the current experimental evidence.
