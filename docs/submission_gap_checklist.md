# AICO-Intellig Submission Gap Checklist

## 1. Current Status

The project has progressed beyond the prototype stage and already contains a paper-shaped method narrative, working LOFO evaluation scripts, trained lightweight routing models, and a large archive of PDE optimization results. However, it is not yet submission-ready. The main issue is no longer whether the method works at all, but whether the paper claims, the result assets, and the evaluation universe are fully aligned.

At the moment, the strongest part of the project is the routing story itself: pairwise gain prediction from endogenous early trajectories, combined with conservative switching rules, already produces positive mean gain under the current engineering LOFO protocol. The weakest part is the evidence closure: family coverage is incomplete, several scripts use different data universes, and the conformal claim is only partially supported by the current outputs.

An additional engineering note emerged during the refresh on 2026-08-18: several paper-facing scripts originally built LOFO splits from unsorted Python sets, and CatBoost was left in its default parallel mode. Both choices can introduce small run-to-run drift in the summary tables. The current working tree has been patched to sort group/family identifiers and force single-thread CatBoost execution in the main paper scripts, so the next exported snapshot should be more stable.

## 2. Low-Cost Experiments Run on 2026-08-18

These experiments were re-run directly from the current working tree using the existing scripts under `scripts/paper_experiments/`.

### 2.1 Unified Engineering LOFO

Script:

- `scripts/paper_experiments/run_unified_lofo_metrics.py`

Observed results:

- Majority baseline: win `42.3%`, loss `44.0%`, switch `86.3%`, unconditional mean gain `-1.115`
- Static ELA top-1: win `53.2%`, loss `12.8%`, switch `66.0%`, unconditional mean gain `-0.157`
- Ours: win `30.6%`, loss `11.5%`, switch `42.0%`, unconditional mean gain `+0.066`
- Wilcoxon: Ours vs Majority `p = 3.40e-18`
- Wilcoxon: Ours vs Static ELA `p = 7.25e-07`
- Bootstrap 95% CI for Ours mean gain: `[+0.003, +0.109]`

Interpretation:

- The current pairwise-gated method already has the strongest directly usable main result.
- The paper can safely claim positive expected gain against both majority and static ELA under the present LOFO engineering setup.
- The method does not maximize switch frequency or win rate; it trades those for lower downside.

### 2.2 Family-Level Aggregation

Script:

- `scripts/paper_experiments/run_family_level_stats.py`

Observed results:

- Macro-average gain:
  - Majority: `-1.092`
  - Static: `-0.094`
  - Ours: `+0.050`
- Variance of family-level mean gains:
  - Majority: `2.492`
  - Static: `0.078`
  - Ours: `0.002`
- Families present in the output:
  - `ex_advection2d_family_v1`
  - `ex_blackscholes2d_family_v1`
  - `ex_heat_time_family_v1`
  - `neutron_diffusion_family_v1`
  - `thermal_fins_family_v1`

Interpretation:

- The strongest family-level message is not significance but stability: the method has by far the lowest inter-family variance.
- The previous `unknown` label has been removed by improving family inference in `train_curve_policy.py`, so the family-level table is now much closer to paper-ready form.
- The table is still not fully closed because the evaluation universe itself remains mixed across result directories.

### 2.3 K Ablation

Script:

- `scripts/paper_experiments/run_K_ablation.py`

Observed results:

- `K=5`: mean gain `+0.062`, loss `12.3%`, switch `55.6%`, CVaR_10 `-0.637`
- `K=10`: mean gain `+0.093`, loss `14.9%`, switch `60.0%`, CVaR_10 `-0.642`
- `K=15`: mean gain `+0.037`, loss `10.4%`, switch `37.3%`, CVaR_10 `-0.529`
- `K=20`: mean gain `+0.051`, loss `7.1%`, switch `32.2%`, CVaR_10 `-0.048`
- `K=30`: mean gain `+0.027`, loss `7.5%`, switch `29.4%`, CVaR_10 `-0.068`

Interpretation:

- This is already usable as evidence that `K` controls an explicit risk-return trade-off.
- The result does not support a simplistic "larger K is always better" story.
- `K=10` maximizes gain in the current script, whereas `K=20` is much safer in the tail.

### 2.4 Conformal Risk Routing

Script:

- `scripts/paper_experiments/run_conformal_risk_routing.py`

Observed results:

- Majority baseline: mean gain `+0.055`, loss `27.7%`, switch `91.2%`, CVaR_10 `-1.222`
- Fixed empirical gate (`tau=0.1`): mean gain `+0.035`, loss `7.9%`, switch `26.3%`, CVaR_10 `-0.048`
- Conformal gate (`alpha=0.1`): mean gain `+0.049`, loss `8.8%`, switch `31.1%`, CVaR_10 `-0.053`
- Wilcoxon against Majority is not significant in this script output

Interpretation:

- The script supports the broad claim that conservative gating sharply lowers downside relative to aggressive switching.
- It does not yet support a strong paper claim that conformal gating is clearly superior to the simpler fixed gate.
- The current conformal implementation should be treated as exploratory until it is cleaned and aligned with the main evaluation universe.

### 2.5 Final Ablations and Orthogonal Baselines

Scripts:

- `scripts/paper_experiments/run_final_ablations.py`
- `scripts/paper_experiments/run_comprehensive_baselines.py`

Observed results from `run_final_ablations.py`:

- `Static_NoGate`: mean gain `+0.019`, loss `35.8%`
- `Static_PointGate`: mean gain `+0.091`, loss `17.5%`
- `Static_ConfGate(a=0.1)`: mean gain `+0.000`, switch `0.0%`
- `Traj_NoGate`: mean gain `-0.052`, loss `37.5%`
- `Traj_PointGate`: mean gain `+0.055`, loss `17.7%`
- `Traj_ConfGate(a=0.3)`: mean gain `+0.016`, loss `9.5%`
- `Traj_ConfGate(a=0.2)`: mean gain `-0.012`, loss `2.5%`
- `Traj_ConfGate(a=0.1)`: mean gain `+0.010`, loss `0.9%`
- `Traj_ConfGate(a=0.05)`: mean gain `+0.002`, switch `0.1%`

Observed results from `run_comprehensive_baselines.py`:

- `B4_StatTop1`: mean gain `-0.109`
- `B5_TrajTop1`: mean gain `-0.252`
- `B6_StatGate`: mean gain `+0.089`
- `B7_TrajNoGate`: mean gain `+0.027`
- `Ours_TrajGate_0.1`: mean gain `+0.013`
- Tau frontier:
  - `tau=0.0`: `+0.027`
  - `tau=0.05`: `+0.024`
  - `tau=0.1`: `+0.013`
  - `tau=0.2`: `-0.006`
  - `tau=0.5`: `-0.008`

Interpretation:

- These scripts support a valid conservative-switching story.
- They do not support a simple "trajectory features alone beat static features" story.
- The evidence suggests that gating is doing a substantial part of the work.
- The current alpha and tau sweeps are useful for method analysis, but the exact paper-facing story must be chosen carefully.

## 3. What Is Already Supported

The following claims are already reasonably supported by current evidence:

- Classical static top-1 style routing is not reliable under engineering LOFO transfer.
- A pairwise gain framing can produce positive expected gain where static top-1 does not.
- The method's main advantage is conservative risk control rather than aggressive switching.
- Family-level stability is currently stronger than family-level significance.
- The choice of `K` materially changes the gain-risk frontier.

## 4. What Is Not Yet Supported Enough

The following claims are not yet strong enough for direct submission:

- A strong headline claim that conformal gating decisively beats simpler threshold gating.
- A clean "8 families / 2211 instances" statement unless the data universe is first frozen and re-counted.
- A claim that trajectory features alone are consistently superior to static features.
- Any statement that mixes different evaluation universes without explicit explanation.

## 5. Innovation Positioning

The innovation should not be presented as "another algorithm recommender." The more defensible framing is a shift in problem definition and evaluation philosophy.

### 5.1 Main Innovation

Reformulate algorithm selection for expensive PDE optimization from absolute top-1 prediction into a risk-constrained pairwise routing decision relative to a stable default optimizer.

This is the strongest conceptual point because it changes the optimization target itself:

- not "which algorithm is globally best?"
- but "is it worth switching away from the default now, under risk constraints?"

### 5.2 Second Innovation

Use endogenous early-search trajectories as budget-neutral diagnostic evidence instead of extra pre-sampling for static landscape analysis.

This should be emphasized as a practical and methodological contribution:

- feature acquisition is coupled to the actual optimization run
- no separate up-front probing budget is required
- diagnostic evidence is aligned with the deployed default optimizer

### 5.3 Third Innovation

Use conservative switching rules, optionally with conformal lower-bound logic, to control downside risk under LOFO/OOD family transfer.

This point should be stated carefully. The current evidence strongly supports conservative gating in general, but only partially supports a strong unique conformal advantage.

### 5.4 Fourth Innovation

Evaluate routing quality using engineering-safe metrics such as:

- mean gain
- loss rate
- switch rate
- tail risk / CVaR

This is more appropriate than pure top-1 accuracy for expensive black-box simulation settings.

## 6. Minimum Submission Tasks

The minimum set of tasks before a serious submission attempt should be:

1. Freeze the evaluation universe.
   - Decide the exact family set.
   - Decide the exact instance count.
   - Remove `unknown`.

2. Re-export one clean main result table.
   - Use a single metric schema:
     - mean gain
     - win rate
     - loss rate
     - switch rate
     - CVaR_10

3. Re-export one clean family-level table.
   - Only valid families.
   - Only one evaluation universe.

4. Re-export one K-ablation figure/table.
   - Use the current `run_K_ablation.py` result as the starting point.

5. Decide the conformal claim level.
   - Either:
     - keep conformal as a central contribution and clean the protocol thoroughly
   - or:
     - downgrade conformal to one conservative gating variant and let the main contribution be pairwise risk-aware routing

6. Rewrite the abstract and introduction around the supported claims only.

## 7. Recommended Paper Claim Set

If the goal is to converge quickly, the cleanest paper claim set is:

- Claim 1:
  Expensive PDE algorithm selection should be cast as relative gain-based routing instead of absolute top-1 classification.

- Claim 2:
  Endogenous early-search trajectories provide budget-neutral routing evidence that transfers better than static top-1 selection under LOFO engineering shifts.

- Claim 3:
  Conservative gating yields a controllable risk-return frontier and can maintain positive mean gain while suppressing harmful switches.

- Claim 4:
  Evaluation must prioritize gain, loss, switch frequency, and tail risk, not just prediction accuracy.

## 8. Immediate Next Experiments

The following experiments are worth doing next because they are likely low cost and directly useful:

1. Clean family relabeling and unified recount.
2. Re-run the main unified LOFO table on the frozen family universe.
3. Re-run family-level aggregation on the same frozen universe.
4. Export `K` ablation to a stable CSV or markdown table.
5. If conformal remains central, run one cleaned alpha sweep with a single agreed protocol.

The following should be delayed until after the main evidence is frozen:

- new large simulation sweeps
- new external problem families
- deep-model expansion
- publication-oriented packaging or open-source release
