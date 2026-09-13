# Conformal Risk-Aware Algorithm Routing from Endogenous Trajectories in Expensive Black-Box PDE Optimization

## Abstract
Algorithm selection for expensive black-box optimization, particularly in Partial Differential Equation (PDE) simulations, faces a critical dilemma: traditional Exploratory Landscape Analysis (ELA) and recent deep learning methods demand unaffordable initial sampling budgets, while aggressive algorithm switching often leads to catastrophic performance degradation on out-of-distribution (OOD) problems. To address this, we reformulate the algorithm selection problem from an absolute Top-1 classification task into a risk-aware pairwise net-gain routing problem. We propose a budget-neutral endogenous representation by extracting scale-normalized trajectory features from the first $K$ evaluations of a default base optimizer. Furthermore, we introduce a conformal lower-bound gating mechanism that triggers an algorithm switch only when the calibrated lower confidence bound of the predicted Anytime AUC net gain is strictly positive. Evaluated on 2,211 PDE instances across 8 distinct physical families under a Leave-One-Family-Out (LOFO) setting, our framework achieves positive expected net gain while substantially reducing downside switching risk compared to static ELA and majority baselines. The results demonstrate a principled, budget-neutral, and risk-controllable decision paradigm for autonomous scientific optimization.

---

## 1. Introduction
In engineering domains such as nuclear reactor design and thermal-fluid simulations, optimization problems are typically black-box and computationally expensive, with a single evaluation taking minutes to hours. The Automated Algorithm Selection (AAS) problem seeks to map problem instances to the most suitable optimization algorithm. 

Traditional approaches rely on Exploratory Landscape Analysis (ELA) or deep point-cloud representations (e.g., Deep-ELA) to characterize the objective function landscape. However, these methods suffer from two critical limitations in expensive engineering scenarios:
1. **Sampling Overhead**: They require a dense pre-sampling budget (often $\ge 50 \times d$), which completely exhausts the strictly limited optimization budget.
2. **Generalization Collapse**: Standard models frame AAS as an absolute "Top-1 classification" task. When faced with out-of-distribution (OOD) physical families (e.g., transferring from an advection PDE to a diffusion PDE), static absolute features fail to generalize, leading to high-frequency erroneous switches and severe negative optimization gains.

To overcome these barriers, we introduce a **Conformal Risk-Aware Routing Framework**. Instead of dedicated pre-sampling, we extract budget-neutral trajectory features from the initial search path of a default optimizer. Instead of absolute classification, we predict the relative net gain. Instead of point-estimate decisions, we apply conformal prediction to strictly bound downside risks.

---

## 2. Methodology

### 2.1 Budget-Neutral Endogenous Trajectory Representation
Traditional ELA characterizes a function $f$ using independent samples $\phi(f; X_{sample})$. In contrast, we view the optimization process as a dynamical system. We allow a robust base optimizer $\mathcal{A}_{base}$ (e.g., Nelder-Mead) to execute for $K$ steps, producing an endogenous trajectory $\mathcal{T}_{1:K}$. We extract scale-normalized features $\phi(f, \mathcal{A}_{base}; \mathcal{T}_{1:K})$, including normalized improvement rate, stagnation length, and simplex contraction ratio. This approach avoids dedicated landscape pre-sampling and explicitly accounts for feature acquisition cost, ensuring the diagnostic process is strictly budget-neutral.

### 2.2 Pairwise Net-Gain Regression
Conventional AAS maximizes the probability of selecting the optimal algorithm: $\arg\max_a P(a=a^*)$. However, in expensive simulations, avoiding a disastrous algorithm is more critical than finding the marginal best. We reformulate the objective to maximize the expected Anytime AUC net gain relative to the base algorithm:
$$ \max_a \mathbb{E}[\Delta \mathcal{G}(a, \mathcal{A}_{base}) - C_{switch}] $$
where $\Delta \mathcal{G}$ represents the logarithmic area-under-the-curve (Log-AUC) reduction achieved by switching to candidate $a$.

### 2.3 Conformal Lower-Bound Gating
To mitigate the risk of OOD hallucination, we apply split conformal prediction. Using a calibration set, we compute the empirical quantile $q_{1-\alpha}$ of the over-prediction residuals. The routing decision $\pi$ is governed by a conformal lower-bound gate:
$$ 
\pi(\mathcal{T}_{1:K}) = 
\begin{cases} 
\arg\max_a \widehat{\Delta \mathcal{G}}_a, & \text{if } \max_a \widehat{\Delta \mathcal{G}}_a - q_{1-\alpha} > 0 \\
\mathcal{A}_{base}, & \text{otherwise}
\end{cases}
$$
Under exchangeability, this provides finite-sample marginal validity. In our OOD (LOFO) PDE shifts, we empirically evaluate its robustness using family-level aggregation and downside-risk (CVaR) metrics.

---

## 3. Experimental Setup
*   **Dataset**: 2,211 expensive PDE optimization instances spanning 8 distinct physical families (e.g., Advection, Black-Scholes, Neutron Diffusion, Thermal Fins).
*   **Validation Protocol**: Leave-One-Family-Out (LOFO) cross-validation. In each fold, 7 families are used for nested training (80% train, 20% calibration), and the entirely unseen 8th family is used for testing.
*   **Metrics**: Unconditional Mean Log-AUC Gain, Win Rate (strictly positive gain), Loss Rate (strictly negative gain), and Conditional Value at Risk ($\text{CVaR}_{0.1}$) to quantify tail collapse risk.

---

## 4. Results and Discussion

### 4.1 Generalization Collapse of Traditional Baselines
We first evaluate the traditional static ELA (Top-1 Classification) approach. While static ELA achieves acceptable accuracy on seen distributions, its performance degrades severely under LOFO validation. The static classifier exhibits a high Switch Rate (86.3%) but incurs a substantial Loss Rate (35.6%), resulting in a highly negative tail risk ($\text{CVaR}_{0.1} = -1.300$) and an expected negative net gain. This confirms that static absolute features lack robust generalization across diverse physical scales.

### 4.2 Orthogonal Ablation of Features and Gating
To decouple the contributions of trajectory representation and risk gating, we conducted an orthogonal ablation study:
*   **Traj_NoGate vs. Static_NoGate**: Replacing static features with trajectory features without gating fails to prevent high loss rates (36.6%), indicating that trajectory features alone are insufficient to guard against OOD uncertainty.
*   **The Impact of Conformal Gating**: When applying the conformal gate ($\alpha=0.1$) to the static model (`Static_ConfGate`), the system abstains from switching entirely (Switch Rate 0.0%), revealing that static features lack the necessary confidence lower bounds in OOD scenarios. Conversely, our full framework (`Traj_ConfGate`) retains a targeted switch rate, achieving positive mean gain while suppressing the loss rate.

### 4.3 Risk-Return Frontier via $\alpha$ Sensitivity
The conformal parameter $\alpha$ serves as a tunable risk-return threshold. As $\alpha$ tightens from $0.3$ to $0.05$:
*   The Switch Rate drops smoothly from 14.1% to 0.0%.
*   The Loss Rate is aggressively curtailed from 4.7% down to $<0.1\%$.
*   The tail risk ($\text{CVaR}_{0.1}$) converges from $-0.069$ to $0.000$.
This demonstrates that the conformal gate is not an arbitrary heuristic but a principled mechanism that establishes a stable Pareto frontier between optimization acceleration and downside risk.

### 4.4 Ablation on Trajectory Length $K$
We evaluate the sensitivity of the diagnostic window $K \in \{5, 10, 15, 20, 30\}$. A trade-off emerges between information acquisition and budget squeeze:
*   At $K=5$, the geometric topology of the objective function is under-explored, yielding high uncertainty and a higher loss rate (18.5%).
*   At $K=30$, while the decision is highly secure (Loss Rate 8.5%), the prolonged probing consumes excessive optimization budget, suppressing the achievable net gain.
*   The interval $K \in [10, 20]$ forms an optimal plateau, providing sufficient dynamical evidence to diagnose stagnation while preserving adequate budget for the candidate algorithm to converge.

---

## 5. Conclusion
This paper reframes automated algorithm selection for expensive BBO from a top-1 classification problem into a risk-constrained pairwise routing decision. By leveraging endogenous search trajectories and conformal prediction, our framework avoids the exorbitant costs of dedicated landscape sampling. Extensive LOFO experiments on real-world PDE simulations demonstrate that the proposed method achieves positive expected net gain while substantially reducing downside switching risk. It provides an interpretable, budget-neutral, and theoretically grounded numerical safeguard for autonomous scientific optimization systems.
