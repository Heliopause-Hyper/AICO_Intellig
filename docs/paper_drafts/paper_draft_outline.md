# 论文拟定标题
**Conformal Risk-Aware Algorithm Routing from Endogenous Trajectories in Expensive Black-Box PDE Optimization**
(面向昂贵黑箱优化的保形风险感知内生轨迹算法路由)

---

## 1. 核心摘要与结论 (Abstract & Conclusion)
本文提出了一种面向昂贵黑箱优化的保形风险感知算法路由框架。与传统将算法选择视为绝对最优分类问题的方法不同，本文将其重构为基于默认底座算法的相对净收益决策问题。该框架利用底座优化器前 $K$ 步产生的预算内轨迹构造尺度归一化的内生表征，并通过成对回归预测候选算法相对于底座算法的 Anytime AUC 净收益。进一步地，本文引入共形下界门控机制，仅当校准后的预测净收益下界为正时触发切换，从而将算法路由由点估计决策转化为风险感知决策。

在包含 2211 个 PDE 实例和 8 个物理族的 LOFO 跨域验证中，实验结果表明，缺乏门控的路由器容易产生高频错误切换和显著尾部损失；经验点估计门控能够缓解该问题，而共形下界门控进一步降低了 Loss Rate 与 CVaR 下行风险。尽管 LOFO 场景不完全满足标准共形预测的可交换性假设，实验仍显示该门控机制随 $\alpha$ 收紧呈现一致的经验风险控制趋势。总体而言，本文方法在保持正向期望净收益的同时显著降低了负收益切换风险，为复杂工程仿真环境下的自动算法选择提供了一种可解释、预算中性且风险可调的决策范式。

---

## 2. 方法论创新 (Methodology)

### 2.1 预算中性的内生轨迹表征 (Budget-Neutral Endogenous Trajectory Representation)
*   **痛点**：传统 ELA 依赖额外全局采样，消耗大量预算。
*   **方法**：使用默认底座算法（Nelder-Mead）前 $K$ 步轨迹。
*   **改进**：提取尺度归一化（Scale-normalized）特征（如归一化下降率、停滞长度等），增强对目标平移、缩放和物理量纲的鲁棒性。

### 2.2 成对净收益回归 (Pairwise Net-Gain Regression)
*   **痛点**：Top-1 绝对分类在工程中风险极高且不关注切换收益。
*   **方法**：将目标重构为 $\max_a \mathbb{E}[\Delta G(a, A_{base}) - C_{switch}]$。
*   **改进**：直接回归预测 Log-AUC 净收益，符合工程风控逻辑。

### 2.3 共形下界门控 (Conformal Lower-Bound Gating)
*   **痛点**：点估计 $\widehat{\Delta G} > 0$ 过于自信，无法控制下行风险。
*   **方法**：引入共形预测计算残差分位数 $q_{1-\alpha}$，切换条件变为 $\widehat{\Delta G} - q_{1-\alpha} > 0$。
*   **无数据泄漏声明 (Data Leakage Prevention)**：在 LOFO 验证中，7 个训练 Family 采用 Nested Split（80% Training, 20% Calibration）。模型在 Training 拟合，$q_{1-\alpha}$ 在 Calibration 估计，完全隔离测试集。
*   **理论边界**：在标准可交换性假设下提供有限样本边际风险保证；在 LOFO（OOD）下评估其经验稳健性。

---

## 3. 图表设计规划 (Figures & Tables)

### Figure 1：方法框架图 (Framework Overview)
1. Nelder-Mead 前 $K$ 步轨迹
2. $\rightarrow$ 尺度归一化轨迹特征 (Scale-normalized trajectory features)
3. $\rightarrow$ 成对增益回归器 (Pairwise gain regressor)
4. $\rightarrow$ 共形下置信界计算 (Conformal lower bound)
5. $\rightarrow$ 切换 / 弃权决策 (Switch / Abstain)

### Figure 2：正交消融柱状图 (Orthogonal Ablation)
*   **横轴**：Static_NoGate, Traj_NoGate, Static_PointGate, Traj_PointGate, Static_ConfGate, Traj_ConfGate。
*   **纵轴**：并列展示 Mean Gain, Loss Rate, $\text{CVaR}_{0.1}$。
*   **目的**：解耦特征与决策机制，证明“轨迹特征 + 共形门控”缺一不可。

### Figure 3：风险-收益前沿曲线 (Risk-Return Frontier)
*   **横轴**：Loss Rate 或 $\text{CVaR}_{0.1}$
*   **纵轴**：Mean Gain
*   **数据点**：标注不同的 $\alpha \in \{0.05, 0.1, 0.2, 0.3\}$
*   **目的**：直观展示共形门控如何平滑调节风险-收益边界。

### Figure 4：轨迹长度 $K$ 的消融 (Trajectory Length Ablation)
*   **横轴**：$K \in \{5, 10, 15, 20, 30, 40\}$
*   **纵轴**：Mean Gain, Loss Rate, Switch Rate。
*   **目的**：支撑“20步轨迹足够诊断”的论点，展示信息不足的误切换与信息过剩的预算挤压。

### Table 1：跨物理族泛化性能表 (Family-Level LOFO Results)
*   **行**：8 个 PDE Family（ex_advection2d, ex_blackscholes2d 等）。
*   **列**：Static Gain, Deep-ELA Gain, Traj ConfGate Gain, Loss Rate, CVaR$_{0.1}$。
*   **目的**：展示细粒度结果，防止被质疑“单一族主导结果”，验证 OOD 泛化能力。

---

## 4. 评估指标体系 (Unified Metrics)
所有实验强制报告以下指标矩阵，以全面评估风控表现：
*   **Switch Rate**：考察门控是否过度保守
*   **Win Rate**：考察切换是否带来真实正收益
*   **Loss Rate**：核心下行风险指标
*   **Mean Gain**：无条件期望收益（Unconditional Gain）
*   **Conditional Mean Gain**：触发切换后的期望收益
*   **CVaR$_{0.1}$**：衡量最坏 10% 尾部崩溃风险
*   **Worst-family Gain**：跨族最差表现

