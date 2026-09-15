**AICO-Intellig：相关工作与理论选题调研**

检索日期：2026-09-13。项目参考版本：`2c339f0`。本笔记用于选题与后续精读，不是穷尽性系统综述，也不构成“某方向无人研究”的证明。

证据层级：对最接近的轨迹选择、热启动、RL2CO、预算和风险控制工作查看了原文相关章节；其余以作者预印本摘要或正式出版页面核对。尚未逐条复核所有证明、代码或实验。论文报告的结果不等于已独立复现。会议、预印本和工作坊摘要分别标示。微信链接未取得正文，尚不能确认它对应哪篇论文。

**本轮检索改变了什么判断**

“默认优化器运行一段时间→从已有轨迹提取特征→选择后继算法→热启动”已有非常直接的先例。2022 年的 per-run algorithm selection 是当前必须对照的方法。2026 年 9 月的 RL2CO 进一步联合学习算法选择与运行时长。因此，真实切换、自适应 K、状态传递都不能单独作为首次提出的贡献。

“保留默认策略、校准门控、控制错误切换”也处于既有安全策略改进与风险控制体系内。较有潜力的选题需要明确新条件、新困难和超出标准工具直接推论的结果，例如：档案支持不足和状态迁移误差下，达到安全改进所必需的补充信息，以及安全与预算效率之间的可证明关系。以下方向判断均为本次分析提出的研究假设。

**直接相关：轨迹选择、热启动与调度**

| 编号 | 工作与来源 | 已有内容 | 对 AICO 的影响 |
|---|---|---|---|
| A1 | [Towards Dynamic Algorithm Selection for Numerical Black-Box Optimization: Investigating BBOB as a Use Case](https://arxiv.org/abs/2006.06586)，Vermetten 等，2020 | 研究 BBOB 档案中的单次算法切换潜力。 | “动态切换优于固定算法”不能作为新概念。 |
| A2 | [Switching between Numerical Black-box Optimization Algorithms with Warm-starting Policies](https://arxiv.org/abs/2204.06539)，Schröder 等，2022 | 实际执行档案预测的切换，分析热启动和切换时刻对收益的影响。 | 预测收益与实际切换收益的差异已有直接研究；需做更明确的识别或误差理论。 |
| A3 | [Trajectory-based Algorithm Selection with Warm-starting](https://arxiv.org/abs/2204.06397)，Jankovic 等，2022 | 从优化路径计算特征，在统一预算下进行单次切换与热启动。 | “省去独立 ELA 采样”已有先例。 |
| A4 | [Per-run Algorithm Selection with Warm-starting using Trajectory-based Features](https://arxiv.org/abs/2204.09483)，Kostovska 等，2022 | 默认 CMA-ES 前缀；轨迹 ELA 与内部状态时间序列特征；预测后继算法表现；热启动。在 BBOB 和 YABBOB 上评估。 | 当前最直接的对照之一。阅读重点是数据生成：后继运行共享初始前缀，而非仅比较独立运行。 |
| A5 | [Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution](https://arxiv.org/abs/2403.02131)，Guo 等，IEEE TSMC: Systems，2024 | 强化学习根据搜索状态选择 DE 算法，包含上下文恢复机制。[作者代码](https://github.com/MetaEvo/RL-DAS)。 | 动态调度与算法状态特征都有直接先例；其 DE 组合和训练成本需与 PDE 场景区分。 |
| A6 | [Greedy Restart Schedules: A Baseline for Dynamic Algorithm Selection on Numerical Black-box Optimization Problems](https://arxiv.org/abs/2504.11440)，Schäpermeier，GECCO 2025 | 根据尚未解决的训练问题分布构建不依赖新实例特征的贪心重启计划。 | 必须加入简单强基线，检验模型收益是否超过组合调度自身的收益。 |
| A7 | [Reinforcement Learning to Choose Optimizers](https://arxiv.org/abs/2609.01811)，van der Schelling、Toshniwal、Bessa，2026-09-01 预印本 | RL2CO 联合选择优化器与执行时长，传递当前最佳解和代表性步长，组合梯度与无梯度算法。[作者代码](https://github.com/bessagroup/rl2co)。 | 直接覆盖“算法与时长联合学习”。原文主要研究可微设计问题，并报告跨分布实验；这些实验不是一般性的风险保证。 |
| A8 | [Towards Dynamic Switching in Per-Run Algorithm Selection，COSEAL 2026 摘要](https://www.coseal.net/wp-content/uploads/2026/04/COSEAL-2026-Workshop-Schedule.pdf)，Geuchen 等 | 摘要明确描述联合选择切换点和后继算法。 | 活跃竞争方向；本轮仅核实工作坊摘要，不能据此评价完整方法或证明。 |

**预算、信息获取与理论基线**

| 编号 | 工作与来源 | 已有内容 | 对 AICO 的影响 |
|---|---|---|---|
| B1 | [Non-stochastic Best Arm Identification and Hyperparameter Optimization](https://proceedings.mlr.press/v51/jamieson16.html)，Jamieson、Talwalkar，AISTATS 2016 | 将迭代算法的资源分配建模为非随机最优臂识别，分析 Successive Halving。 | 不能把“提前淘汰差算法、给好算法更多预算”单独当作理论创新。 |
| B2 | [Stochastic Rising Bandits](https://proceedings.mlr.press/v162/metelli22a.html)，Metelli 等，ICML 2022 | 处理回报随投入增长的臂，并给出相应遗憾分析。 | 优化器表现随已用预算变化，可从这里找理论参照；必须核实所需增长结构是否符合实际数据。 |
| B3 | [Best Arm Identification for Stochastic Rising Bandits](https://proceedings.mlr.press/v235/mussi24b.html)，Mussi 等，ICML 2024 | 固定预算识别、误选概率、simple regret 和下界；说明预算门槛的必要性。 | “识别好算法需要多少预算”已有较强理论；新结果必须体现共享状态、切换、基线约束等额外结构。 |
| B4 | [On the Influence of the Feature Computation Budget on Per-Instance Algorithm Selection for Black-Box Optimization](https://arxiv.org/abs/2605.04954)，van der Blom、Vermetten，2026-05 预印本 | 系统改变特征采样预算，研究特征精度与最终选择收益的权衡。 | K 扫描本身不足；应研究可证明的自适应诊断代价，并区分前缀优化价值与额外探测成本。 |

AICO 中继续运行会同时改善解和获得信息，而热启动又可能让候选算法的未来表现依赖其他算法的历史。因此不能未经说明就把每个优化器当成一条只依赖自身投入次数、互不影响的独立学习曲线。这个差别是值得形式化的研究入口，尚不是已确立的新结论。

**安全改进、共形路由与顺序控制**

| 编号 | 工作与来源 | 已有内容 | 对 AICO 的影响 |
|---|---|---|---|
| C1 | [Conservative Bandits](https://proceedings.mlr.press/v48/wu16.html)，Wu 等，ICML 2016 | 在相对固定基线的保守约束下探索，分析风险和遗憾代价。 | “安全与探索效率同时分析”已有基础；其累积回报约束与单次优化终局收益需区分。 |
| C2 | [Safe Policy Improvement with Baseline Bootstrapping](https://proceedings.mlr.press/v97/laroche19a.html)，Laroche 等，ICML 2019 | 利用离线数据改进基线，在数据支持不足处限制偏离。 | 档案驱动、缺乏支持时回退默认策略的直接理论参照。 |
| C3 | [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052)，Angelopoulos 等，2021 起的预印本 | 将风险校准化为多重假设检验。 | 候选门控与阈值选择需处理多重选择；LTT 是应比较的标准方案。 |
| C4 | [Conformal Risk Control](https://arxiv.org/abs/2208.02814)，Angelopoulos 等，2022 起的预印本 | 将共形校准扩展到单调损失的期望风险控制。 | 普通 CRC 下界或门控是基础工具，不是新理论本身。 |
| C5 | [Conformal Decision Theory: Safe Autonomous Decisions from Imperfect Predictions](https://arxiv.org/abs/2310.05921)，Lekeufack 等，2023 起的预印本 | 直接校准决策风险，研究在线风险控制。 | 需精确比较长期平均风险、单次决策风险和反馈条件，不能混用保证。 |
| C6 | [Fast yet Safe: Early-Exiting with Risk Control](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract-Conference.html)，Jazbec 等，NeurIPS 2024 | 对提前退出机制做风险校准，平衡计算与输出质量。 | “顺序停止＋风险控制”已有近邻；优化中的状态改变、预算消耗与反事实收益需要单独刻画。 |
| C7 | [Conformal Arbitrage: Risk-Controlled Balancing of Competing Objectives in Language Models](https://papers.nips.cc/paper_files/paper/2025/hash/65a655c5a267f678fd3e897e4137ef53-Abstract-Conference.html)，Overman、Bayati，NeurIPS 2025 | 在主要模型和保守 Guardian 之间以共形校准平衡目标。 | 相对保守参考、效用与风险的门控组合已有工作，不能只更换成优化器就宣称理论首创。 |
| C8 | [Proactive Routing to Interpretable Surrogates with Distribution-Free Safety Guarantees](https://arxiv.org/abs/2603.14623)，Uddin 等，2026-03 预印本 | 输入门控；控制路由集合内的违例率，并讨论非空安全路由的可行性。 | “既安全又不能永远拒绝”也已被研究；需比较保证的具体量与阈值选择校正。 |
| C9 | [Conformal LLM Routing with Distribution-Free Safety Guarantees](https://aclanthology.org/2026.acl-srw.70/)，Uddin、Bauer，ACL 2026 Student Research Workshop | 将上述主动安全路由思路用于 LLM，给出路由子集违例率的高概率控制框架。 | 是跨应用近邻；不能误记为 ACL 主会主赛道，也不能把它当作优化轨迹论文。 |
| C10 | [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240)，Howard 等，Annals of Statistics 2021 | 构造时间一致的置信序列。 | 可选停止的理论工具已成熟；直接应用需满足对应过程条件，相关优化轨迹不能视作独立样本。 |
| C11 | [Conformal prediction after data-dependent model selection](https://arxiv.org/abs/2408.07066)，Liang、Zhu、Barber，2024 起，2026-04 修订 | 研究使用同一留出集进行模型选择和共形校准产生的覆盖偏差。 | 需要审计算法、K、模型、阈值多层选择；这与仅固定一个预测器的校准不同。 |

风险定义必须固定：边际有害切换概率、切换集合内部错误率、相对于默认策略的期望性能差、实际收益尾部风险、跨实例长期违例率，都是不同对象。普通共形边际覆盖不能直接推出切换条件下的风险保证；但在独立校准、有效检验及适当采样条件下，可以专门研究路由集合的选择风险。因此不能将上一条限制误读为所有条件风险控制都不可能。

**泛化、反事实与在线监测**

| 编号 | 工作与来源 | 已有内容 | 对 AICO 的影响 |
|---|---|---|---|
| D1 | [Landscape features in single-objective continuous optimization: Have we hit a wall in algorithm selection generalization?](https://doi.org/10.1016/j.swevo.2025.101894)，Swarm and Evolutionary Computation，2025 | 比较 ELA、拓扑及学习表示；在该研究的 OOD 测试中，特征选择器未超过单一最佳求解器。 | 跨族评估很有价值，但“发现 OOD 困难”本身已有明确文献。 |
| D2 | [Conformal prediction beyond exchangeability](https://arxiv.org/abs/2202.13415)，Barber 等，Annals of Statistics 2023 | 研究非可交换数据下的覆盖退化与加权处理。 | LOFO 不能无条件继承 IID/可交换校准保证；要声明漂移假设。 |
| D3 | [On Continuous Monitoring of Risk Violations under Unknown Shift](https://proceedings.mlr.press/v286/timans25a.html)，Timans 等，UAI 2025 | 以顺序检验监测风险违例并控制误报。 | 监测风险恶化与预先保证没有风险恶化不同；且通常需要可观测结果反馈。 |
| D4 | [Conformal Inference of Counterfactuals and Individual Treatment Effects](https://arxiv.org/abs/2006.06138)，Lei、Candès，JRSSB，2021 | 在随机或满足识别条件的观察性数据下构造反事实区间。 | “反事实＋共形”也不是新组合；历史优化档案是否覆盖真实切换干预需另行检查。 |
| D5 | [Quantile Learn-Then-Test: Quantile-Based Risk Control for Hyperparameter Optimization](https://arxiv.org/abs/2407.17358)，Farzaneh 等，2024 预印本 | 对风险分位数提供校准保证。 | 从均值改为尾部风险有先例；分位数与 CVaR 并不等价。 |

另有 [Confidence-bound early stopping of experiments with sequential calibration](https://doi.org/10.1016/j.cherd.2026.05.013)，Chemical Engineering Research and Design，2026：用历史实验训练最终结果预测器，对单个进行中实验实施提前停止，并研究错误停止控制。它与昂贵工程实验场景较接近，后续应精读其校准和验证协议；本轮仅按出版页核对。

**对现有创新表述的处置建议**

| 当前可能的表述 | 调整建议 |
|---|---|
| 首次利用优化早期轨迹选择算法 | 删除首次表述；A3、A4 是直接先例。 |
| 首次在切换时继承历史状态 | 删除首次表述；A2、A4、A5、A7 均涉及状态交接。 |
| 联合决定算法与切换时机 | 需对照 A7 和 A8；不能单独支撑创新。 |
| 默认策略加收益门控有理论依据 | 保留为基本决策结构，不能仅以条件期望最大化当主定理。 |
| 用共形方法保证安全切换 | 明确风险量和条件，并与 C3—C11 比较。 |
| K 存在信息与预算权衡 | 保留问题动机；K 扫描或凹减凸存在性证明较弱。 |
| 跨物理族泛化 | 保留为重要验证；需要超出 D1 已揭示的问题，提供方法或理论解释。 |

**收窄后的候选主线（研究建议，未证实新颖性）**

建议研究“历史档案与少量实际切换实验支持的、有限预算安全算法切换”。主问题不是仅把几个现成模块相加，而是问：当候选优化器的后续表现受当前状态与迁移误差影响时，为获得安全改进，需要多少额外信息，且这些信息是否值得剩余预算？

最小设定可先限制为一个默认算法、两个候选算法、有限决策时刻、一次切换和固定迁移协议。必须先选定效用：终局最好值或全程 anytime 效用；并将默认前缀、候选探测、迁移和后继执行计入同一预算。若墙钟时间为主预算，应把不同求解器的一次迭代成本显式区分。

有希望形成实质贡献的三个问题：

1. **档案信息何时足够。** 明确独立运行轨迹对实际切换收益的识别边界；再分析补充哪些状态或分支执行，能消除哪些不确定性。简单构造缺失干预的反例只是起点；有用的结果应指导最小数据采集。
2. **有效风险控制的价格。** 不仅控制自适应时刻和候选选择后的风险，还要界定安全策略相对可实现参考策略的效用损失、探测成本或发现有益切换所需预算。参考策略必须服从同一预算和信息约束；先知参考只能作为单独的上界。
3. **可迁移状态的误差如何传播。** 在明确的状态充分性或稳定性条件下，将状态摘要、收益估计和迁移误差传到最终策略效用界。不能只假设所有误差很小；应说明如何估计或通过实验约束它们。

直接的联合界、Bonferroni/alpha spending、有限时刻最大残差校准、整条固定策略的独立校准都应先作为基线。若新方法的保证只是这些方法的一行推论，主理论仍然不够。另一方面，即便没有新型共形构造，若能得到有意义的安全样本复杂度下界及接近下界的算法，也可能形成更清晰的贡献。

跨族泛化宜作为后续扩展：区分训练覆盖、协变量变化和收益机制变化，不以“分布无关”掩盖采样假设。相邻 checkpoint、同一实例多 seed 和多个候选不能未经论证当成独立校准样本。实际部署又未必观察到未选择算法或未继续默认策略的反事实结果，在线风险监测必须说明这些标签如何得到。

**接下来最有价值的工作**

1. 精读 A4、A2、A7、B3、C6 和 C2，形成“假设—可用观测—预算—效用—保证”的逐项矩阵。
2. 用少量可分支运行的问题核对三种收益：独立运行差、相同预算冷启动切换、从相同默认前缀热启动切换。
3. 比较默认不切换、固定时刻 per-run selector、贪心重启计划、统一预算的轮流探测/淘汰、经验收益门控、整策略独立校准门控。涉及 RL2CO 时应报告梯度可用性和训练成本差异。
4. 同时报告效用、实际切换率、边际有害切换率、切换条件下损失、下尾指标和全部预算。切换率应按决策记录计算，不能仅用非零收益代替。
5. 在确认真实切换具有稳定机会后再投入证明与扩展仿真。全程冻结 universe 与实例分组；paper 和 training 的成员有重叠，不可直接用作互斥训练测试集。

本轮未修改项目代码，未启动仿真或训练。最终选题仍需围绕一个精确定义的问题与最接近论文逐条查新，而不是仅检索关键词是否组合出现。
