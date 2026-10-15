# joint-credit-v4：可迁移的Beta区间整数Count实验

本版本仍为独立实验，显式指定 `--proposed-design joint-credit-v4`；分配模式为 `stage-budget-v5`。Slow三个决策头及Fast继续联合PPO训练，部署候选、场景预算、时限和公共共享NIC/KKT保持原设置。

## 为什么调整Count

前一整数版本 `stage-budget-v4` 的共享额度评分MLP将图上下文与q的线性特征拼接。若不同q落在相同ReLU激活区间，图上下文对各额度的贡献接近共同偏移，容易得到近线性的额度排名。这是局部表达/优化问题，不是证明该MLP永远不能学习。

真实第12次快照在两个60分钟开发点中，Count归一化熵均值0.9999936，66.09%的确定性Count选择物理最大额度，众数与四舍五入期望值在97.41%的动作上不同。只读解码消融将均值从243.1298降至229.6803ms，但DARP为183.5809ms。解码改动不足以接受该版本，也不能从高熵单独推断完全没有学习。

## 动作概率与执行

新Count复用旧Beta网络输出的两个浓度参数，由决策前状态直接控制分布形状。对全部物理整数q=0,...,M计算Beta在对应区间的质量，再归一化为Categorical分布。实现使用32点Gauss-Legendre求积，梯度经过浓度参数和区间质量计算。

这是一种数值定义的离散策略，不把连续Beta密度冒充整数概率。合法类的相对logit下界为-60以保留正的浮点支持；没有物理候选截断。测试用独立SciPy CDF比较M=0/1/3/5/31及平坦、偏斜和高浓度参数，允许绝对2e-5/相对2e-4的数值误差；该测试范围不等于任意浓度下的全域精度认证。

PPO采样并执行整数q，存储该类的log-prob。后续观测使用已执行q/M和真实资源消耗，不保留未执行的小数动作。真实窗口KKT回报是唯一训练标签。

本版本的显式默认解码为 `categorical-mean`：将该离散分布的期望额度四舍五入到合法整数，并计算该整数的原始类概率。可显式选择 `categorical-mode` 做消融。随机训练均从同一个Categorical分布采样。旧v2仍默认mean，旧整数v3仍默认categorical-mode，旧检查点没有静默升级。

## 初始化、续训与受控实验

新增设计支持旧 `stage-budget-v3` → 新 `stage-budget-v5` 的全Actor参数迁移：保留已学习的Count参数网络、Stage和Placement，给新增额度上下文列补零权重。Count的概率律与后续观测语义改变，因此是新实验；不保留优化器、Critic、价值回放或best历史，不能当作完整续训。

同分配模式也可迁移到新运行。`--init-slow-actor-scope stage-placement`只迁移两个头，保留新Count初始化，用于架构不同且不能迁移Count的实验。完整续训仍校验分配模式、上下文、解码等版本。

每个显式Slow Actor初始化的新运行保存 `checkpoints/initial.pt`，含实际训练前参数、来源、配置和随机状态，且不参与best选择。续训保留原初始化来源。两臂的训练种子、Fast来源和预算应相同；先比较initial，再比较更新后与DARP的差，避免把初始化变化当作学习收益。

```powershell
# 默认从不可变旧v2第140次快照迁移全部Slow Actor和Fast Actor。
.\scripts\train_joint_credit_v4.ps1 -Updates 6 -Device cpu

# 控制组：同来源、相同预算，保留旧Beta连续动作。
.\scripts\train_joint_credit_v2.ps1 -Updates 6 -Device cpu -InitFastCheckpoint <同一来源> -InitSlowCheckpoint <同一来源>
```

参考Fast组件评估和各自联合Fast评估均需保留。独立测试需求、其他训练种子、五场景及更密采样验收仍是后续要求，短程开发点改善不代表最终目标完成。
