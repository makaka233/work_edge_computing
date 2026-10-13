# PPO 学习信号修复：guarded-minibatch-v2

## 备份与诊断

修改前在原分支备份并上传：`0bed5bbf54194fd0cbd049ab15ae4c5cc33a700f`。
不新建分支，不删除实验产物，不停止用户训练。

问题不是“完全没有梯度”：在旧 joint scratch 运行第 16 update 的 Fast
checkpoint 上，重新隔离采集三个窗口（每窗口仅两个时间样本）的 762 个动作，
Actor 全数据梯度范数约 0.01735。回报标准差为 83.22，回报与剩余请求比例
相关系数为 -0.889；批内位置/任务背景占用了很大一部分回报变化。

同一数据、同一网络、同一 Actor 学习率 2e-4，诊断用 minibatch=64：

| 更新方式 | 优化步数 | 采样近似 KL | 平均概率总变差 | 原数据未裁剪 surrogate 增益 |
| --- | ---: | ---: | ---: | ---: |
| 旧整批梯度，每 epoch 一步 | 4 | 0.0000282 | 0.002801 | 0.001339 |
| 普通小批次 PPO | 48 | 0.002125 | 0.024408 | 0.011445 |

这是更新机制的隔离诊断，**不是性能评估**。小批次 64、时间抽样 2、三个窗口
均不代表正式训练设置；surrogate 改善不能证明新轨迹时延一定下降。
诊断脚本为 `scripts/diagnose_policy_signal.py`，只使用独立环境和模型副本，
不写原 checkpoint，不控制其他进程。后续运行会读取当时的 latest，并非固定上述 checkpoint。

## 改动

1. **恢复真实 minibatch 优化。** 每个 epoch 遍历全部动作，每个小批次
   执行 Adam 更新，不再将成千上万动作压成一个优化步。Count、Stage、Placement
   在启用 deployment-aware-v2 时使用相同保护。
2. **整轮原子 KL 保护。** 每轮结束在完整数据上检查全局和分组 KL。
   同时检查原有采样近似 KL 和 categorical/Beta 分布的精确 KL。
   若超限或参数非有限，恢复这一轮的全部参数与 Adam 状态，步长减半后
   重放同一批次顺序，最多七次尝试。上一轮已接受的进度不丢失。
   KL 检查相对采集策略累计计算，不能逐轮重置参照。
3. **修复 Adam 回退快照别名。** 每次恢复都复制快照，防止后续重试修改
   快照中的历史动量。该正确性修复也应用于旧 full-rollout-v1 的回退路径。
4. **Fast 独立条件 Critic。** 原 Critic 只能读取 Actor 编码出的全局和
   图池化向量，且这些编码不接受 value 梯度。新 Critic 另读完整决策前
   global/node 原始特征，包含批内位置、剩余工作及候选节点账本；有自己的
   可训练 MLP，使用 5e-4 学习率、归一化目标和四轮额外 value-only 拟合。
   它不读取本次动作、动作后的代价或未来采样结果作为输入。
5. **独立诊断日志。** `logs/policy_update_metrics.csv` 记录实际优化步数、
   接受 epoch、Actor 梯度范数、精确 KL、分组精确 KL、回退次数、接受步长，
   以及 Critic 的采集前/拟合后解释方差。原 update/episode/rollout 日志保留。

所有动作仍来自 PPO masked 分布，覆盖全部合法节点。不是模仿算法、
候选库选择、确定性路径预选或分阶段训练。原 Fast 精确增量 KKT 奖励、
跨请求累计回报、batch 终止边界、Slow 真实窗口回报均保留。
没有改环境、服务需求、负载分布、抽样器、网络/KKT约束或对比算法。

全局 sample-weight 分母与 critic-window-weight 分母分别保留。乱序只改变
优化顺序，不复制、漏掉样本，不将每个短 minibatch 强行变成同权重。
接受的 epoch 有完整负载覆盖；所有尝试均被拒绝时覆盖率记为零，而不是伪造 100%。

## 兼容性

- CLI `--ppo-update-design auto`：新 repaired + deployment-aware-v2 训练
  选择 `guarded-minibatch-v2`；续训和 Fast-only 初始化优先读取来源 checkpoint 顶层版本，
  避免旧初始化命令意外切换到不兼容的 Critic。
- 缺少版本标记的历史模型视为 `full-rollout-v1`，旧 API 默认也不改变。
- 新模式改变 Fast Critic 的输入维度与优化器分组，**请从头训练 joint**。
  不能把旧模型直接当新模式断点续训；明确报错，不静默加载不兼容权重。
- 新 checkpoint 可以通过已有 Proposed 评估入口加载，不要求修改评估参数。

## 联合训练命令

建议先完整联合训练 80 updates 验证，而非改成 Slow/Fast 分阶段。
两个完整 60 分钟 episode / update，每窗口六个抽样秒、每样本最多代表100秒。
保持既有近似，不把抽样结果宣称为全秒仿真。

```powershell
$runName = "joint_minibatch_v2_s6_u80_$(Get-Date -Format 'yyyyMMdd_HHmmss')"; python .\train_dual_ppo.py --seed 2026 --pressure-profile mec-latency200 --training-repair batch-credit-v1 --slow-deployment-context deployment-aware-v2 --ppo-update-design guarded-minibatch-v2 --training-design trajectory-simultaneous --train-mode joint --rollout-unit window --updates 80 --episode-minutes 60 --deployment-interval-minutes 10 --fast-windows-per-update 12 --slow-windows-per-update 12 --sampled-seconds-per-window 6 --sampling-max-interval-seconds 100 --slow-allocation-mode budget-continuous-v2 --replicas-per-stage 0 --reward-mode latency --reward-scale 10 --fast-policy-kind gat_node_scorer --fast-chain-context chain-kkt-v3 --fast-counterfactual-credit-coef 0 --fast-counterfactual-credit-final-coef 0 --slow-counterfactual-credit-coef 0 --slow-replica-ablation-samples 0 --fast-oracle-diagnostic-requests 0 --fast-k-epochs 4 --slow-k-epochs 2 --slow-stage-k-epochs 3 --slow-placement-k-epochs 3 --fast-minibatch-size 512 --eval-interval 0 --device cuda --run-name $runName --progress-interval-seconds 10
```

不能以“KL 越大越好”或“熵下降就是成功”判断效果。观察同负载下时延、
实际策略变化、采集前 Critic 解释方差和完整样本覆盖；需要重复负载/独立数据
确认性能改善。当前修复不承诺任何预先规定的对比排名或收敛结果。

## 验证边界

专项回归覆盖：真实 minibatch 步数、全部已采样负载覆盖、精确/采样 KL 限制、
有既存 Adam 动量时七次拒绝仍完整回退、已知奖励偏好学习、最低代价节点随
输入轮换时 GAT 仍能学习、Critic 梯度隔离、原评估接口与新 checkpoint、旧初始化兼容。

CUDA 短测目录 `runs_tmp/guarded_minibatch_v2_cuda_smoke_20261014`：
原 32 节点/10 服务场景，从头联合训练一个完整 60 分钟 episode、六个窗口、
36 个抽样结算。Fast 48 步（旧机制仅4步），Count/Stage/Placement 为2/3/3步；
三个 Slow buffer 在这个短测试均未超过2048，故仍只有一个 minibatch/epoch。
所有 head 接受完整数据覆盖；非法动作0；Fast 精确 KL 为7.737e-5、最大分组
精确 KL 为8.112e-5，无回退；Actor 梯度范数0.01370。
Fast Critic 拟合后同批解释方差0.363，**不是独立验证集结果**。
批 KKT 增量总和与联合结算的最大数值残差1.42e-4秒（未除以批请求数），
沿用现有 NIC 求解器精度，未放宽约束或容差。
保存模型已通过普通 Proposed 接口在 CPU 加载并生成合法调度。
全量回归301项通过；最后增加非有限参数保护及兼容性核对后的专项31项通过
（二者包含重叠测试，不相加）。

该 CUDA 短测六窗口/update 耗时7分21秒。正式命令为十二窗口/update，
预计约14–16分钟/update，80 updates约19–22小时；这只是同机短测外推，
负载、部署规模、NIC求解难度和系统占用会造成波动。

短测在增加 Fast 采集前 EV 日志字段前启动，因此其日志不含该字段；
后续新训练会完整记录。这不改变短测时的 Actor/Critic 更新行为。
短测不能证明联合训练已收敛，不能把初始随机策略的2.57秒时延当作优化成绩，
也不能据此认定 Slow Placement 的信用问题已完全解决。
