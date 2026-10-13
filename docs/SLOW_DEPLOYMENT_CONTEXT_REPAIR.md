# Proposed Slow 部署劣势审计与修复

## 备份与范围

改动前备份已在原分支提交并核对 GitHub 远程一致：
`8c031c87495870fb602a7fc010657e80ddfcf5b5`。
[备份链接](https://github.com/makaka233/work_edge_computing/tree/8c031c87495870fb602a7fc010657e80ddfcf5b5)。
不新增分支，不删除 runs/results、模型、图片或历史日志。

依据为最新单种子五场景评估
`comparison_five_scenes_dpms_accel_s6_seed2026_20261013_205746`。
基准负载 Proposed/DARP 为 190.91/179.12 ms；同一个 Fast 和公共 KKT 下，
计算时延为 156.70/144.91 ms。两者内存利用率约相同，但副本分配不同。
这首先指向 Slow 的分配/放置，不证明 Fast 已达到最优。

本次不改变节点、服务、网络、抽样、时限、公共 KKT 或对比算法。
仍是 Slow PPO + Fast PPO，Count/Stage/Placement 为 Slow 内部三个动作头。
不增加固定模板、固定执行路径、候选集合截断、算法动作标签或行为克隆。

## 逐项审计

### 1. 状态别名：同样的副本数不代表同样的有效算力

旧 Stage 看到副本数、资源剩余、邻阶段重叠、有效候选比例，
却看不到每个阶段的副本具体位置和总有效算力。
若两个内存/存储相同的阶段在快慢节点间互换，资源剩余、节点占用、
阶段副本数和重叠均可能不变；旧完整 Stage 观测可以逐项完全相同。
`test_old_stage_observation_alias_is_fixed` 复现了这个问题。

新版本加入所有阶段的动态清单：有效标记、副本数、总有效算力、
预计工作率、计算代理成本、内存与存储占用；每个节点附上完整阶段部署位图。
因此能区分“有十个副本”和“有足够的有效计算能力”。

### 2. Placement 缺少目标阶段和完整链部署信息

旧 Placement 主要依赖服务/阶段归一化编号、全服务静态成本及上一阶段亲和位，
没有明确目标阶段成本、目标阶段已经在哪些节点部署、下一阶段部署等信息。
动作 mask 虽用于概率归一化，但没有作为网络输入编码完整部署事实。

新版本加入本阶段计算量、输入输出数据、资源需求、需求比例与已有算力；
节点输入增加本阶段/下游部署、计算代理增益、路由入口代价和有效算力。
这些是在线观测，不把节点排序或最优候选直接作为 Actor 动作。

### 3. 剩余全局预算隐藏在执行器中

`budget-continuous-v2` 中 Count 输出连续资源比例，后续 Stage/Placement
受该预算约束。然而旧状态没有完整表达预算目标、预算已消耗与剩余量。
新增六项预算观测；Count 决策前这些量不含将要采样的预算，
Stage/Placement 只读取此前已经采样和执行的预算/部署。
不存在用自身动作或未来窗口结果构造决策前基线。

### 4. Stage 条件 Critic 没有读取 Stage 特征

旧 `GraphAttentionActorCritic` 的 Stage Actor 编码了阶段特征，
Critic 却只使用全局嵌入和平均节点图嵌入，没有读取阶段编码。
在 actor backbone detach 时，Critic 又无法训练自己的输入编码。
因此“有条件基线”不等于“对部署状态有充分条件信息”。

新 Critic 在保留 detached 图上下文的同时，直接读取决策前全局、
节点、阶段观测，并使用独立可训练的编码路径。
价值梯度不进入 Actor/图编码；测试验证 Stage 输入影响新 Critic，
却不影响旧 Critic，以及纯价值更新不改变 Actor 概率/权重/Adam 状态。

### 5. 价值学习节奏与更新权重

旧条件 Critic 跟随 Actor 的 2/3/3 epochs 和学习率；独立窗口 Critic
虽然另训8次，在当前 team-credit 模式中并不是这些动作的直接基线。
Stage/Placement 的数百个动作重复同一窗口回报；若所有样本权重相同，
副本较多的窗口会在价值拟合中获得更高权重。

新条件 Critic 使用独立学习率（默认继承 `--slow-critic-lr 5e-4`），
并在 Actor 更新后补充价值训练次数：默认 Count +6、Stage/Placement +5，
正常接受全部 Actor epochs 时各合计8次。
只训练 Critic，不产生额外 Actor 更新。Critic 各窗口等权，
Actor 仍保留完整复合部署动作的 score-function 相对权重，不按序列长度除去信用。
Actor 优势始终使用采样时记录的价值，不能使用本批拟合后的预测泄露动作标签。

新 Slow 每个 epoch 汇总全部样本梯度，minibatch 仅用于控制显存；
更新后测量全批 KL，超限恢复参数和 Adam 状态并回溯步长。
这解决批内覆盖与 KL 超调风险，不保证学习一定变快。

### 6. Stage 学习率原先被 Count 的退火条件牵连

旧控制器把 Count/Stage 放在同一组，用二者 KL 的最大值决定共同退火。
新版三头分别维护状态：Count 停滞或 KL 较大不会自动降低 Stage 学习率。
Actor 学习率控制也不再改动独立条件 Critic 的学习率。
旧模式保留原控制器语义；新模式保存/恢复三头控制状态。

## 物理特征的界限

Slow 原有需求观测为 `log1p(requests/min)/log1p(500)`，不是概率。
新成本特征从原始用户服务权重汇总真实需求比例，不把对数观测当概率。
数据仅来自当前已知场景、负载和部署，不读取未来请求轨迹。

计算代理为 `p_s^2 * arrival_rate * C_stage / pooled_effective_capacity`，
以及加入一个候选节点后的代理成本差。它参考共享资源负载效应，
但没有完整考虑跨服务竞争与共享 NIC，不是精确联合结算或严格下界。
链路特征同样为已知路由上的轻量预测；最终时延与资源可行性仍由公共 KKT 决定。
预测值只进入观测，不替换真实训练回报或报告中的物理时延。

## 奖励与尚未确认的原因

训练目标保持原样：平均/P95混合时延与资源等成本的真实窗口回报。
新版仍不使用稀疏副本消融收益作为 Actor 主目标。
统一窗口收益的高方差、随机训练与确定性评估的差异、best 选择不等于
固定基准最优、低抽样率偏差，仍需后续学习实验量化。
不能把本次状态/基线修复写成已经证明所有劣势都被消除。
尤其不能从短程运行无报错推断 Proposed 已优于 DARP/DPMS。

## 启用、兼容与日志

新版本标记：`--slow-deployment-context deployment-aware-v2`。
训练入口 auto 对新 `mec-latency200 + batch-credit-v1 + budget-continuous-v2`
运行启用新版；完整续训则继承 checkpoint 顶层标记，旧 checkpoint 默认为 legacy。
直接 API 中未解析的 auto 保守地按 legacy 处理，避免旧适配器/测试元数据
隐式切换网络结构。明确要求新版时必须使用完整标记。

新旧 Slow 输入维度与 Critic 参数不同，不能直接续旧 Slow 权重。
可以用 `--init-fast-checkpoint` 只继承兼容环境下的 Fast 权重和价值尺度，
Slow、优化器、更新计数和随机流全部从新运行开始；随后仍联合训练，
没有人为分阶段或冻结 Fast。初始化会检查关键环境配置、预算文档、
物理种子、Fast 类型与奖励尺度，不允许偷偷把旧环境模型当新环境模型。
此参数与完整 `--load-checkpoint` 互斥。

旧训练/评估、MSO 与检查点恢复路径保留。新 PPO 参数全部是可选项，
只在新版 Slow 启用；Fast 不启用这些额外 Critic 通道。

在原有 update/episode/rollout 日志外，新版增加：

- `logs/slow_deployment_metrics.csv`：每窗口一行，`stages_json` 记录每阶段
  副本节点、数量、有效算力、预计工作率和资源成本，避免逐动作写大量日志；
- `logs/slow_update_metrics.csv`：每头的样本覆盖、KL、接受步长、回溯次数、
  额外 Critic 步数、归一化价值尺度和更新前后 EV。

## 验证记录

- 全项目回归293项通过；之后补充 Stage 独立退火检查，最新 Slow/训练入口
  专项59项通过（其中新 Slow 审计测试10项）。
- 从备份提交加载原代码，与当前 legacy 模式在相同随机种子下比较，
  Slow 三头与 Fast 初始化权重及确定性部署逐项完全相同。
- 修正真实需求比例后的 CUDA 诊断完成一个完整60分钟 Episode：
  32节点、12000用户、6个部署窗口、36次抽样结算、一次联合更新，
  Fast/Count/Stage/Placement 接受4/2/3/3步，三头样本覆盖100%，
  无无效动作，额外 Critic 步数6/5/5，checkpoint 保存成功。
  这是6窗口诊断，不是正式12窗口/2 Episode 的学习质量认证。
- 该32节点 CUDA 检查点通过现有评估加载接口恢复到 CPU，
  新版本标签、双优化器参数组与价值归一化状态加载正常。
- 该诊断路径：`runs_tmp/slow_deployment_v2_final_cuda_smoke_20261013`。
  Slow 未训练，首轮平均时延约2.03秒；不得当作提升或收敛成绩。
- 首次12窗口诊断使用了尚未校正真实需求比例的特征，虽正常结束，
  仍作为开发过程产物保留，不用于最终功能/性能证明。

## 建议下一轮命令

保留用户目前的6抽样/100秒维持近似，不声称它等价完整逐秒模拟。
先做80 updates的联合测试，观察阶段算力分配、条件 Critic 和同负载时延，
再决定是否进入320 updates。此命令是新训练，不是从旧 Slow 续训：

```powershell
$runName = "slow_deployment_v2_fast_init_s6_u80_$(Get-Date -Format 'yyyyMMdd_HHmmss')"; python .\train_dual_ppo.py --seed 2026 --pressure-profile mec-latency200 --training-repair batch-credit-v1 --slow-deployment-context deployment-aware-v2 --training-design trajectory-simultaneous --train-mode joint --rollout-unit window --updates 80 --episode-minutes 60 --deployment-interval-minutes 10 --fast-windows-per-update 12 --slow-windows-per-update 12 --sampled-seconds-per-window 6 --sampling-max-interval-seconds 100 --slow-allocation-mode budget-continuous-v2 --replicas-per-stage 0 --reward-mode latency --reward-scale 10 --fast-policy-kind gat_node_scorer --fast-chain-context chain-kkt-v3 --fast-counterfactual-credit-coef 0 --fast-counterfactual-credit-final-coef 0 --slow-counterfactual-credit-coef 0 --slow-replica-ablation-samples 0 --fast-oracle-diagnostic-requests 0 --fast-k-epochs 4 --slow-k-epochs 2 --slow-stage-k-epochs 3 --slow-placement-k-epochs 3 --fast-minibatch-size 512 --eval-interval 0 --device cuda --init-fast-checkpoint .\runs\latency200_accel_s6_h100_u320_20261013_001751\checkpoints\best.pt --run-name $runName --progress-interval-seconds 10
```
