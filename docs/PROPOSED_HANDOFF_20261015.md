# Proposed 交接说明：参数、路径、现状与未解决问题

整理日期：2026-10-15（UTC+8）。本次快照时间：15:58；新版日志与latest模型读取到第46次更新。
训练记录可能继续变化，本文件不是实时状态。没有停止、重启或修改训练进程。

## 1. 首先需要知道的结论

用户认为本轮Slow优化没有达到要求，要求交接，不再由当前助手继续优化。
已确认整体训练前期更快，但**没有证明Slow独立学出了明显更优的部署策略**。
代码正确性测试通过，不等于学习有效、收敛或优于算法方案。
不要把日志中的 `slow_updated=1`、非零梯度或总体时延下降当作Slow改进成功的证据。

完整机器可读交接文件：`docs/PROPOSED_HANDOFF_20261015.json`。
它包含两轮运行的全部保存参数、Git状态、checkpoint元数据、分段统计及诊断结果。
如本说明的摘要与实际运行记录存在差异，以**对应checkpoint内嵌args和对应运行日志**为准。

## 2. 项目、Git与压缩包

项目根目录：`C:\Users\1\Desktop\work_edge_computing`。

| 项目 | 当前值 |
|---|---|
| GitHub仓库 | https://github.com/makaka233/work_edge_computing |
| 当前分支 | `codex/backup-latency200-before-credit-fix-20261012` |
| 当前HEAD | `123a0a3a43df300f7c0473dffa71c92a08f9653d` |
| HEAD说明 | 改进前备份，包含已版本化的DARP/DPMS贪心方案；此前已核实上传GitHub |
| 本轮joint-credit-v2实现 | **本地工作区未提交、未上传**；仅clone当前GitHub分支不会得到本轮实现 |
| 最新代码迁移包 | `dist/proposed_training_20261015_143344.zip` |
| 新包验证报告 | `dist/proposed_training_20261015_143344.verification.json` |
| 旧包 | `dist/proposed_training_20261014_113512.zip` 已按用户要求删除，旧验证报告保留 |

最新ZIP的SHA256：

```text
5052942193a0355fe67601cb2d2f583005d972ad672790130d29564a3700ab2d
```

ZIP包含当前Proposed训练源码、必要依赖、绘图/校准脚本和说明；不含MSO/DARP/HEFT/DPMS
算法实现，不含runs/results/checkpoint、第三方安装包或Git历史。本次后来创建的交接MD/JSON也不在该ZIP中。
若要实际复现或继续已有模型，需另外传递相应完整运行目录，不能只传代码ZIP。
环境、服务预算、共享NIC和KKT实现随代码包保留。

本轮主要未提交文件：

```text
README.md
train_dual_ppo.py
edge_drl/agents/drl.py
edge_drl/agents/slow_context.py
edge_drl/agents/stage_budget.py                 新增
edge_drl/models/ppo.py
edge_drl/models/slow_learning.py
edge_drl/models/prefix_credit.py               新增
edge_drl/models/guarded_chunks.py              新增
edge_drl/checkpoint_selection.py               新增
edge_drl/sampling.py                           新增
edge_drl/comparison/metrics.py
edge_drl/comparison/runner.py
edge_drl/env/environment.py                    仅新增真实采样事件计数
run_comparison.py
scripts/train_joint_credit_v2.ps1              新增
scripts/package_proposed.py
packaging/proposed/README.md
packaging/proposed/train_proposed.ps1
packaging/proposed/verify_bundle.py
tests/test_joint_credit_v2.py                  新增
docs/JOINT_CREDIT_V2.md                        新增
```

不要把未跟踪的runs_tmp、results、dist、缓存或快捷方式误当成新算法源码。
不要执行git reset/clean或覆盖工作树来读取旧版；使用git show或独立工作副本检查旧代码。

## 3. 两轮重点训练与模型位置

### 最新版：joint-credit-v2

```text
runs/joint_credit_v2_s6_u320_20261015_144010/
  metadata.json
  logs/training.csv
  logs/rollout_metrics.csv
  logs/episode_metrics.csv
  logs/slow_deployment_metrics.csv
  logs/slow_update_metrics_prefix_credit_v2.csv
  logs/checkpoint_probe_metrics.csv
  checkpoints/best.pt
  checkpoints/best_rolling.pt
  checkpoints/latest.pt
  checkpoints/snapshots/update_0020.pt
  checkpoints/snapshots/update_0040.pt
```

本次读取：training.csv共46次完整更新；latest.pt为第46次；best.pt为第40次。
本次没有观察到last.pt，不能把目标320更新当作已完成320更新。
该运行**完全从头初始化**，`init_fast_checkpoint=""`、`fast_initialization=null`，没有继承旧Fast。

best.pt记录的219.152 ms，是负载1.0、固定需求轨迹、10分钟、6个采样结算点的独立探针结果；
不是60分钟全量结果、不是五场景均值，也不是训练随机采样均值。

### 改进前完整训练：conditional-replay-v1

```text
runs/slow_repair_scratch_s6_u320_20261014_162352/
  metadata.json
  logs/training.csv
  logs/rollout_metrics.csv
  logs/episode_metrics.csv
  logs/slow_deployment_metrics.csv
  logs/slow_update_metrics_conditional_replay_v1.csv
  checkpoints/best.pt                          第150次更新
  checkpoints/latest.pt                        第320次更新
  checkpoints/last.pt                          第320次更新
  checkpoints/best_rolling.pt
  checkpoints/snapshots/update_0040.pt
  checkpoints/snapshots/update_XXXX.pt          每20次保存
```

旧轮完成320次更新、640个完整Episode。metadata最后一次续训命令的updates=237只是追加数量，
不是整轮训练总数。旧best.pt按历史负载校正滚动训练分数选择，其0.146442 s分数不是独立实测时延。
旧best和新best的选择标准不同，不可直接比较这两个selection分数。

另一个更早的诊断对象：`runs/slow_fast_v2_joint_s6_u320_20261014_005916/`，
有129次完整更新记录；不是本次主要旧版基准。

## 4. 最新版训练参数：实际运行值

### 场景、时间与抽样

| 参数 | 值 |
|---|---|
| seed / 有效physical seed | 2026 / 2026（保存physical_seed=null时回落到seed） |
| pressure-profile | `mec-latency200` |
| 用户 / 节点 / 服务 / 有效阶段总数 | 12000 / 32 / 10 / 29 |
| edge_node_profile / service_workload_profile | `jetson-ai-edge` / `edge-ai-latency200` |
| network_constraint_mode / allocator | `shared-nic` / `shared-nic-certified-v4` |
| service_calibration_mode | `audited-v1`；完整预算文档内嵌metadata/checkpoint args |
| 接入 | 每请求有效300 Mbps、radio_rtt_ms=10；不是共享空口仿真吞吐保证 |
| topology_k_nearest | 6 |
| compute/data/node capacity scales | 1.0 / 1.0 / 1.0 |
| wired_link_bandwidth_scale | 0.8 |
| service_resource_fraction | 0.60，部署RAM/SSD硬预算，不是CPU利用率上限 |
| deadline_scale | 2.75，本轮没有调整 |
| arrival_profile | stationary |
| demand_sampling_mode / scenario_schedule | episode / stream，每Episode刷新需求，物理网络固定 |
| demand_scenario_pool_size | 32为保存参数；stream模式不是固定池轮转实验 |
| active_user_ratio / request rate per active user | 0.2 / 1.75每分钟；实际需求率经环境模型生成 |
| 负载分层 | [0.65,0.80]、[0.80,1.00]、[1.00,1.20]、[1.20,1.45] |
| 分层概率 | 0.20、0.45、0.25、0.10；distribution-random |
| episode_minutes / deployment_interval_minutes | 60 / 10 |
| rollout_unit | window，一次rollout对应一个10分钟部署窗口 |
| Fast/Slow windows per update | 12 / 12，共享同期数据，每更新两个完整Episode |
| sampled_seconds_per_window / hold cap | 6 / 100秒 |
| temporal_sampling_mode / load update | stratified-jitter / interval-average |
| 每update实际结算 / 代表仿真秒数 | 72 / 7200 |
| train_mode / training_design / schedule | joint / trajectory-simultaneous / simultaneous |
| updates目标 / device | 320 / cuda |

保存参数fast_updates_per_cycle=3是兼容字段；本轮simultaneous不应解释为每三次Fast才更新一次Slow。
保存参数replicas_per_stage=0由入口解析为节点数上限，最多同一阶段每节点一个副本；不是禁用副本。

### 学习与优化

| 参数 | 值 |
|---|---|
| proposed_design | joint-credit-v2，显式启用；auto新运行仍可落入legacy |
| training_repair / deployment_context | batch-credit-v1 / deployment-aware-v2 |
| Slow allocation / learning repair | stage-budget-v3 / prefix-credit-v2 |
| Fast policy / chain context | gat_node_scorer / chain-kkt-v3 |
| Count / Stage / Placement / Fast学习率 | 1e-4 / 2e-4 / 1e-4 / 2e-4 |
| Count / Stage / Placement / Fast epochs | 2 / 3 / 3 / 4 |
| Slow / Fast minibatch | 2048 / 512 |
| Count / Stage / Placement / Fast实际KL阈值 | 0.015 / 0.015 / 0.015 / 0.015 |
| Fast独立Critic / update chunks | true / 4，每epoch最多4个Adam步骤，每更新最多16步 |
| Fast负载平衡 / 完整批KL检查 | true / true |
| Critic LR / Slow价值训练epochs / value coef | 5e-4 / 8 / 0.5 |
| Slow条件价值回放 | 96窗口、每窗口最多8个前缀状态，衰减0.9；仅价值标签回放 |
| Slow entropy / Count entropy | 0.001 / 继承0.001（保存值null） |
| Placement entropy | 0.005保持64更新，再64更新降至0.0035；自适应目标1.1 |
| Fast entropy | 0.001；归一化目标0.18，最大0.02，调整率0.002 |
| Slow LR控制 | patience40、factor0.7、最多降1次；低KL保护、低熵例外 |
| Count/Placement最低LR | 5e-5 / 3.75e-5 |
| oracle / BC请求 / counterfactual / replica ablation | 全部0，未使用专家初始化或模仿训练 |
| Fast reservation_microbatch_size | 16，仍保留严格跨请求KKT记账，不等于16请求共享一份静态账本 |

注意：args中slow_target_kl=0.03是通用兼容参数，当前Stage构造使用count_target_kl=0.015，
不要仅看参数名就认为Stage实际阈值为0.03。旧窗口Critic在prefix-credit-v2中不训练，
相关遗留日志可为0；真正要看三个条件Critic的prequential指标。

### 奖励与模型保存

- reward_mode=latency，reward_scale=10。新Slow窗口目标为平均真实时延的负值乘10。
- 新版Slow的P95权重、RAM/SSD附加价格、违约附加价格、共置奖励和共享附加回报默认置0。
  硬预算、时限值和实际违约日志没有因此移除。旧版P95=0.35、RAM=0.03、SSD=0.01、违约=0.1。
- Fast仍是含跨请求资源外部性的批内信用；`avg_reward`不一定等于`avg_train_reward`，
  也不能直接与Slow窗口回报相加作为同尺度总reward。
- `avg_train_reward`本轮为负平均时延（秒）；`slow_window_return`是优化尺度窗口回报。
  新旧Slow目标不同，不能直接用两条slowR的数值证明新版更好。
- 原周期评估eval_interval=0，但checkpoint探针仍在首轮、每10轮及末轮执行。
- best.pt：更新后独立名义负载探针最优；best_rolling.pt：近期训练诊断分数最优，
  最近最多80更新做负载/需求计算量修正；latest.pt每次更新，last.pt在正常结束保存。
- checkpoint_interval=20；best_checkpoint_window=10；原日志目录保留。

## 5. 当前算法做了什么，没做到什么

Slow仍是一个智能体，Count/Stage/Placement是内部动作头。
先由Placement完成每个有效阶段至少一个副本的硬覆盖；覆盖顺序按装箱难度确定，覆盖节点是学习动作。
Stage从未处理且可追加的阶段中选择顺序；Count对该阶段采样连续Beta动作a，
若可追加上限为M，额度=min(floor(a*(M+1)),M)；Placement逐副本选择全部可行节点。
每阶段最多一次额度决策，没有固定部署模板、K条预选路径或候选节点截断。
Coverage在极端碎片化时仍可能失败；当前测试不构成任意预算下可执行性的证明。

Fast仍按请求及其链阶段逐步选择节点并维护KKT账本，不是全局路径规划器直接输出最优解。
新Critic增加原始全局/节点负载输入并隔离价值梯度；Actor未替换成算法调度器。

Slow每窗口仍只有一份真实终端部署回报，不因数百动作而变成数百独立回报样本。
条件Critic预测真实窗口回报；前缀优势只沿本窗口同一动作头的因果顺序构造。
前缀修正使用过去批次的可靠度r：lambda=1-0.05r；
`A=(R-cV)+r*(GAE(R,V,lambda)-(R-V))`，其中c是历史基线倍率。
lambda=1时回到MC；lambda<1存在bootstrap偏差—方差权衡，不是精确的逐动作反事实贡献。
竞争观测中的计算代理只是近似特征，不是实际联合KKT求解、不作为奖励标签。

## 6. 已确认的同进度结果与问题

下面比较主旧版与新版；双方均从头训练，前42次的需求种子、负载、计算量和数据量匹配。
不是只比较一个低负载update。采样数和物理场景相同，但算法结构与目标不同。

| 更新区间 | 旧版平均时延ms | 新版平均时延ms |
|---|---:|---:|
| 1–10 | 941.14 | 761.32 |
| 11–20 | 684.20 | 457.94 |
| 21–30 | 550.70 | 369.87 |
| 31–40 | 338.08 | 301.15 |
| 33–42 | 294.41 | 262.36 |

这些是update均值再取均值，不是合并所有请求重新加权的结果。
前42更新，新版37次低于旧版；零无效动作、Slow更新42次；Fast672优化步骤，旧版168步；
新版Fast完整覆盖所有在场负载组，未发生KL早停/回退，最大分组KL约0.00728<0.015。
总体改善存在，但不能据此归因Slow。

最近10次（33–42）Slow诊断：

| 动作头 | 条件Critic更新前EV | 前缀lambda | 相对MC优势标准差 | 其他 |
|---|---:|---:|---:|---|
| Count | -5.31 | 0.9980 | 0.9986 | 阶段额度仍需检验是否按需求有效分配 |
| Stage | -3.48 | 0.9959 | 0.9953 | 归一化熵约0.9565 |
| Placement | -1.36 | 0.9967 | 0.9749 | 归一化熵约0.9994，KL约2.9e-6 |

负EV说明回报变化解释仍差，不等同于所有缩放基线都必然无效；需结合MSE、协方差、
实际方差缩减和部署收益判断。当前前缀修正很保守，没有证明解决了逐动作信用问题。
Placement有反向传播但偏好变化很弱。记录的奖励梯度远大于熵梯度，不能简单归因熵系数太大。

## 7. 第40次checkpoint的交叉诊断

此诊断在前一轮助手工具中执行，结果原先只有stdout，本说明与JSON现记录其数值；
不是已完成的正式五场景实验。模型均为两目录的snapshots/update_0040.pt。
使用CPU单线程、checkpoint_probe、seed2026、负载1.0、10分钟、6采样、hold100秒，
加权请求数35600；四组同一轨迹：

```text
e132d1745aecd45f508e2c4a9c527210846c8fbf2872d941a9f5b28c9168e541
```

| Slow | Fast | 平均时延ms |
|---|---|---:|
| 旧 | 旧 | 241.7906 |
| 旧 | 新 | 224.0427 |
| 新 | 旧 | 252.1459 |
| 新 | 新 | 219.1521 |

固定新版Fast时，新Slow比旧Slow好约4.89ms（2.18%）；固定旧Fast时新Slow更差。
证据支持Fast收益和组合适配，不足以证明Slow独立改善或全部场景更优。
当前不能把Slow问题宣布解决，也不能断言继续训练一定解决。

## 8. 完整对比实验位置与可比性边界

最近已完成五场景对比：

```text
results/comparison/five_scenes_greedy_v2_s6_seed2026_20261015_121351/
  metadata.json
  raw/
  summary/
  diagnostics/
  plots/
```

方案为Proposed/DARP/HEFT/DPMS/MSO，共28参数点×5方案=140行，seed2026，
每窗口6采样、60分钟评估，DARP/DPMS实现版本为demand-greedy-v2，旧实现legacy-v1仍可选。
这轮Proposed用的是**旧训练第150次best.pt**，不是最新joint-credit-v2。
MSO使用 `runs/mso_latency200_sync_s6_u80_20261013_164520/checkpoints/best.pt`。
五场景为request_load、compute_capacity、wired_bandwidth、intermediate_data、stage_heterogeneity。
历史各点均值再平均：Proposed203.836ms、DARP202.549、DPMS216.788、HEFT325.148、MSO311.766；
这是28点等权汇总，不能称作现实流量加权总体均值或多seed显著优势。

更早对比：`results/comparison/comparison_five_scenes_best_u150_s6_seed2026_20261015_091357/`。
不要混淆不同DARP/DPMS实现版本。不得通过修改汇总结果、隐藏失败点或不对等资源限制制造排名。

新版comparison的sampling_protocol=auto会针对joint-credit-v2选择training-equivalent-v2，
统一分层抖动与维持上限；旧模型默认legacy-left-hold。历史metadata虽然写过
training_equivalent_weighted_settlement，不代表它与新抖动机制数值相同。
新版五场景尚未完成，不可将旧五场景均值和新10分钟探针直接比较。

## 9. 代码定位：建议从哪里读

| 文件 | 责任 |
|---|---|
| train_dual_ppo.py | 参数/profile解析；apply_proposed_design；rollout；更新循环；save/load_checkpoint；初始化与日志 |
| scripts/train_joint_credit_v2.ps1 | 当前仓库启动封装，默认Fast Actor迁移，可-FromScratch |
| edge_drl/agents/drl.py | Slow/Fast类、观测、窗口回报、逐请求KKT特征与总智能体接口 |
| edge_drl/agents/stage_budget.py | plan_stage_budgets：覆盖、Stage顺序、Beta额度及Placement |
| edge_drl/agents/slow_context.py | 部分部署、候选代价及竞争近似观测 |
| edge_drl/models/ppo.py | Actor/Critic架构、PPO、归一化、prefix优势接入 |
| edge_drl/models/slow_learning.py | 条件价值回放、过去基线倍率、可靠度与Slow受控更新 |
| edge_drl/models/prefix_credit.py | prefix_advantages：窗口内终端回报GAE |
| edge_drl/models/guarded_chunks.py | Fast分组混合多步骤、完整epoch的KL拒绝及Adam回退 |
| edge_drl/checkpoint_selection.py | recent_workload_score和隔离checkpoint_probe |
| edge_drl/sampling.py | 采样间隔计算 |
| edge_drl/env/environment.py | 场景执行、资源结算和真实/加权事件统计 |
| edge_drl/env/latency200.py | 节点和10类服务标称数值 |
| edge_drl/env/calibration.py | 预算文档与审计；工程估计并非硬件测量 |
| edge_drl/env/scenario.py | 实例化、扰动、拓扑和需求生成 |
| edge_drl/allocators/kkt.py、nic_kkt.py | 计算与共享NIC连续资源求解；不可降低精度或放松约束掩盖故障 |
| edge_drl/comparison/checkpoint.py | 按checkpoint参数重建Proposed，不能用默认args随意加载 |
| edge_drl/comparison/metrics.py、runner.py | 公共评估、版本化采样、场景对比 |
| tests/test_joint_credit_v2.py | 新机制、KL回退、隔离、入口联合更新与续训测试 |
| scripts/plot_rollout_training.py | 可复用rollout级训练绘图 |
| scripts/plot_training_run.py | update级绘图 |

设计文档：`docs/JOINT_CREDIT_V2.md`。其他历史说明在docs下，可能反映较早状态，
尤其旧网络说明不得替代当前shared-nic配置和可执行代码。
内嵌服务预算为estimated，参考计算能力72 Gcycle/s，C=72×compute_ms/1000 Gcycle；
不能把它说成真实设备对每个工作负载的实测。逐阶段上传/输出、计算、RAM、SSD完整值见JSON的
`runs.new.args.service_calibration_document`，无需从聊天片段重建。

## 10. 运行、续训与验证注意事项

原项目从头运行命令（仅记录，不自动执行）：

```powershell
.\scripts\train_joint_credit_v2.ps1 -Updates 320 -SampledSeconds 6 -Device cuda -FromScratch
```

省略-FromScratch会仅迁移指定旧Fast Actor，随后双方联合训练，不是本次正在讨论的完全从头运行。
参数展开及全部默认值已在本交接JSON保存。默认preset是显式启用而不是全项目默认替换。

ZIP解压后进入proposed_training目录，代码包的从头训练命令：

```powershell
python -m pip install -r requirements.txt
.\train_proposed.ps1 -Updates 320 -Device cuda
```

代码包不含checkpoint；需要迁移训练记录时复制整个相应运行目录。
旧budget-continuous-v2不能完整续训为新stage-budget-v3；仅Fast Actor初始化是显式迁移方式。
同设计断点续训需load-checkpoint、原run-name、append-log，updates是**追加数**。
若训练已完成N次且目标320，追加320-N，N应取checkpoint已提交更新数而非未提交rollout日志。
本交接未提供会启动重复进程的续训命令；操作前应由接手者核对进程与实际checkpoint。

本轮基础验证：346项pytest通过；32节点真实规模CUDA单次联合更新正常，
开发短程记录在 `runs_tmp/joint_credit_v2_final_actor_init_cuda_20261015/`。
该短程继承旧Fast Actor，与当前完全从头训练并非同初始化条件。
这一次约103秒不能外推固定320次耗时，也不能证明学习有效。

压缩包独立解压验证位置：
`runs_tmp/bundle_verification_20261015_143344/proposed_training/`。
验证了独立导入、物理签名、NIC证书、小场景训练/续训/初始化/推理/绘图。
新旧代码包物理签名均为：

```text
d4f740a29d932442f7d9a830e985a5bada9783805a913f4b36b946255cb43df0
```

## 11. 用户约束与接手优先事项

用户要正统Slow PPO+Fast PPO，不接受固定模板、预先固定候选路径、偷偷替换为专家算法，
不希望再分阶段训练；训练速度重要。不要反复创建新分支。
保留runs/results/final、所有训练/评估记录、checkpoint、图片，不得随意清理。
对环境、服务预算、网络或时限的改变应明确告知并先确认；本轮joint-credit-v2未调整这些数值。

接手优先验证：在相同Fast和相同请求下，Slow是否优于其随机/初始部署，
并将Count、Stage、Placement的实际部署贡献与Actor变化分开；不要先增加一轮大规模训练。
重点审计12个窗口回报如何支持数百部署动作、Critic目标在Fast变化下的非平稳性、
基线倍率/前缀可靠度是否真正降低方差、Placement为何接近均匀。
上述是待验证方向，不是已确认唯一根因或保证有效的修复承诺。

本次交接只新增说明和参数快照，不更改算法、不提交Git、不清理产物、不操作训练进程。
