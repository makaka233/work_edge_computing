# Proposed Slow 整数额度与组件评估：joint-credit-v3

本版本在备份 `dbd1e80` 上显式增加 `--proposed-design joint-credit-v3`。保留 Slow PPO + Fast PPO 同时联合更新。场景、服务资源、网络、时限、共享NIC结算与DARP实现保持原设置。当前实现对应审计方案的第一批：正式组件评估、版本化解码和整数Count。

## Slow动作与观测

`stage-budget-v4`先由Placement选择每阶段的硬覆盖节点，再由Stage选择未处理且可追加的阶段。Count直接对 `q=0,...,M` 采样Categorical动作；M为当前RAM/SSD可行节点数及物理副本限制的较小值。共享额度评分器为每一个可行整数评分，完整范围均可到达；没有部署模板、预选路径或专家标签。

PPO记录和执行同一个整数及其原始离散log-prob。确定性评估选择该离散分布的众数，记为 `slow_quota_decode=categorical-mode`。Placement仍逐副本从全部可行节点学习选择。

新增5项全局观测：本阶段最大可追加额度、已选择额度、当前剩余额度、Count决策标记、额度Placement标记。额度均按物理节点数归一化。Count决策前不包含自身将要选择的额度；Placement读取此前已选择额度。预算汇总使用实际整数额度比例与实际RAM/SSD消耗，避免在后续观测保留未执行的Beta小数信息。

Stage仍每阶段至多一次额度决策。尾部再分配和新Critic属于后续增量实验。另提供独立、默认关闭的[链路观测实验](SLOW_CHAIN_OBSERVATIONS_20261015.md)，通过显式上下文版本选择，不改变已启动的原观测实验。

## 旧Beta模型的显式解码

`stage-budget-v3`保留原来的 `mean` 解码；可在新训练中显式设置 `--slow-quota-decode integer-mode`，或在组件评估中显式覆盖。整数众数由Beta在各额度区间的CDF质量确定，返回对应区间内部的连续动作；连续PPO训练仍使用其连续动作密度。

该选项只控制确定性解码，不改变随机训练采样。旧checkpoint缺少解码字段时解释为mean。旧设计的默认结构、初始化顺序、权重和推理路径保留。完整续训检查分配模式与解码规则，避免沿用不一致的best选择历史。

## checkpoint与续训

- 新checkpoint保存 `stage-budget-v4`、解码版本和完整args；自己的检查点支持自动解析与同目录续训。
- 旧v2不能完整续训为v3：Count动作头和Slow观测维度不同。
- `scripts/train_joint_credit_v3.ps1`默认从固定的v2第60次快照仅迁移Fast Actor，Slow、Fast Critic和优化器从头开始；随后两个PPO都更新。`-FromScratch`完全随机初始化。
- `scripts/train_joint_credit_v2.ps1`默认仍为v2。

```powershell
.\scripts\train_joint_credit_v3.ps1 -Updates 12 -Device cuda
# 完全从头：
.\scripts\train_joint_credit_v3.ps1 -Updates 12 -Device cuda -FromScratch
```

## 固定Fast的正式Slow组件评估

```powershell
python .\evaluate_slow.py --fast-checkpoint .\runs\joint_credit_v2_s6_u320_20261015_144010\checkpoints\snapshots\update_0060.pt --slow-checkpoints <新模型路径> --seeds 2030,2031,2032 --loads 0.7,1.0,1.35 --episode-minutes 60 --device cpu
```

所有方案共享固定Fast权重、完整外生请求轨迹和分层抖动采样。各方案分别推进自己的负载与上窗口反馈。默认6采样/窗口、100秒维持上限；它不代表逐秒仿真认证。

对旧v2做解码消融：

```powershell
python .\evaluate_slow.py --fast-checkpoint <v2模型路径> --quota-decoders checkpoint integer-mode --seeds 2030,2031,2032 --loads 0.7,1.0,1.35 --episode-minutes 60
```

输出到新的 `results/slow_components/` 子目录：metadata、Fast和checkpoint指纹、轨迹清单、raw.csv、paired.csv、逐服务时延、失败记录及summary.json。每点断言请求权重和采样数量一致，并核对Fast权重未改变。任一失败不填零、不从总均值中隐去；相应候选的完整均值标为不可用，CLI以非零退出。

主验收是同一个Fast下Slow PPO与DARP v2的真实时延差，随后再做联合训练模型及五场景验证。随机初始策略、非零梯度、熵或测试通过均不构成优于DARP的证明。实现和验证结果应分别报告。

Proposed最小导出包仅包含训练依赖；DARP组件评估入口使用完整仓库，不把DARP实现加入Proposed训练依赖。
