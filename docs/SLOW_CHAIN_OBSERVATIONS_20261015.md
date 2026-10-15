# Slow 链路观测增量实验

本选项为 `joint-credit-v3` 的独立观测消融：显式设置
`--slow-deployment-context deployment-chain-v3`。默认仍为 `deployment-aware-v2`，已启动的整数Count实验使用原观测。

## 依据和新增输入

旧v2第100次快照在固定第60次Fast、需求种子2030/2031、名义负载、60分钟的配对检查中：原解码188.8125 ms，整数众数188.2030 ms，DARP为183.5809 ms。原解码在种子2030的计算时延已接近DARP（142.2531 vs 142.3379 ms），但链路时延仍高（18.5893 vs 15.2017 ms）；2031也存在链路差距。它支持实验链路观测，不构成新增观测有效的结论。

节点在原有全部输入后增加8列：

1. 从已部署上游副本到候选节点的容量加权传输代价；首阶段使用用户入口分布。
2. 从候选节点到已部署下游副本的容量加权传输代价；尚无下游时为零。
3. 从用户入口到候选节点的传输代价代理。
4. 保留正负号的计算代理增益，含同串行阶段其他服务的计算竞争外部性。
5. 由历史链路EWMA负载估计的节点发送NIC余量比例。
6. 对应的接收NIC余量比例。
7. 放置该副本后，其他阶段在本节点失去物理可行位置的比例。
8. 当前串行阶段的计算工作压力代理。

Stage增加4列：可行候选的最小/平均上下游传输代价、最大有符号计算代理增益、平均其他阶段机会损失。

代理只作为决策前输入，具有近似误差；并非当前Fast的真实路由或精确KKT解。Actor仍对所有物理可行候选输出概率，训练回报仍来自真实窗口结算。当前请求批次及未来结果不进入新增输入。

## 明确初始化和兼容性

新增 `--init-slow-checkpoint <整数Slow checkpoint>`：只迁移Count、Stage、Placement三个Actor。新节点和Stage输入列的权重置零，原Actor参数完整复制，故迁移时三个头的概率与来源模型一致（允许浮点容差）。新列可通过PPO梯度学习；Critic、优化器、价值回放、运行计数均从头建立。

该初始化只支持相同物理配置、整数额度模式及副本限制。允许原观测→链路观测或同观测初始化，不允许把已学习的链路列丢弃。它不是完整续训。两种观测的完整续训仍严格校验架构；自动解析读取checkpoint保存的上下文。

为比较不同输入维度的迁移实验，显式Slow Actor初始化后将动作采样Torch种子设为 `seed + 7000019`，写入 `slow_initialization`。需求和环境继续使用独立训练随机流。迁移不导入DARP动作、专家标签或模仿损失。

受控比较应同时建立原观测与链路观测两个新运行，迁移同一个Slow来源和同一个Fast Actor，匹配训练种子、窗口数和超参数。不要把Critic未重置的原续训直接称作这一迁移实验的同初始化对照。

```powershell
# 两个运行使用同一个不可变的整数模型快照作为Slow来源。
.\scripts\train_joint_credit_v3.ps1 -Updates 12 -Device cpu -RunName chain_control -InitSlowCheckpoint <整数快照> -SlowDeploymentContext deployment-aware-v2
.\scripts\train_joint_credit_v3.ps1 -Updates 12 -Device cpu -RunName chain_candidate -InitSlowCheckpoint <同一整数快照> -SlowDeploymentContext deployment-chain-v3
```

两个命令默认都从固定旧v2第60次快照迁移Fast Actor。可用 `-InitFastCheckpoint` 同时指定另一个共同来源。

## 观测诊断和收益验收

组件评估新增 `deployment_windows.csv` 和 `quota_actions.csv`，记录每个窗口的部署、副本额度、剩余资源以及结束时分别仍可追加的位置数。后者不代表所有位置可同时追加，也不证明追加必然降低真实时延。

检查三个Actor概率在迁移点一致、额外列可学习、部署覆盖/RAM/SSD约束、真实回报更新、两种上下文自动续训和组件评估。性能判断以固定Fast组件配对及各自联合训练后的配对为准。此文记录实验能力，尚未声称链路观测提高实测性能。
