# Monolithic 窗口负载与按需抽样修复

修复前代码已备份至 GitHub 提交 `688ec109f3156f27d9870ee9df60b1c7c2191e44`。
本次仅修改 Monolithic 训练入口，并新增其专用训练环境及回归测试。
不修改 Proposed、公共物理结算、PPO 更新器或正式对比评估的不可变回放流程。
旧训练目录、日志、图片和模型均保留。

## 修复原因

旧入口按窗口负载预生成整个 episode 的请求，但只在创建环境时写入第一个
窗口的 `demand_load_multiplier` 和 `demand_load_group`。后续请求负载发生变化，
Slow 的预计需求观测以及 Fast 的学习分组却仍使用第一窗口的配置。
例如更新内实际覆盖四组负载时，Fast 可能只记录一至两组。
这不代表其他请求完全没有训练，而是观测和分组与实际负载不一致。

此外，旧入口即使仅抽样每窗口六秒，也先生成并保存 3600 秒请求。
Proposed 按需生成请求，不推进未观察秒的请求随机流。
因此仅保持同一个种子不能保证二者抽到同样的请求。

## 新流程

1. 沿用相同的主种子、物理种子、需求种子、环境种子及窗口负载抽取方式。
2. 每个窗口开始前，同时写入负载倍数与组号。负载改变时，按照 Proposed
   相同的规则重生成边界批次，包括相同的 `np.isclose` 判断。
3. `MonolithicTrainingEnv` 直接继承公共环境的请求生成器、时间推进和
   `sample_interval_requests`，不再预生成整个 episode 的轨迹。
4. 从原分阶段服务目录生成任务，再将每个任务计算量求和成一个阶段，
   阶段间传输量置零。保留请求编号、用户、接入节点、上传量、时限与到达时间。
   不直接从折叠目录生成请求，避免改变随机数消耗及任务总计算量的分布。
5. 执行视图仍为完整服务：计算量、常驻资源和上下文资源分别按阶段求和。
   物理网络、共享 NIC 约束、服务资源预算、奖励和 PPO 参数不因此改变。

相同配置和种子下，基准场景的抽样请求流应逐请求一致；调度、部署和结算
时延不必一致，因为两种方案具有不同的执行结构和学习策略。

`request_load` 的非基准 `scenario-value` 作用于实际窗口负载，同时写入日志；
基准 `scenario-value=1.0` 不改变负载分布。

## 速度与抽样范围

60 分钟 episode、10 分钟部署窗口、每窗口六个样本、100 秒代表间隔，
仍是 36 次结算，而不是 3600 次结算。继承 Proposed 原有的边界预取和
jitter 重采样，因此会生成少量随后被替换的批次，不代表每次生成均参与结算。
这些批次保留是为了严格同步随机流，不宣称只生成六个批次。

`training_manifest.json` 记录 `request_generator=proposed-on-demand-v1`、
`trace_pre_generated=false`、生成批次数、实际生成请求数及结算抽样次数。
不再为不存在的完整训练轨迹记录 `trace_hash` 或完整轨迹请求数。
`generated_request_events` 是实际创建的对象数；日志 `requests` 是按代表
间隔加权的估计请求数，二者不能混用。

本次消除预生成开销，但不保证整个 update 的固定加速倍数，也不证明
六秒抽样与逐秒完整模拟等价，更不保证最终时延或收敛结果。

## 从头训练与断点重续

停止的 `presync` 模型不用于此次修复后的续训。旧模型仍可在原有评估入口
加载，但新训练入口拒绝缺少或不匹配生成器版本的旧训练检查点，以免将
两个不同训练过程混合在一起。新模型保存生成器版本和已解析的网络、
校准、时间抽样配置，最新模型仍可在 update 边界断点重续。

可先执行 80 updates 测试；Proposed 的 `--base-checkpoint` 仅作为配置来源：

```powershell
$runName = "mso_latency200_sync_s6_u80_$(Get-Date -Format 'yyyyMMdd_HHmmss')"; python .\train_mso.py --base-checkpoint ".\runs\latency200_accel_s6_h100_u320_20261013_001751\checkpoints\best.pt" --updates 80 --episode-minutes 60 --episodes-per-update 2 --sampled-seconds-per-window 6 --fast-reservation-microbatch-size 1 --scenario-family request_load --scenario-value 1.0 --seed 2026 --device cuda --run-root ".\runs" --run-name $runName --progress-interval-seconds 10
```

## 验证覆盖

- 两窗口不同负载、完整逐秒/六秒/四十秒抽样、jitter/left-hold、stationary/daily：
  逐请求元数据、上传量、总计算量及两种随机流状态一致。
- 重置与窗口边界验证；组号变化而倍数不变时不额外消耗随机数。
- 小规模真实训练入口验证四组 Fast 学习信号、完整样本覆盖、检查点保存及续训；
  拒绝旧生成器模型直接续训。
- 32 节点、12000 用户、10 服务 CUDA 验证：首个抽样批次与 Proposed 逐请求一致；
  完成一个 10 分钟窗口的六次结算和一次 Fast/Slow 更新，无无效动作。
  Fast 完成四个 epoch，Count/Stage/Placement 分别完成 2/3/3 次优化。
  此项只检查单窗口调用，不是正式的 12 窗口完整训练 update。

这些是正确性和兼容性测试，不是新版本的大规模收敛实验。
