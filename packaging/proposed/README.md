# Proposed 独立训练迁移包

本包保留当前项目 Proposed 的原始训练代码与功能：Slow PPO（Count、Stage、
Placement）+ Fast PPO、串行服务链、计算KKT与共享NIC连续带宽分配。
核心 Python 文件未经裁剪或重写。包含最新 shared-nic-certified-v4 严格精度修复，
不包含已撤回的 guarded-minibatch-v2／独立 Fast Critic 改动。

## 安装与独立运行

解压后进入 **proposed_training** 文件夹，所有命令都在该目录执行。
推荐 Python 3.11（打包验证环境为3.11.7）。

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python .\verify_bundle.py
```

`requirements.txt` 使用打包机器已验证的版本；Torch写主版本号，已有匹配CUDA
构建可保留。GPU运行需要可用的CUDA版PyTorch与兼容驱动；如果CUDA不可用，
使用`--device cpu`，不能把GPU依赖缺失当作代码故障。
Python、第三方库和CUDA驱动不包含在这个**代码**压缩包中。
不同库版本/硬件不保证数值逐位或训练曲线一致。

`verify_bundle.py` 在本包中创建`_verification_runs`，验证完整60分钟episode、
联合更新、同目录续训、模型加载、Fast-only初始化、绘图及校准工具。
采用8节点、2服务的低成本配置，**不是性能/收敛证明**。
要重复运行完整验证，请使用新的解压目录，旧验证产物不会自动删除。
只检查哈希、导入、32节点物理场景签名及NIC精度：

```powershell
python .\verify_bundle.py --imports-only
```

## 当前配置：从头联合训练320 updates

```powershell
.\train_proposed.ps1 -Updates 320 -Device cuda
```

脚本在本包目录启动原`train_dual_ppo.py`，使用：

- seed2026，mec-latency200，12000用户、32节点、10服务；
- 60分钟episode、10分钟部署窗口；每update十二窗口（两个完整episode）；
- 每窗口六个抽样秒，维持上限100秒，stratified-jitter；
- batch-credit-v1、deployment-aware-v2、budget-continuous-v2；
- Slow/Fast始终joint，Count/Stage/Placement/Fast分别2/3/3/4 epochs；
- masked GAT Fast，chain-kkt-v3，无oracle／counterfactual初始化；
- shared-nic、audited-v1服务预算、300Mbps接入+10ms，reward-scale10；
- 不启用周期性评估（eval-interval0），保存原有必要检查点和分级日志；
- 数值库线程限制为1。训练时间与收敛情况不能由迁移包保证。

脚本支持`-RunName 自定义名称`；`-Device cpu`可在无GPU环境运行。
原训练入口的其他模式、参数、内置评估和可选BC功能均保留：

```powershell
python .\train_dual_ppo.py --help
```

## 迁移已有模型／续训

压缩包**不含已有runs、results、checkpoint、图片或Git历史**。
要继续已有训练，将原项目相应整个`runs/<运行名称>`复制到本包的`runs/`中。
旧模型需要对应`metadata.json`恢复场景；新检查点内嵌参数也保留该文件，方便核对。
不要只复制不含参数的旧checkpoint。

在从头训练脚本的完整参数基础上，将`--updates`设为**追加更新数**，加入：

```text
--load-checkpoint .\runs\<原运行名称>\checkpoints\latest.pt
--run-name <原运行名称>
--append-log
```

例如49次已完成、目标累计320次，则追加271次；控制台应从050/320开始。
对于本包的当前seed2026／mec-latency200配置，可以直接使用：

```powershell
.\train_proposed.ps1 -Updates 271 -LoadCheckpoint ".\runs\<原运行名称>\checkpoints\latest.pt" -RunName "<原运行名称>" -AppendLog
```

其他历史场景请使用原入口并传入它们原本的完整训练参数，不套用当前配置脚本。
`--append-log`不可遗漏，否则原主日志被清空。
如果异常中断留下了超过checkpoint更新编号的部分rollout/episode记录，
先备份并归档未提交记录，再同目录续训，避免重复计入绘图。
仅加载自己生成或可信来源的checkpoint。

## 绘图、诊断与参数校准

```powershell
python .\scripts\plot_rollout_training.py .\runs\<运行名称>
python .\scripts\plot_training_run.py .\runs\<运行名称>\logs\training.csv
python .\scripts\analyze_convergence.py .\runs\<运行名称>\logs\training.csv
python .\scripts\calibrate_environment.py --output .\reports\resource_report.json
```

绘图沿用原脚本（包含原有均值/P95、Fast/Slow reward及策略诊断功能）。
`config/service_budget.json`为当前内置预算的导出副本；默认训练仍使用相同的
内置预算，需要显式导入时可传`--service-calibration-file`。
资源预算中未实测的项目仍为工程估计，不因打包变成实测数据。

## 文件与边界

- `train_dual_ppo.py`：原Proposed完整入口，含训练/内置评估/断点恢复。
- `edge_drl/agents`、`models`、`env`、`allocators`：原核心实现与全部必要依赖。
- `edge_drl/comparison`：仅通用检查点和请求回放模块；不是额外对比算法。
- `scripts`：原训练绘图、趋势分析和资源/抽样校准工具。
- `docs`：当前场景、训练信用、Slow条件观测和NIC严格精度说明。
- `validation/fixtures`：真实NIC故障及病态尺度测试数据，无实验训练产物。
- `manifest.json`：打包时间、原Git HEAD、本地改动、逐文件哈希、依赖环境。

不包含MSO/Monolithic、DPMS/Algorithmic、DARP、HEFT或Unified PPO训练。
必要的`hierarchical.py`基线工具及equal/learned资源分配辅助函数仍保留，因为
原Proposed入口/环境直接依赖它们；打包未改变Proposed默认算法或动作选择。
