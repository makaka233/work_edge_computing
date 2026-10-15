# Proposed 改进前备份：joint-credit-v2

备份日期：2026-10-15（UTC+8）。本文件所在提交是继续改进前的代码备份。

- 沿用分支：`codex/backup-latency200-before-credit-fix-20261012`。
- 前一远端备份：`123a0a3a43df300f7c0473dffa71c92a08f9653d`。
- 本次保存当前 Proposed 源码、启动与打包脚本、专项测试、设计文档和交接 MD/JSON。
- `PROPOSED_HANDOFF_20261015.md/.json` 是 15:58 的历史快照，其中“未提交”描述的是交接当时状态；本次提交将该实现纳入版本控制。

## 已保存的功能

`--proposed-design joint-credit-v2` 显式启用阶段额度部署（`stage-budget-v3`）、真实窗口终端回报与前缀信用（`prefix-credit-v2`）、竞争感知观测、Fast 独立 Critic 和受控分块优化、独立更新后 checkpoint 探针，以及训练与评估的版本化采样机制。仍采用 Slow PPO + Fast PPO 联合训练。

## 运行记录与证据边界

2026-10-15 16:20:45 读取训练日志时：

- 最新运行：`runs/joint_credit_v2_s6_u320_20261015_144010`，已记录 62 次完整更新，未观察到 `last.pt`。
- 第 60 次固定探针平均时延为 `0.20962562102125706 s`；这是 10 分钟名义负载抽样探针，不能当作完整五场景评估。
- 旧完整训练：`runs/slow_repair_scratch_s6_u320_20261014_162352`，完成 320 次更新。
- Slow 是否独立优于初始/随机部署仍待固定 Fast、固定请求轨迹的配对验证；测试通过、非零梯度和总体时延下降不构成该结论。

训练继续变化时，上述数字仅代表读取时点。本次备份未操作训练进程。

## 验证与本地产物

- 本次专项检查：`python -m pytest -q tests/test_joint_credit_v2.py`，18 项通过（18.68 秒）。
- `git diff --check` 通过。完整回归和独立打包验证的此前记录见 `docs/JOINT_CREDIT_V2.md` 与交接资料。
- 代码包：`dist/proposed_training_20261015_143344.zip`。
- ZIP SHA256：`5052942193a0355fe67601cb2d2f583005d972ad672790130d29564a3700ab2d`。

`runs/`、`runs_tmp/`、`results/`、`dist/`、模型、图像、缓存及快捷方式保留在本地，不纳入此次源码提交。复现或迁移训练需另外复制相应运行目录；代码包不含 checkpoint。
