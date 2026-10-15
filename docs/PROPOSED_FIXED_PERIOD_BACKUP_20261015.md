# 自适应周期改动前备份

用户于2026-10-15明确要求备份本次改进、创建分支并实现自适应部署更新周期。

本备份保存固定10分钟的`joint-credit-v5`源码、测试、说明、Slow组件评估记录和独立代码包，作为新时间决策机制的可回退对照。此前Git基点为`dbd1e80`。

- 备份提交位于原分支`codex/backup-latency200-before-credit-fix-20261012`。
- 固定版本标签：`backup/proposed-fixed-period-v5-20261015`。
- 后续开发分支：`codex/adaptive-deployment-period-20261015`。
- 独立包：`dist/proposed_training_20261015_195157.zip`。
- 包SHA256：`6d551efad17442ceb7992791f205b00bf443177124bfaa6051f6642eefa87bb3`。
- 包验证报告：同名`.verification.json`；无优化器更新的真实32节点推理、环境签名及NIC证书通过。
- 功能测试覆盖412项，证据范围及首次全量测试中的测试映射修复记录见`PROPOSED_OPTIMIZATION_PROGRESS_20261015.md`。

训练权重、全部历史运行与图像继续保留在本机原路径，不纳入这个源码提交。原v2进程已不存在，最后完整更新207；已验证并额外保存不可变`runs/joint_credit_v2_s6_u320_20261015_144010/checkpoints/snapshots/update_0207.pt`，SHA256为`213cbc0f1b010cb10ed05460f9791bc44a6bc2572ee366d738fe6a8fd16a0f19`。

本备份不证明v5训练后优于DARP。正式性能训练由用户执行。自适应分支保持固定周期对照，公共计算/NIC/KKT、服务预算与时限保持原值；新增时间决策和相关采样/回报规则需在独立文档明确。
