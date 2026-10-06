# UnifiedPPO 失败算法对比归档

状态：按项目当前决策标记为**失败的算法对比尝试**，保留代码供复查，不作为已验证的论文对比结果。归档日期：2026-10-06。

本分支以 GitHub 提交 `7cfad490be9796eab289c87c5e819933121d2a60` 为起点，保存单一 PPO 同时控制阶段部署、请求路由、计算优先级和链路优先级的实现。其资源结算使用 `learned-share`，不调用 Proposed 的 KKT 最优分配。当前代码为 `single-policy-ledger-credit-v3`：共享编码器与优化器、条件动作头、连续 Beta 资源动作、批内资源账本和按真实结算步计算的信用。

归档时未完成与 Proposed 匹配训练预算、共享评估轨迹的正式五场景对比，因而不能把本分支的训练日志解释为已经证明 UnifiedPPO 收敛后的性能排名。已有本地训练尝试分别记录了早期版本与 v3 的部分 updates；完整原始记录保留在本地 `runs/unified_ppo/`，该目录受项目现有 `.gitignore` 排除。此前实验的 `results/` 及其中 checkpoint 也原样保留在本地，不混入本次算法代码归档。

本归档包含 `train_unified_ppo.py`、`edge_drl/agents/unified.py`、`edge_drl/models/unified_ppo.py`、`edge_drl/allocators/learned_share.py`、评估适配器、环境接口、测试和设计文档。旧版 UnifiedPPO checkpoint 与 v3 网络和信用结构不兼容，不能直接继续训练。
