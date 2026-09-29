# PopularityDeploy：算法部署、Proposed 调度的部署侧对照

`PopularityDeploy` 用于回答一个单独的问题：Proposed 的 Slow PPO 学习式部署是否优于固定算法部署？

## 唯一改变

- **Proposed**：Slow PPO 部署 + Fast PPO 调度 + KKT 连续资源分配。
- **PopularityDeploy**：热度/容量/局部性算法部署 + 同一个冻结的 Fast PPO + 同一个 KKT 连续资源分配。

需求轨迹、物理网络、随机种子、Fast checkpoint、KKT 结算和指标计算均保持一致。该方案不训练，也不读取未来请求或评估结果。

## 固定部署算法

部署算法先用二元整数规划保证每个有效服务阶段至少有一个副本，并满足节点内存与存储约束；随后按“预期计算工作量 / 已部署计算能力 / 资源占用”确定扩副本优先级，在全部可行节点中选择需求加权接入时延和执行时延较小的位置。算法规则在评估前固定。

## 快速单场景命令

```powershell
$runName = "popularity_deploy_phase1_s6_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python .\run_comparison.py `
  --checkpoint ".\runs\global_incremental_cf_s6_u180_20260903_192623\checkpoints\best.pt" `
  --schemes Proposed PopularityDeploy `
  --phase 1 `
  --sampled-seconds-per-window 6 `
  --device cuda `
  --run-id $runName
```

## 解释边界

如果 `PopularityDeploy` 弱于 Proposed，差值支持“学习式部署在冻结的学习式调度之上仍有贡献”。它不能单独证明所有场景下都更优；正式论文结论仍需固定规则后的五场景、多种子置信区间。

## 固定规则后的快速预检

目录：`results/comparison/popularity_deploy_phase1_s6_20260929`

| 指标 | Proposed | PopularityDeploy | Proposed 相对降低 |
|---|---:|---:|---:|
| 平均时延 | 481.142 ms | 516.247 ms | 6.80% |
| P95 时延 | 774.862 ms | 873.943 ms | 11.34% |
| 超时率 | 0.32335 | 0.47605 | 32.08% |

两者使用同一条需求轨迹（trace hash 完全相同）、相同的 33,400 个请求、相同的 236 个平均副本，且无效动作率均为 0。因此，这次差异不是由请求数量或副本总量不同造成的；主要表现为算法部署使平均计算时延从 341.080 ms 增至 388.964 ms。该结果只用于确认实现方向和明显故障，不作为正式统计结论。
