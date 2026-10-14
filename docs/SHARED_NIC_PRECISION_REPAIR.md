# Shared NIC 严格精度修复（2026-10-14）

原代码备份为 `9470a0e`；修复不新建分支，不删除训练产物。
此次仅改变共享 NIC 的数值求解兜底与断点配置恢复，不改变 PPO 结构、奖励、
场景、服务预算、带宽、网卡容量或精度验收要求。

## 真实故障与验证

运行 `slow_fast_v2_joint_s6_u320_20261014_005916` 在第50 update 第11个窗口中断。
`latest.pt` 保存49个完整 update（98个完整episode）。第50次未完成采样需重做，
不用从头训练。

隔离重放捕获原始负载矩阵、链路容量、NIC容量及warm-start对偶变量，存为
`tests/fixtures/shared_nic_update50_failure.npz`。用备份代码直接计算该矩阵，
复现与用户终端一致的错误：

```text
max_port_capacity_ratio=1.0000386
relative_gap=2.15e-06
joint_dual_status=CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
```

同一输入，修复代码的默认精度仍为1e-7：

- 相对 primal-dual gap：1.8530924491927706e-9。
- 返回分配的最大发送／接收端口容量比例：均为1.0。
- 目标值：0.08782064815824883。

旧报错的容量比例来自对偶生成的原始候选带宽，并非已接受/结算的分配。
修复后同口径原始候选比例为1.000000037349134；原有可行primal恢复后为1.0。
恢复所产生的目标值变化计入独立primal-dual gap，不能靠投影隐藏精度误差。

这里比较的是数值解与独立可行primal/dual界，不是调整超时门槛或接受优化器
的success标志。每一条边的容量约束也逐项检查。

## 原因与修复

小流量与大流量、相互依赖的发送/接收端口共同出现时，问题数值尺度跨度大：
35次二分端口求根会产生分辨率下限，L-BFGS与交替坐标法也可能在病态方向
停滞。仅增加迭代次数或二分次数不足以解决所有压力样本。

保留常见路径不变，仅未通过原certificate时执行：

1. float64的64次端口求根，提高根精度；
2. 同一凸对偶问题的联合SLSQP求解；
3. 解析Hessian的联合投影Newton精修：对Hessian对角均衡，SVD处理端口
   的发送/接收对偶gauge零空间，非负边界与回溯保证候选对偶合法。

所有阶段解的都是原来的 `min sum(S_e²/b_e)`，没有目标正则化或新增奖励。
Hessian均衡与SVD仅改善线性代数的条件性，不放宽最终精度要求。
始终要求可行primal与dual的相对间隙≤1e-7；没有通过时仍明确报错。
没有改成edge-only，没有跳过失败请求，也不使用上一次带宽分配冒充新解。

## 续训配置恢复

原训练checkpoint并未内嵌args，主入口续训却只读取内嵌args，导致配置缺失时
回落到edge-only、legacy工作量、left-hold抽样。现改为：

- 优先内嵌原配置；
- 旧checkpoint读取原run的metadata.json（包含snapshots子目录的情况）；
- repaired checkpoint缺少原配置时停止并说明，不能默默切换环境；
- 新checkpoint内嵌完整args，单独移动文件也可恢复原配置。

普通旧模型加载接口与权重兼容性不变。明确指定的CLI覆盖选项仍保持原接口。
续训指令应继承shared-nic、audited-v1、stratified-jitter及100秒维持上限。

## 有效与无效诊断产物

`runs_tmp/nic_update50_exact_replay_20261014`：修复代码直接重放，原物理／抽样
配置，完成第50次更新、两个60分钟episode、十二窗口、72次抽样结算，非法动作0，
模型保存正常。

`runs_tmp/nic_update50_reference_replay_20261014`：先调用旧求解器，在同一真实
故障矩阵上确认失败，再调用修复求解器并独立检查certificate后继续；完成第50次。
不是降低网络约束后的绕过。

`runs_tmp/nic_update50_reproduction_20261014`：最初诊断触发了旧续训配置缺失
问题，环境回落到了legacy，因此标记无效并保留，不能用于证明修复成功。

两个故障fixture与64个大动态范围压力样本均按原1e-7标准检查。
已有32节点稀疏网络、逐请求prefix、warm/cold启动、环境与Fast信用一致性测试保留。
完整回归303项通过（86.73秒）；源latest仍为update49，隔离重放的update50
模型保存了shared-nic、audited-v1、stratified-jitter、100秒维持上限的完整args。
数值压力测试不是对任意病态矩阵永不失败的数学保证，失败仍必须拒绝而非降标准。
