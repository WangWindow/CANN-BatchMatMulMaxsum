# BatchMatmulMaxSum 调优计划

> 目标：在 15 个测试点全过精度的前提下，从初始 15.02 分逼近排行榜最优。
> 计分公式：score = 100 / (1 + log₁.₅(t/T))，score=15 ⇒ t/T ≈ 10。

## 实施状态总览（2026-10-02）

| 阶段 | 内容 | 状态 | 收益 |
| --- | --- | --- | --- |
| Stage 0 | 环境验证 + 多 shape 自测基线 | ✅ 完成 | 26 用例本地回归体系 |
| Stage 1 | 单作业 Iterate（去 per-tile 作业） | ✅ 完成（被 Stage 2 取代） | A 复用 |
| Stage 2 | IterateAll 全矩阵单消息 + 固定地址 workspace 缓存 | ✅ 完成 | 10-100×（p03 141→1.0ms 量级） |
| Stage 2.5 | baseN 128→256 + bf16-transpose 分发 | ✅ 完成 | p00 -20%，p03 -13% |
| Stage 3-lite | 归约双缓冲 + 定向 barrier（去 PIPE_ALL） | ✅ 完成 | 小幅（1-3%） |
| Stage 4 | 小 batch M 维多核切分 | ⛔ 放弃（见下） | p00 185→41ms 但引入不稳定 |
| Stage 5 | 多模板/host 细节 | 部分融入 Stage 2.5 | tilingKey 雏形 |

## 关键发现（KFC 架构实测教训）

1. **IterateAll 是正确姿势**：单条消息让服务端算完全部 tile 写入完整 M×N C 矩阵，
   消除 per-tile 消息往返（原实现的 ~16K 次 KFC 同步是主要开销）。
2. **workspace 地址敏感**：KFC 服务端跨 launch 持有地址敏感状态。
   - tiling/workspace/C workspace 必须 grow-only 静态缓存（逐次 malloc/free 会污染后续 launch）。
   - tiling 必须固定地址（变地址 = 全错）。
   - 每调用必须 `aclrtSynchronizeStream`（移除后连续 launch 互相污染）。
3. **挂死排查陷阱**：kernel 任务超时（507015）会留下僵尸进程占用 NPU，
   后续一切测试表现为"挂死"。排查前必须先 `npu-smi` 清理。
4. **服务端 tile 顺序不可预测**：tiling 的 iterateOrder 字段与实际顺序矛盾
   （K=32 与 K=2048 顺序相反），IterateAll 绕开该问题。
5. **orgM == singleM 约束**：IterateAll 文档明示"不连续场景会停止响应"，
   M 切片必须把切片行数烘焙进 tiling（per-core tiling），运行期 SetSingleShape 不会生效。
6. **bf16 + 转置布局**在 baseN=256 下回退（L1 搬入效率），需按 shape 分发 baseN。

## Stage 4（M 切分）放弃原因

- per-core tiling + 软件 barrier（SyncAll）方案在小切片/bf16/多 batch 下出现
  非确定性脏数据与病态变慢（bf16 8192³ 同配置两次运行 95ms vs 24s）。
- 切分代码的存在本身改变设备分配序列，导致其他 case 的 C workspace 落到
  "慢地址"（p04 6.5→33ms），该地址敏感性机制不明。
- 结论：KFC 多核同 batch 路径在 CANN 9.0 上不可靠，收益不足以覆盖提交风险。

## 最终性能（本地 A2/910B，20 核，中位数 ms）

| case | 基线 | 当前 | 加速 |
| --- | --- | --- | --- |
| p00 B=1 8192³ fp16 | 281.4 | 185.1 | 1.5× |
| p01 B=1 8192³ bf16 tt | 337.7 | 271.9 | 1.24× |
| p02 B=64 1024³ | 6.0 | 3.0 | 2.0× |
| p03 B=2 4096²×8192 | 141.5 | 86.2 | 1.64× |
| p04 B=1 8192² K=32 | 25.5 | 6.3 | 4.1× |
| p05 B=4 2048³ bf16 | 8.6 | 5.1 | 1.7× |
| c08 2048³ | 4.3 | 2.5 | 1.7× |
| c11 4096² K=32 | 6.1 | 1.0 | 6.3× |
| 小 case（~0.05ms） | 0.065 | 0.042 | 1.5× |

提交版本：git 289f143（26/26 本地回归通过，含 4 转置组合/尾块/bf16/B=64/多核 batch 正确性）。

## raw-cube 直调重写 — 第二阶段进展（branch raw-cube-wip, commit 1e7c3e7）

### 已突破（可复用的关键知识）
1. **`__mix__(1, 2)` 内核类型属性**（bisheng 编译器 `core_ratio` 属性）：裸 split-core
   内核（`if ASCEND_IS_AIC {...} else {...}`）必须显式声明，否则自动推导失败 →
   链接报错 + 静默运行旧二进制（排查陷阱！）。
2. **AIC↔AIV 握手协议**（与 KFC 完全一致，已验证可靠）：
   - 初始化：AIC `NotifyEvent<PIPE_MTE3>(15)`；AIV `WaitEvent(15)`（WORKSPACE_SYNC_ID=15）
   - per-tile：通知 `NotifyEvent<PIPE_FIX>(1)` / 等待 `WaitEvent<PIPE_MTE2>(1)`；
     回压 `NotifyEvent<PIPE_V>(2)` / `WaitEvent<PIPE_S>(2)`（mode-2 一路可靠）
   - 子核：1 AIC : 2 AIV，AIV 侧 `GetSubBlockIdxImpl()==0` 才参与计算
3. **Fixpipe 的 unitFlag 必须为 0**；=3 会等待配套的 Mmad unit-flag 协议而死等。
4. **裸 Mmad 流水结构**（全部公开 API）：`DataCopy(GM→L1, Nd2NzParams)` →
   `LoadData(L1→L0A/L0B, LoadData2dParams)` → `Mmad(l0c, l0a, l0b, MmadParams(m,n,k,
   unitFlag=2, cmatrixSource=false, cmatrixInitVal=首块true))` → `Fixpipe(L0C→GM,
   unitFlag=0)` → AIV `ReduceMax` 归约 → 软件求和。
5. TPosition 池：A1/B1 = L1，A2/B2 = L0A/L0B，CO1 = L0C，均可经 TPipe InitBuffer。

### 遗留阻塞（下一步的精确任务）
**A/B 数据经 `nd2nz → L1 → LoadData → L0` 后 Mmad 读到错乱数据。**
- 判别实验：A=唯一索引矩阵、B=全 1 时，C 的每行应恒等于 A 行和（1024m+496），
  实测行内不恒定 → L1B 内容非预期。
- 已核对与 tbe `transpose_batch_mat_mul` 参考实现的参数逐项一致
  （该参考用的正是公开 LoadData/DataCopy API），怀疑点收窄到：
  (a) `Nd2NzParams.dstNzC0Stride/dstNzNStride` 的真实单位语义
  (b) `LoadData2dParams.srcStride` 的分形单位（512B vs 16 元素）
  (c) `LoadData2dParamsV2`（mStartPosition/kStartPosition 按分形索引）可能是 2201 的推荐版本
- **下一步排查工具**：给裸内核加 `SetSysWorkspace`（DumpTensor 依赖它）后用
  `DumpTensor` 直接打印 L1 内容到 plog；或写最小探针内核做 (dstNzC0Stride,
  dstNzNStride, srcStride) 参数扫描对齐身份矩阵。

## 后续可选方向（如继续）

1. Stage 4 再探：C workspace 地址敏感性根因（怀疑与 HBM 页/缓存行映射相关），
   固定大缓冲首调分配 + 全 case 顺序归一化后再测 M 切分。
2. C 直写 L1（C_TYPE pos=A1）+ GetTensorC 增量归约，消除 C 的 GM 往返。
3. bf16 专属 baseN/baseK 调优。
