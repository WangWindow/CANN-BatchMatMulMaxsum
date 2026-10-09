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

## raw-cube 第三阶段（2026-10-09 会话）：正确性打通 + 剩余阻塞

### 本会话重大突破（全部已验证）
1. **ND2NZ 真实布局实测确定**（S1 探针实验）：
   `out[d1*(dstNzC0Stride*16) + n1*256 + n0*16 + d0] = src[n1*16+n0, d1*16+d0]`
   其中源矩阵 (nValue 行 × dValue 列)；d 块间距由 dstNzC0Stride 参数控制。
2. **分形 = 16x16 fp16 = 256 元素（512B）** —— 此前把 512 *元素* 当步长（差 2 倍），
   是所有数据错乱的根源。L0A/L0B 的 LoadData 源步长必须按 256 元素/分形计。
3. **数据搬运通路全部实测验证**：L1 平拷贝（blockLen 单位=32B）、L1↔GM 双向、
   `SetFlag/WaitFlag` 事件、LoadData+Mmad+Fixpipe 全链路（c01-c05 全过，含四种转置）。
4. **转置加载路径（已验证）**：
   - tx2 B：L1[k块][n块] → `LoadData(l0b[i*nFrag*256], bHalfBuf[i*kNTask*16], params(0,nFrag,1,0,0,false,0))`
   - tx1 A：L1[m块][k块] → `LoadData(l0a[i*kFrag*256], aHalfBuf[i*kRound*16], params(0,kFrag,1,0,0,true,0))`
5. **AIV 归约修复**：C 块行距恒为 kNTask；部分 N 分片需掩蔽无效列
   （矢量 Duplicate 起点必须 32B 对齐；不足 8 元素的尾部用标量 SetValue 补齐——
   未对齐 Duplicate 会导致 16ms 级矢量异常！）。
6. **多核 AIV 子核选择**：`GetSubBlockIdxImpl()` 在 AIV 上返回 = blockIdx（核索引），
   不是 AIV 对内编号；当前用"不过滤"（两 AIV 冗余执行，幂等）。
7. **ND2NZ 越界读防护**：所有 nd2nz 的 nValue/dValue 必须钳制到实际范围
   （越界到未映射页会导致任务卡死/异常）。
8. **事件 ID 分工**（挂起角色独立 ID）：MTE1_MTE2=0/1；M_MTE1=2；
   FIX_M=7；自含对共用 MTE2_MTE1=4、MTE1_M=5、M_FIX=6。

### 当前实测状态（关键修正！exit=0 ≠ 数值正确）
- **数值精确通过（金标验证）**：c00-c05（含奇数形状 / 四种转置 / 双 batch 双 block）。
- **>8 tile 的用例全部数值错误**（如 c08 2048³ diff=1e5；c07 41/64 批次错）并伴有：
  - 部分场景挂起（c06/c11/c14/p04，~第 8 个 tile 起）；
  - 部分场景 507015（AI 核异常，地址布局相关，偶发）。
- **结论：存在的唯一核心阻塞 = AIC↔AIV 的 tile 流同步**。
  - 现象：约前 6-8 个 tile 正确；之后同步崩溃（表现为数据错乱/停滞/异常）。
  - 机制（高置信）：AIC 单向 `NotifyEvent(mode2, flag1)` 洪泛 + 无背压 →
    低频 ffts 邮箱/标志计数器溢出（深度约 8）。
  - 已验证可用的部分：ND2NZ 布局、四种转置加载、L0 流水、Mmad、Fixpipe、
    AIV 归约（mask+reduce）、事件对（MTE2_MTE1/MTE1_M/MTE1_MTE2/M_MTE1/M_FIX/FIX_M）。
  - 已验证失败的尝试：mode-4 配对（挂）、flag 交替 1+ping（挂）、
    AIV→AIC 回执 NotifyEvent(mode2, flag2)（死锁）、PIPE_ALL 屏障（死锁）、
    AIC 标量自旋减速（无效）、`sb==blockIdx` 过滤（仅 2 块成立，8 块误杀）。

### 第四轮 probing 新增的确凿知识（2026-10-09 晚）
#### ⚠️ 测试协议教训（最重要！！）
**FFTS 标志与硬事件跨发射持久**：任何提前 return 的测试路径若不消费内核的预置硬事件
（M_MTE1(2)/MTE1_MTE2(0,1)/FIX_M(7)）与 FFTS 消息，会导致后续发射相位漂移，
表现为"同一代码先通过、后连续挂起"——本轮 A2-A4 的二分结论（"TBuf/DataCopy 破坏握手"）
**全部因此失效**。今后任何内核级测试必须：
1. 每条测试路径都消费全部预置事件（或全程无预置）；
2. 每个配置在**独立进程**中运行，且同一进程内重复 ≥3 次验证确定性；
3. 判定"通过"必须同时满足：进程退出码 0 + 数值金标（如适用）+ 重复运行稳定。
#### ffts 协议确凿结论（保留）
1. AIV→AIC 方向可用：`CrossCoreSetFlag<2, PIPE_MTE2>` + `CrossCoreWaitFlag<0, PIPE_S>`。
2. 电平语义（幂等置位）：同一 flag 连续两次 Wait 会死锁。
3. gemm_v3 精确常量：C_NOTIFY_V=0 / V_NOTIFY_C=2；INTRA_BLOCK_SYNC=2；置位
   PIPE_FIX（C 向）/PIPE_MTE2（V 向）；等待默认 <0,PIPE_S>；flag 随槽位 ping 交替 {0,1}/{2,3}。
4. 纯握手 ping-pong（20 轮、无 cleanup）**首次运行通过**（后因污染漂移）——消息层本身可用，
   关键在"每轮严格配对、全程清理"。
5. AIC 的 `NotifyEvent<PIPE_FIX>` 洪泛 + 单 flag 是 >8 tile 失败的最可能来源；
   要么实现严格双向配对（带 cleanup），要么改用多 flag 交替 + 多槽环（kCSlots=8 已实现过一次，
   单独使用时 h4 仍挂——需与清理协议一起验证）。

1. **AIV→AIC ffts 方向已验证可用**（此前一直误判不通）：
   - AIV 发：`CrossCoreSetFlag<2, PIPE_MTE2>(flagId)`（mode 2 + PIPE_MTE2）
   - AIC 等：`CrossCoreWaitFlag<0, PIPE_S>(flagId)`（**必须用默认模板 <0, PIPE_S>**；等待端
     的 mode 被忽略，但 pipe 决定阻塞位置——用 PIPE_FIX 会挂）
   - 最小探针（AIV 起点发 flag7 / AIC 起点等 flag7）实测通过。
2. **ffts 标志是电平语义（幂等置位）**：同一 flag 连续两次 Wait 会死锁（第二次无新置位）；
   "多次置位=一次"。
3. **gemm_v3 的精确常量**：`C_NOTIFY_V=0`、`V_NOTIFY_C=2`（两对 flag 随槽位 ping 交替
   {0,1}/{2,3}）；`INTRA_BLOCK_SYNC=2`（mode）；置位用 PIPE_FIX（C 向）/PIPE_MTE2（V 向）；
   等待用默认 <0, PIPE_S>；第一轮无需预置（源码中无 pre-set，电平初值=可用）。
4. **本轮尝试失败的组合（重要线索）**：把完整 4-flag 双向方案套进现内核后，
   单 tile/单块（c01）过，但 6 tile 起首发射即挂（≤6 tile 的简单单 flag 方案反而全对）。
   → 说明失败点随 tile 数从 ">8" 前移到 ">=2"，即**双向交换本身引入了一个顺序/相位问题**，
   尚差最后一步定位（下一步首选：用小规模 dump/marker 定位新方案在第 1→2 tile 的相位）。

### 下一步（精确任务，优先级排序）
1. **读 ffts 语义**：`kernel_operator_sync_impl.h` 的 `wait_flag_dev`/`ffts_cross_core_sync` 注释；
   grep `gemm_v3_base_kernel.h`/`mmad_*.h` 中 `C_NOTIFY_V`、`V_NOTIFY_C` 常量与
   `INTRA_BLOCK_SYNC` 值，逐字复刻 KFC 的"服务端通知 + 客户端回执"完整协议
   （含双向 mode 与 flag 编号）。
2. **首选参考**：`matmul_server_aux.h:304`（服务端等客户端）+ `:485`（服务端通知客户端）+
   客户端侧 `CrossCoreSetFlag<INTRA_MODE, PIPE_V>`（客户端回执方向）——
   三处一起对照实现双向握手，先用 h4（2048x2048x32）验证 >16 tile 正确。
3. **背压**：AIC 在复用 C 槽前必须等到 AIV 对该槽的读取完成（回执）。
4. 全 26 用例金标回归（注意 exit=0 不等于数值对！必须逐点跑 verify）+ 替换 main 提交。

## 后续可选方向（如继续）

1. Stage 4 再探：C workspace 地址敏感性根因（怀疑与 HBM 页/缓存行映射相关），
   固定大缓冲首调分配 + 全 case 顺序归一化后再测 M 切分。
2. C 直写 L1（C_TYPE pos=A1）+ GetTensorC 增量归约，消除 C 的 GM 往返。
3. bf16 专属 baseN/baseK 调优。
