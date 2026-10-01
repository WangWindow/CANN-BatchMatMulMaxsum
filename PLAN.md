# BatchMatmulMaxSum 调优计划

> 目标：在 15 个测试点全过精度的前提下，从当前 15.02 分逼近排行榜最优。
> 计分公式反推：score=15 ⇒ t/T ≈ 10×，即当前比最优慢约一个数量级。

## 环境核查结论（2026-10-01 实测）

- 当前机器即昇腾开发容器：`developer` 用户、`/mnt/workspace` 持久化、CANN 9.0.0 位于 `/home/developer/Ascend/cann-9.0.0`（ASCEND_HOME_PATH 已设置）。
- `npu-smi`：2× Ascend910 卡，Health OK，无占用进程。
- 旧二进制（Sep 17 构建）直接运行通过：case0 PASSED，max diff 2.4e-7。
- `msprof` 可用：`/home/developer/Ascend/cann-9.0.0/bin/msprof`。
- 旧 `.npus.yaml` 已不存在，但**不需要**——我们在 NPU 机器本地，直接用 `run.sh` 即可。
- 注意 SOC：CMakeLists 默认 `dav-2201`（A2/910B，与本机硬件一致）；kernel.asc 注释称面向 910C。本地优化结论在 910B 上测，提交后在裁判机（CANN 9.0.0）复核。核心数在运行时用 `aclrtGetDeviceInfo(ACL_DEV_ATTR_CUBE_CORE_NUM)` 查询，不写死。

## 现状诊断（为什么慢 ~10×）

对照 kernel.asc 当前实现，三大瓶颈：

1. **A/B 被反复从 GM 重读（最大瓶颈）**：每个 (mTile=64 行 × nShard=128 列) 都是一个独立 MatMul 作业（各自 SetTensorA/B + Iterate），A 块和 B 块反复重读。M=N=K=8192 时 MTE 流量约 26 GB，是唯一数据量（~400 MB）的 60 倍。
2. **小 Batch 单核**：`blockCount = min(coreCount, batch)`，B=1 时全量计算压在一个核上，其余核空闲。
3. **PIPE_ALL 串行化**：内层循环每个 op 后 `PipeBarrier<PIPE_ALL>`，cube/vector/MTE 无法重叠；另有 `GetValue(0)` 向量↔标量往返与 `SetValue` 标量循环补尾（命中仓库红线 B3）。

次要：`run_kernel` 每次调用 3× aclrtMalloc/Free + aclrtSynchronizeStream，小 shape host 开销占比高。

## 分阶段计划（每阶段先精度后性能，15 用例全过再进入下一阶段）

### Stage 0 — 环境验证与本地基线（不改算法）

- [x] 环境核验：CANN 9.0.0、npu-smi、旧二进制 case0 跑通（已完成）。
- [ ] 扩展本地测试：改造 `scripts/gen_data.py` + `main.asc` 为多 shape 循环（覆盖：B∈{1,4,64}、M/N/K 小/大、四种 transpose 组合、M/N 非 16 对齐尾块、小 K），每 shape 计时（取中位数），全量精度校验。
- [ ] `msprof` 采集基线：Task Duration、Cube/Vector/MTE 利用率，验证 26 GB 搬运放大假设。
- [ ] 记录基线分数映射（本地计时 → 裁判机分数），后续每阶段对比。

### Stage 1 — 主循环重构为单作业 Iterate（正确性优先，收益 ~2.8×）

- 去掉手写 mTile×nShard 双层循环：每个 (core, batch) 只做一次 `SetTensorA/B` + `SetSingleShape(m,n,k)` + `SetFixSplit(baseM, baseN, -1)`，`Iterate` 内自动遍历全部 tile，`GetTensorC` 逐个拿。
- 保守参数 baseM=64 / baseN=128 / FIRSTM（M 内层 → B 块留 L1 复用），GM 暂存 C tile（单缓冲）+ 现有归约逻辑不动，保证只改结构不改语义。
- 门禁：15 用例精度全过 + 本地计时对比。

### Stage 2 — 搬运量优化：baseN↑ + traverse 启发式（收益 ~3×）

- baseN 128→256/512；traverse 按 M/baseM 与 N/baseN 谁小选谁：M·baseN < N·baseM 时 FIRSTN，否则 FIRSTM。
- host 侧按 (m,n,k,B) 选 baseM∈{64,128}、baseN∈{128,256,512}（受 UB 192KB / L0C 256KB 约束），代表性 shape 小网格实测。
- 现场验证：SetFixSplit 的 baseN 上限、FIRSTM/FIRSTN 遍历语义（读 matmul_intf.h + msprof 实测）。

### Stage 3 — 流水化：队列 + 定向 barrier（对应"合理 API""32B 对齐"）

- C tile GM ping-pong + `TQue<VECIN,2>` 双缓冲，cube 算 tile i+1 时 vector 归约 tile i。
- 删除全部 `PIPE_ALL`（红线 B3），换 EnQue/DeQue + 定向 barrier。
- 尾块 `DataCopyPad`（行宽补 32B，pad -inf），消除 `SetValue` 标量循环。
- 实验：若 matmul_intf 支持 C 直写 UB（VECOUT），对比 "C-in-UB (baseN=256)" vs "GM 暂存 (baseN=512)" 实测 MTE，取优。

### Stage 4 — 小 Batch 多核切分 + 归约（B=1 类用例，收益最大）

- grid = batch × coresPerBatch（coresPerBatch = min(coreCount/batch, mTiles)），每核一个连续 M 段（按 mTiles 均分，负载均衡）。
- 每核归约自己 M 段 → partial，`AtomicAdd(y[b])`；host launch 前 `aclrtMemsetAsync(y, B*4, 0, stream)`；coresPerBatch==1 直接写 y（无需 memset/原子）。
- 验证 910B 上 float32 AtomicAdd 可用；否则备选 "partial 写 workspace + 二段小 kernel 归约"。

### Stage 5 — 多模板 tilingKey + 细节 + host 开销（对应"多模板""减少 scalar"）

- tilingKey 按 shape 分发：① 极小 shape（单 tile）→ 最简路径；② M 小（mTiles=1）→ 单核 FIRSTN；③ 大 shape → M 切分多核。transpose 组合用运行时参数（SetTensorA transpose 已覆盖）。
- scalar→vector：补尾 Duplicate 向量清零；`GetValue(0)` 累加改 UB 向量累加、最后一次性 ReduceSum。
- host：workspace/tiling 静态 device buffer 跨调用复用（不够再扩），命中缓存时跳过 malloc/free 与 `aclrtSynchronizeStream`；只缓存 workspace 绝不缓存结果（红线 A5）。

### Stage 6 — 终验与提交

- 全量自测 + msprof 终版报告（Cube/Vector/MTE 利用率、Task Duration）。
- 对照得分公式预估排名，提交；若与最优仍有差距，回到 Stage 2/3 做第二轮网格微调。

## 精度风险控制

- 全程保持 fp32 cube 累加与 fp32 归约不动（当前精度已过，改动不涉及数值路径）。
- Max 归约与求和顺序变化在 1e-4 容差内安全；每阶段先全量跑精度再测速。
- 每次提交前用 `scripts/verify_result.py` 全量校验 + 本地多 shape 自测。

## 预估收益

- Stage 1~2：最大 case 的 MTE 从 ~26 GB 降到 ~2.8 GB（约 9×）。
- Stage 4：解决 B=1 单核问题。
- 叠加流水重叠后预计整体进入 t/T ≈ 2 以内（60 分+），Stage 5 模板细化与 msprof 微调继续逼近最优。

## 与用户调优手段的对应

| 用户手段 | 落点 |
| --- | --- |
| 运算步骤少 | Stage 1 单作业 Iterate；Stage 5 单 tile 最简模板 |
| 减少多余 scalar 操作 | Stage 3 去 SetValue 循环；Stage 5 GetValue 累加改向量 |
| 选择合理 API | Stage 3 队列/定向 barrier 替代 PIPE_ALL；DataCopyPad |
| 数据对齐 32B | Stage 3 DataCopyPad 行宽补 32B |
| 多模板 + tilingKey | Stage 5 tilingKey 分发三类模板 |
| 大 shape 暂存 UB 还是 GM | Stage 3 C-in-UB vs GM 暂存实测决策 |

## 执行方式说明

- 本文件当前写入 `/home/developer/.opencode/plan/PLAN.md`（Plan mode 不允许写仓库）。
- 切换到实施模式后，第一步将本文件复制到仓库根 `PLAN.md`，并按 git-version-management skill 每阶段 checkpoint 提交。
