# Phase 0 Research（更新版）: 性能差距诊断与改进框架

**Feature**: 002-perf-gap-analysis | **Date**: 2026-10-14（第二轮更新，追加 R7–R12）


## R1 — 根因量化口径（不变）

三口径交叉验证：判题逐点（设备侧）/ 本地 TFLOPS / 结构流量手算。已由 R7 强化。

## R2 — 差距归因结论（更新：C 类已收敛，W 类细化）

- **Decision**: 四类根因收敛为——
  - **A. 计算单元空转（最大杠杆，仍成立）**：K 扫描斜率 0.1µs/K ≈ 84 TFLOPS ≈ 峰值 21%；
  - **B. C 数据 GM 往返（降级）**：判别实验（同 MAC、C 翻倍 K 减半：0.304→0.287ms）证明**非瓶颈**；
  - **C. N 轴并行触发缺失（已交付）**：S12f 判题 27.21，P14 10.7×→3.7×；
  - **D. 固定开销（成立）**：K 扫描截距 ~99µs（含 C 往返仅 ~33µs，其余为每 tile/块开销 ~66µs）。
- **Rationale**: 判别与缩放实验交叉验证；判题同分/位移模式一致。
- **Alternatives considered**: 继续按"C 通路"主线投入（被判别实验否决）。

## R3 — 首项行动：tileSplit 触发解耦（已交付）

- **Decision**: 结构性资格 + 抗碎片条件 `nShards ≥ 8`（c06 实测否决过宽触发）。
- **Outcome**: S12f 判题 27.21（P14 ≥2×，15/15）。修复链（V→MTE3 依赖缺失为根因等）见 PLAN.md。

## R4 — 代价门槛标定方法（已应用）

- **Outcome**: `nShards ≥ 8 && workItems*nShards ≥ 核数` 经 c06（3 切片 12.7× 回归）与 nsd（32-64 切片全受益）双向标定。

## R5 — 结构性方向框架化（更新：已由 R7–R12 具化）

- 方向②（Cube 喂数）→ 具化为 **A2 fullload + A5 环瘦身**；
- 方向③（C 通路）→ 判别实验降级，保留 A5 环瘦身作为唯一实验；
- 新增 **A3 streamK（MN 欠并行）**、**A4 swat（B 侧复用）**。

## R6 — 微型点方向与已否证清单（更新）

- **Decision**: 微型点（④）改为"零改动探针先行"（形状未知，盲目实现风险大）。
- **已否证清单（扩充）**：
  - K 预重排（memcpy2d ~6µs/行，历史）；
  - L1 半区 2→4（S10 本地无收益）；
  - **B 侧装载同构重组**（8 次顺序 4-分形 vs 4 次跨步 nFrag：实测 k1024 +7.6% 变差；结论：transpose 路径大 repeat 更优）；
  - **B 侧 nd2nz 布局对齐**（转置代价由输入朝向决定，行块恒定内层，无法消除）；
  - 判题单点微优化（<15-20% 不可归因，噪声 ±5% 实证）。

## R7 — 第二轮实测证据（计量收敛）

- **Decision**: 主攻喂数域。证据：
  - K 扫描 (4,1024,1024,K)：0.125/0.151/0.206/0.304ms(K=256..2048) → 斜率 0.1µs/K、截距 99µs；
  - 判别实验：同 MAC 下 C 翻倍+K 减半 0.304→0.287ms（C 通路非瓶颈）；
  - tx 同形对比：0.206/0.222/0.157/0.199ms（tx00/01/10/11）→ 装载路径敏感 ±20-40%，`srcStride=1` 顺序化最快；
  - 核数缩放：slope @20=0.1、@4=0.3466µs/K（5×少核 3.47×慢）⇒ 斜率 ~70% 每核 + 30% 共享；每核 Cube 利用率 ~43-49%。
- **Rationale**: 四项独立实验交叉；每核装载管线是硬底，共享带宽为余量。
- **Alternatives considered**: 纯推测"框架慢"（拒绝）。

## R8 — A2：A-Full-Load（官方 fullload 配方）决策

- **Decision**: 采纳。条件核算：A 侧 128×K×2B ≤ L1/2=256KB ⇔ **K≤1024**；对侧循环 T = nShards ≥2（trigger 形状必满足）；喂数域为实测主瓶颈（R7）。
  实施要点（官方配方）：A 全量独立 L1 区；**取消 A 侧 pingpong 释放事件**；A 的 MTE2 仅在本 item 首轮发；后续 kc 直接按 `kc×kKChunk` 偏移做 L1→L0 切片；B 侧保留 pingpong 不变。
- **Rationale**: 官方实测 -5~15%（同构场景 -7.5%）；本地等价证据：A 装载路径敏感（±20-40%）。
- **Alternatives considered**: B-Full-Load（对称）——B 侧跨对侧循环为 mTiles，但 B 区 K×128×2B 同量且 B 侧当前已每 nShard 只读一次，收益面窄，暂缓。
- **约束**: 官方声明 fullload 与 streamK 互斥（本特性二选一于同一形状：K≤1024 走 A2，长 K 欠并行走 A3）。

## R9 — A3：K 轴切分（streamK 化）决策

- **Decision**: 以**扩展 S12 两阶段机器**的方式实现 K 段切分（非照搬 matmul_streamk 模板）：宿主把"欠并行 + 足长 K"形状的 flat 空间扩展为 (tile × kSeg)；AIC 各段产出部分 C 到 workspace（现有环槽语义扩展）；AIV 归约从"跨块 MAX"扩为"K 段 ADD → 跨块 MAX → 对 m 求和"。
- **Rationale**: 官方 streamK 触发条件（mCnt·nCnt ≤ aicNum/2 且 K≥4096 等）与我们 P8/P15 类欠并行假设吻合；复用既有 workspace/计数/合成 = 改动面小于新模板。
- **Alternatives considered**:
  - 单纯放宽 N 轴 nShards≥8 门槛（可覆盖 nShards 4-7 类）→ 作为 A3 的低成本补充子项，但 c06 抗碎片边界仍需保留（拒绝裸放宽）；
  - 全套 matmul_streamk 模板移植（依赖 Matmul API 栈，与单文件裸内核不兼容，拒绝）。

## R10 — A4：SWAT Serpentine（机制 A 先行）决策

- **Decision**: 仅实现机制 A（4 行窗口遍历，B 片窗口内复用）；机制 B/C（尾块合并、末轮二维再切分）留后续。评估与 A2 的叠加（官方 A_FULL_LOAD_MODE）。
- **Rationale**: 重复读放大（134MB vs 17MB @kscan 形状）是实测过的结构代价；WINDOW_LEN=4 为官方常量，host/kernel 边界需对齐。
- **Alternatives considered**: 立即实现 B/C（尾碎片在当前 128×128 粒度下影响小于 B 复用，拒绝此批次）。

## R11 — 微型点探针决策

- **Decision**: 先做零改动分解探针（c00 及其变体的耗时构成：launch / 序言 / 同步往返），再决定是否立项 A6 超轻路径。**本批次不实施快速路径**。
- **Rationale**: 判题形状未知（NEEDS CLARIFICATION）；8µs 构成未分解前任何实现都是赌博。
- **Alternatives considered**: 直接实现纯向量直算路径（改动面大、形状假设未证，拒绝）。

## R12 — A5：环瘦身实验决策（CATLASS workspace 原则）

- **Decision**: 做 32 槽 → 8 槽的常量级 A/B 实验。
- **Rationale**: CATLASS 明确"小型轮转 workspace（L2 驻留）+ 细粒度回调"为融合 matmul+epilogue 的性能最优形态；"整块 [M,N] 的 HBM 往返是示例级实现"。现环 32 槽×64KB×20 核 = 40MB 聚合；代码注释曾写"8 槽环"（疑历史遗留）。
- **Alternatives considered**: 环直接降为 4 槽（背压风险大，先 8 槽实验）。

**所有 NEEDS CLARIFICATION 已消解**（唯一保留项：P1/P2 判题形状——经 R11 探针方案处理为可接受风险）。
