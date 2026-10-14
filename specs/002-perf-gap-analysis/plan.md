# Implementation Plan（更新版）: 性能差距诊断与改进框架（对标榜首 5~10×）

**Branch**: `002-perf-gap-analysis` | **Date**: 2026-10-14（第二轮更新） | **Spec**: [spec.md](spec.md)


## Summary

**交付进展**：US1（根因诊断）✅、US2（改进框架）✅、**US3（首项行动 = tileSplit 触发解耦）已判题实证达成**——S12f 提交 27.21 分（前值 25.70），P14 用时 87.6µs→30.28µs（10.7×→3.7×，SC-003 的 ≥2× 要求满足，且 15/15 保持）。

本轮计划更新纳入**第二轮实测证据**与 **CANN 官方最佳实践（matmul 族）**：
- 实测（本地）：K 扫描（斜率 0.1µs/K ≈ 84 TFLOPS ≈ 峰值 21%，截距 ~99µs）；判别实验（C 翻倍+K 减半反而更快 ⇒ **C 通路非瓶颈**）；tx 对比（装载路径敏感 ±20-40%）；核数缩放（5× 少核仅 2.5× 慢 ⇒ 斜率 ~70% 每核 + 30% 共享）；判题噪声 ±5% 实证（S13a1 同分）。
- 官方知识（技能库）：`fullload`（小侧 L1 常驻，条件与 k1024 类形状逐条命中，-5~15%）；`streamk`（MN 欠并行 + 长 K 的切 K 方案）；`swat`（Serpentine 行窗口治 B 侧复用，可与 A_FULL_LOAD_MODE 叠加）；CATLASS 原则（**小型轮转 workspace（L2 驻留）+ 细粒度回调重叠**；"整块 C 的 HBM 往返是示例级实现，非性能最优"——我们的 32 槽 ×64KB×20 核 = 40MB 环待瘦身验证）。

产出：**第二批次行动项 A2–A5**（见下文），按"收益 × 置信度 ÷ 成本风险"排序；已否证清单同步扩充。

## Technical Context

**Language/Version**: AscendC 单文件内核（`kernel.asc`，CANN 9.0.0，dav_c220 / 910B，20 cube 核）

**Primary Dependencies**: 既有构建/验证链路（main.asc 本地框架、scripts 门禁资产、判题平台）；
**新增设计来源**：CANN 官方优化知识库（技能 `ascendc-performance-best-practices` 的 matmul 族：pingpong/fullload/streamk/swat；`catlass-op-design`：workspace 与 DispatchPolicy 原则）

**Storage**: N/A

**Testing**: 全量回归（seq15 + 244 矩阵 + soak + 核数变体）为交付唯一前置（宪法 v1.1.0 已取消快门禁）+ 判题逐点数据；
本地实验资产：kscan/tx/disc/disc 用例集、`BMMS_CORE_OVERRIDE` 核数变体开关、nsd/ns/t1/c06/c07 对照集

**Target Platform**: Ascend 910B（dav-2201，20 cube 核）

**Project Type**: 单文件算子内核（提交物仅 `kernel.asc`）

**Performance Goals**（更新）:
- P8–P12 大点：比值下降 ≥20%/项（A2 fullload 预期 -5~15%；A3 streamK 覆盖 MN 欠并行类；A4 swat 治重复读）
- P1/P2 微型点：先零改动探针分解 8µs（NEEDS CLARIFICATION：判题侧形状未知），探明后定目标（潜在 8µs→3-4µs）
- 噪声纪律：**单次判题提交须捆绑 ≥15-20% 量级变化**（判题噪声 ±5% 实证，微优化不可归因）

**Constraints**: 15/15 零回归；一次一个可归因变化集；**fullload 与 streamK 互斥**（官方约束）；已否证方向不再投入；回退点 aed32b9（S13a1, 27.21 实证）/ baa7ec7（S12f）；截止 10/17

**Scale/Scope**: 行动项 A2–A5 各 ≤1 个内核级变化集 + ≤2 套本地对照；每项一轮内可判定有效/无效

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 检查点 | 结论 |
|---|---|---|
| I. 判题实证唯一真理 | 全部结论以判题逐点数据（27.21 最新）+ 本地三口径（TFLOPS/流量/核数缩放）为证据；CAN N 官方文档作为机制依据（非臆测） | ✅ |
| II. 本地验证门禁 | 每项行动交付前必须全量回归全绿（seq15 + 244 矩阵 + soak + 核数变体）；矩阵口径 = 失败集合与 S13a1 逐例一致 | ✅ |
| III. 最小步进与可回退 | 每行动单一变化集；回退点 aed32b9 / baa7ec7 / 47dc041 三级 | ✅ |
| IV. 正确性先于性能 | 15/15 为前置；收益按机制归因（装载路径/重复读/欠并行/固定开销） | ✅ |
| V. 环境韧性与可追溯 | 版本标记 `[S13-xx]`；判题结果与实验证据归档 PLAN.md；技能库引用写入 research.md | ✅ |

无违规 ⇒ Complexity Tracking 留空。**Post-Design Re-check**：设计产物未引入新违规，复检通过。

## Project Structure

### Documentation (this feature)

```text
specs/002-perf-gap-analysis/
├── plan.md              # 本文件（本轮更新：+第二批次行动项 A2–A5）
├── spec.md              # 需求规格（不变）
├── research.md          # Phase 0（本轮更新：+R7–R12，含 CANN 技能依据）
├── data-model.md        # Phase 1（本轮更新：新增实体与实例刷新）
├── quickstart.md        # Phase 1（本轮更新：全量回归协议 v1.1.0 + 资产清单）
├── contracts/
│   └── diagnosis-report.md  # 契约（行动项扩展至第二批）
└── checklists/requirements.md
```

### Source Code (repository root)

```text
kernel.asc    # 行动项 A2–A5 的唯一变更面（AIC 装载/调度 + 宿主触发）
build/        # 本地资产：kscan_cases.txt / tx_cases.txt / disc_cases.txt / nsd_cases.txt /
              #   ns_cases.txt / t_cases.txt / seq15.txt / matrix_all.txt / s12_pts.txt + 金标
PLAN.md       # 判题事实、实验证据、方向进展的归档处（含 27.21 判题记录）
```

**Structure Decision**: 沿用单文件内核工程；本轮全部行动项限于 `kernel.asc` 与本地资产，不引入新框架。

## 第二批次行动项（A2–A5）

> 契约字段齐全（机制假设 | 预期收益 | 置信度 | 成本 | 风险 | 验证方法 | 回退点 | 判题可验证性）；排序 = 收益×置信度÷成本风险。

### A2 — A-Full-Load（小侧 A 矩阵 L1 常驻）【优先级 1】

| 字段 | 内容 |
|---|---|
| 机制假设 | 消除 nShards 循环中对 A 的重复 GM→L1 搬运（MTE2 压缩 (T-1)/T，T=nShards）；官方文档条件逐条命中：A 侧 128×K×2B ≤ L1/2（K≤1024）、T≥2、喂数域为实测主瓶颈 |
| 预期收益 | 受影响形状 -5~15%（官方实测 -7.5% 同构场景；本地 kscan K≤1024 子集可测） |
| 置信度 | 高（官方配方 + 本地 tx/装载敏感证据） |
| 成本 | 中（L1 布局改造 + A 侧事件模型：取消 pingpong 释放事件、仅首轮 MTE2、L0 按 kc 偏移切片） |
| 风险 | L1 预算冲突（B pingpong 须保留）；K>1024 回退当前路径 |
| 验证方法 | 本地：kscan(K≤1024)+tx A/B ≥5% 且正确性全绿 → 全量回归 → 判题捆绑提交 |
| 回退点 | `git checkout aed32b9 -- kernel.asc` |
| 判题可验证性 | 可（捆绑提交，看大点/触发点位移） |
| 互斥约束 | 与 A3(streamK) 互斥（官方）；与 A4(swat A_FULL_LOAD_MODE) 可叠加 |

### A3 — K 轴切分（streamK 化现有两阶段机器）【优先级 2】

| 字段 | 内容 |
|---|---|
| 机制假设 | MN 欠并行（mCnt·nCnt < 核数）+ 长 K（≥2048）时，把空闲核拉去切 K；跨核归约复用 S12 的 workspace/计数/合成架构（归约语义从 MAX 扩为 MAX+ADD 两段） |
| 预期收益 | 该类形状并行度提升至近核数倍（如 4 tile → ~20 核份） |
| 置信度 | 中高（官方 streamK 场景与 P8/P15 类形状假设吻合） |
| 成本 | 高（宿主触发 + AIC 段切分 + AIV 归约扩展） |
| 风险 | 与 N 轴/M 轴切分的模式互斥管理复杂；尾段负载均衡 |
| 验证方法 | 构造 (1,256,256,16384) 类用例 + 触发面回归（nsd/ns/t1/c06）+ 全量回归 |
| 回退点 | `git checkout aed32b9 -- kernel.asc` |
| 判题可验证性 | 可（针对欠并行点） |

### A4 — SWAT Serpentine（B 侧窗口复用）【优先级 3】

| 字段 | 内容 |
|---|---|
| 机制假设 | 4 行窗口（WINDOW_LEN=4）遍历使 B 片在窗口内复用，治输入重复读放大（实测 134MB vs 最小 17MB，@(4,1024,1024,1024)） |
| 预期收益 | 大点的 L2 带宽压力下降；比例待实测（预测 -5~15%） |
| 置信度 | 中（机制依据充分，但需与本内核的环/同步序兼容） |
| 成本 | 中（仅改 AIC 的 flat 遍历序 + 宿主触发面） |
| 风险 | 遍历序影响与 AIV 消费的耦合；需正确性全绿 |
| 验证方法 | 本地 disc/kscan A/B + 全量回归 |
| 回退点 | `git checkout aed32b9 -- kernel.asc` |
| 备注 | 机制 B（尾块合并）/C（末轮二维再切分）留后续 |

### A5 — 环瘦身（32 槽 → 8 槽）实验【优先级 3.5，成本最低】

| 字段 | 内容 |
|---|---|
| 机制假设 | CATLASS 原则"小型轮转 workspace（L2 驻留）"；现环 32 槽×64KB×20 核=40MB 聚合远超 L2 友好工作集，可能在 L2 挤掉 A/B 片 |
| 预期收益 | 未知（实验性）；代码注释本写"8 槽环"，疑为历史遗留未同步 |
| 置信度 | 低-中（判别实验显示 C 通路有裕量，环大小未必是瓶颈） |
| 成本 | 极低（一个常量 + 同步阈值联动） |
| 风险 | 环过小导致 AIC 生产被 AIV 消费阻塞（背压加重） |
| 验证方法 | 常量级 A/B（kscan/tx）+ 全量回归 |
| 回退点 | 单行回改 |

### 微型点探针（前置研究项，非行动项）

零改动探针：分解 c00（1×1×1×32）8µs 构成（launch / 核内序言 / 同步往返）。探明后另行立项（潜在 A6：超轻路径，8µs→3-4µs，但判题形状未知 = NEEDS CLARIFICATION）。

## Complexity Tracking

> 无宪法违规；无需豁免。


## 第二批深化：Phase D — 数据流/分片/搬移层级化再设计（2026-10-14 追加，用户方向）

> 背景：A2 fullload 落地后（K≤1024 类，本地 -8~31%），K≈1024 形状的成本版图变为：
> 固定域 ~99µs（第一）> B 喂数 ~32µs > AIV 侧 C 读 ~33µs×2（双子核冗余）> A 喂数 ~8µs（已削）。
> 用户要求跳出"小补丁"：从数据流/分片/搬移三层寻找更优结构。

### D1｜AIV 双子核分工（消灭 2× C 读）【数据搬移】
- 机制：现"两子核全量冗余循环"⇒ 每环槽读两遍。改为按 item 奇偶分工（子核 s 只处理 item%2==s），每 item 单写者；notify 广播消费与计数框架不变。
- 预期：环读减半 -30µs 级 + 归约算力减半；**前置：E1-NR 消融**（取消 AIV 归约）实测 AIV 侧真实占比。
- 风险：ack 记账/背压配对需重核（每子核只为其处理过的 tile 打 ack）。

### D2｜SWAT Serpentine 遍历（跨块 B 复用）【数据流重排】
- 机制：官方 4 行窗口（WINDOW_LEN=4）；同窗口块共享 B 片驻留 L2。治 B 喂数跨块重复读（每 mTile 块重读同一 B slab）。
- 预期：B 域 -20~40% ⇒ 大点 -5~12%。可与 A2(fullload) 叠加（官方 A_FULL_LOAD_MODE 同构）。

### D3｜C 环瘦身/驻留 + 可选 fp16 环【搬移路径】
- 现 32 槽×64KB×20=40MB 聚合，与 CATLASS"小型 L2 驻留轮转 workspace"原则相悖；先 8/16 槽 A/B。
- fp16 环：C 字节减半但 10-bit 尾数 vs 1e-4 容差 = **高风险实验**，以全量回归容差判定，不通过即弃。

### D4｜K 轴切分（streamK 化）【数据分片根治】
- 覆盖 MN 欠并行（tile<核数）+ 长 K 类（P8/P11/P15 嫌疑病因）；复用 S12 两阶段机器，合成扩为
  "K 段加法归约 → 跨块 max → 对 m 求和"三段。
- 预期：该类形状近核数倍并行（该项若命中可达 2~4×）。

### D5｜大条带归约（2×nShard strip）【每 tile 成本】
- 同 item 相邻 nShard 环槽时序连续 ⇒ 128×256 条带单次 ReduceMax ⇒ 每 tile 归约轮次减半（固定域 -10~20µs）。

### D6｜B 侧锚定（A2 对偶）【分片/搬移】
- N≪M（N≤128、每 item 1 格）形状：B slab 驻留 + A 流式。与 D2 合流为"B 侧驻留与复用"主题。

### 执行序
E1-NR 消融 →（按数据）D1 或 D2 先行 → D3 环瘦身穿插 → D4 独立工作流 → D2/D6 合流。
