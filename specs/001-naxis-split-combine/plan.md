# Implementation Plan: N 轴并行切分与最大值合成

**Branch**: `001-naxis-split-combine` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-naxis-split-combine/spec.md`

## Summary

在已判题 15/15（S9，25.7 分）的单文件内核基线上，为 "batch×M 分块不足以铺满核数" 的形状引入
**tile 级（含 N 切片）工作划分 + 两阶段最大值合成**：AIC 按 (batch, m 分块, N 切片) 扁平区间生产 C 块
（单 tile 流水不变）；AIV 只累计自己 N 切片上的逐行最大值，写入工作区尾部的部分结果区；
以设备侧原子计数 + 轮询实现"全部就绪"判定，随后各块冗余地合成最终每 batch 分数并直写（幂等）。
零新增契约参数、零新增内核、只使用判题实证过的同步原语。

## Technical Context

**Language/Version**: AscendC（单文件 raw 内核 `kernel.asc`，CANN 9.0.0 工具链，host/device 混合编译）

**Primary Dependencies**: bisheng/ccec 编译流程、libascendcl（含既有使用的 `aclrtMemsetAsync`）；
本地测试框架 `main.asc`；判题契约入口 `run_kernel`

**Storage**: N/A（设备侧 GM：既有 32 槽 C 环工作区；本特性在其尾部追加部分结果区与就绪计数器）

**Testing**: 本地分层门禁（标准 15 例 + 244 例矩阵 + soak + 核数变体，资产已归档 `scripts/`）；
判题平台 15 个测试点（设备侧用时口径）

**Target Platform**: Ascend 910B（dav-2201），20 个 cube 核；CANN 9.0.0

**Project Type**: 单文件算子内核（提交物仅 `kernel.asc`）

**Performance Goals**: 目标形状类设备侧耗时 ≥2× 下降（P14 ~10.7×、P15 ~6.6× 差距显著收敛）；
其余测试点不变差（±2% 噪声内）；总分 25.7 → ≥27.7

**Constraints**: 内核内部启动参数维持 14 个（禁止新增）；不新增 `__global__` 内核；
不引入新的同步机制（仅用已有且判题实证过的 notify/wait/事件/原子写）；
受 UB(192KB)/L0C(128KB)/L1(512KB) 容量的既有约束

**Scale/Scope**: `kernel.asc` 约 500 行基线，本特性预计改动 120–180 行；
部分结果区 ≤ 194KB（触发条件下 batch×M ≤ 2432 元素/块）；合成计算量 ≤ 2432 元素/块

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 检查点 | 结论 |
|---|---|---|
| I. 判题实证唯一真理 | 每条机制均有判题/本地实证依据（M 并行、屏障、口径推定）；禁止探测式改动 | ✅ 通过 |
| II. 本地验证门禁 | 计划包含快门禁（seq15+子集）与全量回归（244+soak+核数）两阶段 | ✅ 通过 |
| III. 最小步进与可回退 | 单一可归因变化集；S9 为回退点；单次提交 | ✅ 通过 |
| IV. 正确性先于性能 | 15/15 保持为前置条件；收益按并行度机制归因；已知代价显式记录 | ✅ 通过 |
| V. 环境韧性与可追溯 | 版本标记 `[S11-TSPLIT]`；md5 核对；git 提交为持久化存储 | ✅ 通过 |

无违规项 ⇒ Complexity Tracking 表留空（无豁免需求）。

**Post-Design Re-check（Phase 1 之后）**: 设计产物（research / data-model / contracts / quickstart）
未引入新违规——对外契约零变化、同步原语限于既有实证要素清单、幂等合成与"就绪先于合成"
被固化为硬不变量 ⇒ **复检仍全部通过**。

## Project Structure

### Documentation (this feature)

```text
specs/001-naxis-split-combine/
├── plan.md              # 本文件
├── spec.md              # 需求规格
├── research.md          # Phase 0：决策与论证
├── data-model.md        # Phase 1：实体/布局/不变量
├── quickstart.md        # Phase 1：端到端验证指南
├── contracts/
│   └── kernel-entry.md  # 入口契约与内部启动契约（不变式）
└── checklists/
    └── requirements.md  # 规格质量检查清单
```

```text
（tasks.md 由 /speckit.tasks 生成，不在本命令范围）
```

### Source Code (repository root)

```text
kernel.asc                # 唯一提交物：run_kernel 契约入口 + AIC/AIV 内核（本特性改动点）
main.asc                  # 本地测试框架（非提交物）
scripts/                  # 门禁与资产：seq15.txt / chk.py / soak.py / matrix_all.txt / kp_cases.txt / npu_reset
build/                    # 本地构建与用例数据（input/ output/）
PLAN.md                   # 跨会话交接记录（判题事实与结论）
```

**Structure Decision**: 单文件内核工程（由赛题与判题契约决定），无多模块结构；
本特性仅在 `kernel.asc` 内改动（宿主分支 + AIC 外层循环 + AIV 外层循环与收尾合成）。

## Complexity Tracking

> 无宪法违规；无需豁免。

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| （无） | — | — |
