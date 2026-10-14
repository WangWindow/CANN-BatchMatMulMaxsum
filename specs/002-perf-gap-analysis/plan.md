# Implementation Plan: 性能差距诊断与改进框架（对标榜首 5~10×）

**Branch**: `002-perf-gap-analysis` | **Date**: 2026-10-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-perf-gap-analysis/spec.md`

## Summary

以判题实测（S9/S11 逐点数据）与本地可测指标为唯一依据，对"当前实现 vs 榜首 5~10×"做结构化归因，
产出四类根因（计算单元空转 / C 数据 GM 往返 / N 轴并行触发缺失 / 固定开销）与
按"收益×置信度÷成本风险"排序的改进框架，并落地首项行动（tileSplit 触发与经济性门解耦）。
技术路径：复用既有单文件内核工程与验证门禁，不引入新框架。

## Technical Context

**Language/Version**: AscendC 单文件内核（`kernel.asc`，CANN 9.0.0）

**Primary Dependencies**: 既有构建/验证链路（main.asc 本地框架、scripts 门禁资产、判题平台）

**Storage**: N/A

**Testing**: 全量回归（seq15 + 244 矩阵 + soak + 核数变体）为交付唯一前置（2026-10-14 澄清取消快门禁）+ 判题逐点数据

**Target Platform**: Ascend 910B（dav-2201，20 cube 核）

**Project Type**: 单文件算子内核（提交物仅 `kernel.asc`）

**Performance Goals**: 缩小 P14/P15 的 10.7×/6.7×（触发解耦后 ≥2× 改善）；为后续大点（P8–P12）明确 ≤2 条结构性主线

**Constraints**: 15/15 零回归；一次一个可归因变化；不新增内核参数/内核；已否证方向不再投入；截止 10/17

**Scale/Scope**: 诊断覆盖 15 点 4 个形状族；框架 ≤5 条方向；首项行动 = 触发阈值改造 + 标定（~1 行级代码 + 2–3 条本地对照用例）

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| 原则 | 检查点 | 结论 |
|---|---|---|
| I. 判题实证唯一真理 | 全部根因以判题逐点数据 + 本地 TFLOPS/流量为证据；领先者结构不可观测处明确标注为假设 | ✅ |
| II. 本地验证门禁 | 首项行动交付前必须全量回归全绿（快门禁已取消，防子集漏检 c06/ns 级问题） | ✅ |
| III. 最小步进与可回退 | 单一变化集；回退点 47dc041（S9，25.7 分） | ✅ |
| IV. 正确性先于性能 | 15/15 为前置；收益按机制归因（触发缺失/空转/流量/固定开销） | ✅ |
| V. 环境韧性与可追溯 | 版本标记；判题结果与诊断证据归档 PLAN.md | ✅ |

无违规 ⇒ Complexity Tracking 留空。**Post-Design Re-check**：设计产物未引入新违规，复检通过。

## Project Structure

### Documentation (this feature)

```text
specs/002-perf-gap-analysis/
├── plan.md              # 本文件
├── spec.md              # 需求规格
├── research.md          # Phase 0：根因量化与方向决策
├── data-model.md        # Phase 1：差距族/根因/方向实体与度量口径
├── quickstart.md        # Phase 1：诊断与首项行动的验证指南
├── contracts/
│   └── diagnosis-report.md  # 诊断报告与框架表的字段契约
└── checklists/requirements.md
```

### Source Code (repository root)

```text
kernel.asc    # 本特性仅动宿主触发判定（tileSplit 解耦）+ 阈值（首项行动）；AIC/AIV 机制沿用 S11
build/        # ns 用例、smoke 清单、对照数据（不进 scripts/）
PLAN.md       # 判题事实、诊断结论、方向进展的归档处
```

**Structure Decision**: 沿用单文件内核工程；本特性以"数据+阈值"改动为主，结构性方向（②③）仅出方案不实施。

## Complexity Tracking

> 无宪法违规；无需豁免。
