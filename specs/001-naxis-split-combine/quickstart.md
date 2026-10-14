# Quickstart: N 轴并行切分与最大值合成 — 端到端验证指南

**Feature**: 001-naxis-split-combine | **Date**: 2026-10-14

本指南证明特性"可运行、可判题、可回退"。所有命令在仓库根目录或 `build/` 下执行；
资产引用见 [spec.md](spec.md)、[data-model.md](data-model.md)、[contracts/kernel-entry.md](contracts/kernel-entry.md)。

## 0. 前置条件

- 已完成实现并有本地构建产物（与判题同源工具链构建）
- 门禁资产存在：`scripts/seq15.txt`、`scripts/chk.py`、`scripts/soak.py`、
  `scripts/matrix_all.txt`、`scripts/npu_reset`
- 新增目标类用例 `ns1/ns2/ns3`（见 research.md R6）已生成输入与 FP64 金标

## 1. 单测目标形状（故事 1 的独立验证）

```bash
cd build
timeout 600 ./batch_matmul_max_sum_custom <ns_cases.txt>   # 输出每例中位耗时
python3 ../scripts/chk.py <ns_cases.txt>                    # 与 FP64 金标逐值比对
```

**期望**: 逐值一致（1e-4）；`ns1` 类耗时相比 S9 基线（同环境重跑基线二进制）≥2× 下降。

## 2. 快门禁（≤5 分钟，交付判题前必过）

```bash
cd build
timeout 300 ./batch_matmul_max_sum_custom ../scripts/seq15.txt >/dev/null 2>&1
python3 ../scripts/chk.py ../scripts/seq15.txt        # 期望 15/15
timeout 900 python3 ../scripts/run_matrix.py <smoke 子集> --batch 24 --timeout 240
```

**期望**: seq15 15/15；子集结果与基线逐例一致（含域外守卫拒绝项）。

## 3. 全量回归（后台，与判题等待并行）

```bash
cd build
timeout 3000 python3 ../scripts/run_matrix.py ../scripts/matrix_all.txt --batch 24 --timeout 240
timeout 2400 python3 ../scripts/soak.py 20
# 核数变体: BMMS_CORE_OVERRIDE in {1,4,20,32} 各跑 seq15
```

**期望**: 矩阵 162/82(域外)/0 挂死（与基线一致）；soak 6 用例 ×20 全稳；
核数 1/4/20/32 全部 15/15。

## 4. 判题（故事 1/2 的最终验证）

1. 核对交付三件套：`grep -c "S11-TSPLIT" kernel.asc`、`md5sum kernel.asc`；
2. 提交 `kernel.asc` 至判题平台；
3. 期望结果：**15/15**；目标形状对应测试点耗时显著下降；总分 ≥27.7。

## 5. 失败回退（故事 3）

```bash
git checkout 47dc041 -- kernel.asc   # 回退到上一判题实证版本（S9）
```

回退后重跑快门禁确认恢复基线，再把失败证据（逐点用时/错误分类）记入 PLAN.md。
