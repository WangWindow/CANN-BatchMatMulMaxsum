# Quickstart（更新版）: 性能差距诊断与改进框架 — 验证指南

**Feature**: 002-perf-gap-analysis | **Date**: 2026-10-14（第二轮更新）


验证三件事：诊断可信（三口径证据）、框架可判定（每条一轮可验）、行动项可落地（全量回归门槛）。

## 1. 诊断可信性验证（US1 — 已交付，保留复核口径）

```bash
# 判题数据：最近轮次逐点表（27.21）与榜首列对比
# 本地利用率：kscan/disc 用例 → 斜率(TFLOPS)与截距（固定域）分解
cd /tmp/opencode/s8repo/build && ./batch_matmul_max_sum_custom /tmp/opencode/kscan_cases.txt
# 手算校验：slope≈0.1µs/K? intercept≈99µs? disc(C翻倍K减半)不慢于 k2048?
```

**期望**：根因 A/D 的量级与实测同阶（±2× 内）；每条根因 ≥2 项证据。

## 2. 框架可判定性验证（US2 — 已交付）

对行动项 A2–A5 任取一条，按 plan.md 的「验证方法」执行，必须一轮内得有效/无效；
无验证方法的方向不得进入框架。契约见 [contracts/diagnosis-report.md](contracts/diagnosis-report.md)。

## 3. 行动项落地与全量回归协议（US3 起的统一门槛，宪法 v1.1.0）

```bash
cd /tmp/opencode/s8repo/build
# a) 构建（0 错误）后跑引导集：
./batch_matmul_max_sum_custom /tmp/opencode/kscan_cases.txt     # 喂数域 A/B
./batch_matmul_max_sum_custom /tmp/opencode/tx_cases.txt        # 装载路径 4 组合
./batch_matmul_max_sum_custom /tmp/opencode/nsd_cases.txt       # 触发面 32/48/64 切片
./batch_matmul_max_sum_custom /tmp/opencode/ns_cases.txt        # 触发面基准
./batch_matmul_max_sum_custom /tmp/opencode/t_cases.txt         # 全 1 定点复现
BMMS_CORE_OVERRIDE=4  ./batch_matmul_max_sum_custom /tmp/opencode/nsd_cases.txt
BMMS_CORE_OVERRIDE=32 ./batch_matmul_max_sum_custom /tmp/opencode/nsd_cases.txt
python3 /tmp/opencode/chk.py /tmp/opencode/nsd_cases.txt        # 值校验
# b) 全量回归（交付判题前必须全绿）：
./batch_matmul_max_sum_custom /tmp/opencode/seq15.txt           # 15/15
./batch_matmul_max_sum_custom /tmp/opencode/matrix_all.txt      # 244 矩阵
python3 /tmp/opencode/chk.py /tmp/opencode/matrix_all.txt       # 失败集合须与基线逐例一致
python3 /tmp/opencode/soak_ts.py 30                             # 触发路径 30×7
```

**通过判据**：0 编译错误；引导集全 PASS；seq15 15/15；矩阵失败集合与基线逐例一致（不得新增）；
soak 零 wrong/零 hang；核数变体（1/4/32）全过。
**捆绑提交原则**：单次判题提交须含 ≥15-20% 量级变化（噪声 ±5% 实证）。

## 4. 回退点

| 版本 | commit | 判题实证 |
|---|---|---|
| S13a1（当前基线） | `aed32b9` | 27.21（同分，含 A 装载顺序化） |
| S12f | `baa7ec7` | 27.21（触发解耦生效轮，P14 10.7×→3.7×） |
| S9 | `47dc041` | 25.70（保底回退） |

回退命令：`git checkout <commit> -- kernel.asc`（并核对 md5 与 `[Sxx-*]` 标记数）。
