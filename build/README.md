# build/ 测试资产与协议说明

## 验证协议（2026-10-14 修正）
- **单块（1-block）形状**在 <100µs 区间中位数不稳（实测 ±30-50%，随跑序/设备状态抖动）
  ⇒ 此类 A/B 与验收**必须"三次取最小"或 reps≥50**（源：mac2048 异常调查撤回结论）。
- 多块形状沿用 median-of-20；全量回归门槛见宪法 v1.1.0（seq15 + 244 矩阵 + soak + 核数变体）。

## 资产清单（关键）
- 用例集：`kscan_cases` / `tx_cases` / `disc_cases` / `nsd_cases` / `ns_cases` / `t_cases` /
  `s12_pts` / `seq15` / `matrix_all` / `mac_ladder`（金标在 `output/<name>/golden_y.bin`）
- 消融脚本：`../scripts/abl/abl_nr2.py`（AIV 侧）、`abl_nc.py`（Fixpipe）
- 核数变体：环境变量 `BMMS_CORE_OVERRIDE=1/4/20/32`
