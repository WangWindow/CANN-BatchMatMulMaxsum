# Tasks: 性能差距诊断与改进框架 — 第二批次（streamK 与小点）

**Input**: Design documents from `/specs/002-perf-gap-analysis/`（plan.md / spec.md / research.md / PLAN.md 冻结设计）
**Prerequisites**: T001–T014、E1–E3、D0、S14-A2 系列、S15-PL/Mmad/mac2048 结论均已完成（见"已完成档案"）
**Tests**: 测试内嵌于各任务的验证步骤（项目门禁协议：全量回归 = seq15 + 244 矩阵 + soak + 核数变体），不单列测试任务。
**Format**: `- [ ] [TaskID] [P?] [Story?] 描述 — 文件：路径`

---

## 已完成档案（[X] 汇总；明细见 git 历史）

- [X] T001–T014 诊断/框架/触发解耦（US1–US3；判题 27.21 达成）
- [X] E1 四消融（NA/NB/NR2/NC：喂数/AIV/Fixpipe 全分解）+ E2 架构候选评估 + E3 工具链评估
- [X] D0 E1-NR2（AIV 侧 9-15µs ⇒ D1 降权）
- [X] S14-A2 系列（A-Full-Load；判题 26.87，P8/P14 双历史最佳；门控 nShards>=2）
- [X] S15-PL 空结果（chunk 级重叠非瓶颈）+ Mmad 吞吐验证（~3.2 T MAC/s/核，20 核效率 65%）+ mac2048 撤回（测量方差）

## Phase 1: Setup（无新增任务——工程/链路/资产已就绪）

## Phase 2: Foundational（本批次前置，阻塞所有故事）

- [X] T015 验证协议修正：单块形状（1-block）A/B 一律"三次取最小或 reps≥50"，写入流程与 `build/README.md`（新建；源：mac2048 测量方差教训）
- [X] T016 基线冻结校验：从 git `a1f8bf0` 提取 kernel.asc 至 /tmp 沙箱构建 0 错并跑 kscan 留档（回退点可用性）— 文件：kernel.asc（校验）/ build/

## Phase 3: [US-streamK] K 轴切分（欠并行类）【Priority P1】

**Goal**: `workItemsAll*nShards < coreCount` 且 `k>=2048` 的形状进入 K 段并行；预期该类 1.7~3×。
**Independent Test**: (1,256,256,16384) 类本地 A/B ≥1.5×（三次取最小）+ nsd/ns 值对 + 全量回归绿。

- [ ] T017 [US-streamK] 宿主触发：`kernel.asc` 宿主段新增 K 轴资格（`workItemsAll*nShards<coreCount && k>=2048`），与 `tileEligible` 互斥（N 轴优先）；推导 `kSegs` 并纳入 workspace 尺寸计算 — 文件：kernel.asc（宿主段）
- [ ] T018 [US-streamK] AIC 段切分：flat 空间扩展为 (tile × kSeg)；段块仅计算其 K 段并 Fixpipe 至 workspace[tileIdx][seg] 槽（复用环槽寻址模式；保留分段就绪计数） — 文件：kernel.asc（RunCube flat 映射与 Fixpipe 段）
- [ ] T019 [US-streamK] AIV 三段归约：就绪计数目标改为 `totalTiles×kSegs`；每 tile 先逐 seg **ADD**（fp32）→ 行 ReduceMax → 复用 S12 合成（跨块 max + 对 m 求和） — 文件：kernel.asc（RunVector 合计段）
- [ ] T020 [P] [US-streamK] 用例资产：生成 (1,256,256,16384) 与 (1,128,128,8192) 金标用例 — 文件：build/input|output（生成脚本 scripts/gen_streamk_cases.py）
- [ ] T021 [US-streamK] 本地验证：A/B（三次取最小）+ 值校验 + nsd/ns/t1/c07×3 全绿 — 文件：kernel.asc
- [ ] T022 [US-streamK] 全量回归（矩阵零回归 + 核数 1/4/32 + seq15 + soak 30×7）→ 提交 + 判题三件套 — 文件：kernel.asc

## Phase 4: [US-tiny] 小点序言剥减【Priority P2】

**Goal**: P1–P4 类（全表最大比值 5.8–6.8×）削减启动/序言固定成本。
**Independent Test**: c05/c00/c01 类（reps≥50 或三次取最小）较基线 -20%+，且全回归绿。

- [ ] T023 [US-tiny] 固定成本分解探针：c05 类三连消融（去 NotifyEvent(15) / 去部分 InitBuffer / 去事件预置；诊断构建不入交付）— 文件：scripts/abl/ + /tmp 沙箱
- [ ] T024 [US-tiny] 按探针结果实施小点快速路径（候选：精简事件预置、合并 InitBuffer、条件化 handshake）— 文件：kernel.asc
- [ ] T025 [US-tiny] 验证与交付决策（同 T021/T022 协议；与 streamK 捆绑判题或独立，视 T026 漂移校准）

## Phase 5: Polish & 跨切面

- [ ] T026 判题漂移校准：同码连提两次（间隔 ≥1h）量化平台漂移，写入 PLAN.md 作为后续归因基准
- [ ] T027 每轮判题后按框架重排并联动 specs/002 文档（plan/research/data-model 增补）

---

## Dependencies & Execution Order

- **Foundational**: T015 → T016（T016 可与 T015 并行）
- **US-streamK**: T017 → T018（T020 可并行起草）→ T019 → T021 → T022 →（判题）
- **US-tiny**: T023 → T024 → T025；与 US-streamK 无文件级并行（同用 kernel.asc），开发可错峰、**设备验证必须串行**
- **Polish**: T026/T027 依赖各自判题结果

### Parallel Opportunities

- T016 ∥ T020（沙箱校验与用例生成互不依赖）
- T023 的探针消融可批量生成（scripts/abl/）后一次构建轮换测试

## Implementation Strategy

1. **MVP = T015–T022**（streamK 全链）：直击欠并行类（P8/P11/P15 嫌疑病因），单项预期 1.7–3×；
2. 小点批次（T023–T025）紧随，攻全表最大比值域；
3. 每个交付批次遵守：验证协议（含重复取最小）→ 全量回归绿 → 判题三件套；回退点 = `a1f8bf0`（A2d）/ `aed32b9`（27.21 保底）。
